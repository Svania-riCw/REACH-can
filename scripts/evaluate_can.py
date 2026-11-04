#!/usr/bin/env python3
"""Robomimic Can rollout with actual action execution, detection, retreat and CFG."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
import hashlib
import json
from pathlib import Path
import time

from checkpoint_utils import PROJECT_ROOT,OPENPI_ROOT
import numpy as np
import torch
from error_detector import ErrorDetector
from eval_catalog import metadata_digest
from eval_policy import Policy
from eval_recovery import pose_error,retreat_command
from eval_sim_bridge import Simulation


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def implementation_digest():
    value=hashlib.sha256()
    for name in ('evaluate_can.py','eval_policy.py','eval_sim_worker.py','eval_sim_bridge.py',
                 'eval_recovery.py','eval_catalog.py','policy_diagnostics.py','checkpoint_utils.py',
                 'openpi_robomimic_adapter.py','error_detector.py','runtime_bootstrap.py','train_pi05.py'):
        value.update(name.encode());value.update((PROJECT_ROOT/'scripts'/name).read_bytes())
    value.update((PROJECT_ROOT/'configs/pi05_robomimic_config.py').read_bytes())
    for path in sorted((OPENPI_ROOT/'src/openpi/models_pytorch').glob('*.py')):
        value.update(str(path.name).encode());value.update(path.read_bytes())
    return value.hexdigest()


def run_identity(run):
    # A failed worker may resume on another identical GPU; retain the physical
    # device assignment as provenance, not as an experimental input.
    inputs={k:v for k,v in run.items() if k!='policy_cuda_visible_devices'}
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()


def checkpoint_digest(path,cache):
    path=Path(path).resolve();stat=path.stat()
    cache.mkdir(parents=True,exist_ok=True)
    file=cache/(hashlib.sha256(str(path).encode()).hexdigest()+'.json')
    identity=dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    if file.exists():
        saved=json.loads(file.read_text())
        if all(saved[k]==v for k,v in identity.items()):return saved['sha256']
    value=digest(path)
    tmp=file.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(dict(identity,sha256=value)));tmp.replace(file)
    return value


def build_recovery_prompt(mode,action_pred,last_executed_action):
    if mode=='full_plan':
        return np.asarray(action_pred,dtype=np.float32).copy()
    if mode=='repeat_last_action':
        action=np.asarray(last_executed_action,dtype=np.float32)
        return np.repeat(action[None,:],len(action_pred),axis=0)
    raise ValueError(f'Unknown negative prompt mode: {mode}')


def rollout(sim,policy,detector,initial,settings,episode_seed,trace,frame_callback=None):
    reset=sim.reset(initial['seed'],initial['state'])
    obs=reset['observation'];cached_chunk=None;previous_gripper=0.
    last_action_pred=None;prediction_seed=None
    after_retreat=False;executed_chunk_steps=0
    if frame_callback:frame_callback(obs)
    steps=recoveries=decisions=clipped=0
    success=bool(obs['success']);termination='success' if success else 'budget'
    started=time.perf_counter()
    def score(observation):
        if detector is None:return None
        return float(detector(observation)['score'].item())
    initial_score=score(obs)
    previous_triggered=detector is not None and initial_score>detector.threshold
    while not success and steps<settings['max_steps']:
        pending=None
        if cached_chunk and detector is not None:
            error=score(obs)
            rollback_chunk=cached_chunk
            triggered=error>detector.threshold
            trace.append(dict(step=steps,phase='detection',anomaly_score=error,triggered=triggered,
                              previous_triggered=bool(previous_triggered),after_retreat=after_retreat,
                              policy_call=decisions+1,executed_chunk_steps=executed_chunk_steps))
            if previous_triggered or triggered:
                prompt_mode=settings.get('negative_prompt_mode','full_plan')
                pending=build_recovery_prompt(prompt_mode,last_action_pred,rollback_chunk[-1]['action'])
                prompt_source={'full_plan':'last_normal_action_pred',
                               'repeat_last_action':'last_normal_action_repeated'}[prompt_mode]
            if previous_triggered:
                trace.append(dict(step=steps,phase='recovery_replan',after_retreat=after_retreat,
                                  negative_prompt_source=prompt_source,negative_prompt=pending.tolist(),
                                  prediction_noise_seed=prediction_seed))
            elif triggered:
                recoveries+=1
                trace.append(dict(step=steps,phase='retreat_start',chunk_steps=len(rollback_chunk),
                                  negative_prompt_mode=prompt_mode,negative_prompt_source=prompt_source,
                                  negative_prompt_steps=len(pending),
                                  negative_prompt=pending.tolist(),prediction_noise_seed=prediction_seed))
                for action_index in range(len(rollback_chunk)-1,-1,-1):
                    record=rollback_chunk[action_index]
                    target=record['before']
                    reached=False
                    for retreat_tick in range(settings['rollback_max_control_steps']):
                        distance,angle=pose_error(obs['eef_pose'],target)
                        if retreat_tick>0 and distance<=settings['rollback_position_tolerance'] and angle<=settings['rollback_rotation_tolerance']:
                            reached=True;break
                        if steps>=settings['max_steps']:break
                        reverse=retreat_command(obs['eef_pose'],target,sim.info,record['previous_gripper'])
                        result=sim.step(reverse);obs=result['observation'];steps+=1
                        if frame_callback:frame_callback(obs)
                        trace.append(dict(step=steps,phase='retreat',action=reverse.tolist(),
                                          rollback_action_index=action_index,
                                          anomaly_score=None,success=bool(obs['success'])))
                        if obs['success']:success=True;termination='success';break
                    distance,angle=pose_error(obs['eef_pose'],target)
                    reached=reached or (distance<=settings['rollback_position_tolerance'] and angle<=settings['rollback_rotation_tolerance'])
                    trace.append(dict(step=steps,phase='retreat_result',position_error=distance,
                                      rotation_error=angle,reached=reached,rollback_action_index=action_index))
                    if success:break
                    if not reached:
                        termination='rollback_failed' if steps<settings['max_steps'] else 'budget'
                        break
                if success:break
                if not reached:return dict(success=False,termination=termination,physical_steps=steps,
                    recoveries=recoveries,decisions=decisions,clipped_predictions=clipped,initial_score=initial_score,
                    seconds=time.perf_counter()-started)
                previous_gripper=rollback_chunk[0]['previous_gripper']
                previous_triggered=triggered;after_retreat=True
                continue
            previous_triggered=triggered

        after_retreat=False
        seed=(episode_seed+decisions*1009)%(2**63-1)
        prediction=policy.predict(obs,seed,pending,settings['scale'])
        plan=prediction['action']
        action_pred=np.asarray(prediction['action_pred'],dtype=np.float32)
        if action_pred.shape!=(policy.config.action_horizon,7) or not np.isfinite(action_pred).all():
            raise ValueError('Policy must return the complete finite [horizon,7] action prediction')
        used_negative=pending is not None;pending=None;clipped+=prediction['clipped_predictions'];decisions+=1
        if not used_negative:
            last_action_pred=action_pred.copy();prediction_seed=seed
        executed_chunk=[]
        for raw_action in plan[:settings['executed_steps']]:
            if steps>=settings['max_steps']:break
            before=obs
            result=sim.step(raw_action);obs=result['observation'];steps+=1
            if frame_callback:frame_callback(obs)
            event=dict(step=steps,phase='forward',action=raw_action.tolist(),noise_seed=seed,
                       guided=bool(used_negative and settings['scale']!=0),recovery_replan=used_negative,
                       success=bool(obs['success']))
            record=dict(action=raw_action.copy(),before=before['eef_pose'],after=obs['eef_pose'],
                        previous_gripper=previous_gripper)
            executed_chunk.append(record);previous_gripper=float(raw_action[-1])
            event['anomaly_score']=None
            trace.append(event)
            success=bool(obs['success'])
            if success:termination='success';break
        executed_chunk_steps=len(executed_chunk)
        if executed_chunk and not used_negative:
            cached_chunk=executed_chunk
            trace.append(dict(step=steps,phase='normal_cache_update',chunk_steps=len(cached_chunk),
                              prediction_noise_seed=prediction_seed,
                              action_pred_sha256=hashlib.sha256(last_action_pred.tobytes()).hexdigest()))
    return dict(success=success,termination=termination,physical_steps=steps,recoveries=recoveries,
                decisions=decisions,clipped_predictions=clipped,initial_score=initial_score,
                seconds=time.perf_counter()-started)


def select_episodes(catalog, count, start=0, end=None):
    end = count if end is None else end
    if not 0 <= start < end <= count <= len(catalog['episodes']):
        raise ValueError('Require 0 <= episode-start < episode-end <= rollouts <= catalog size')
    return list(enumerate(catalog['episodes'][start:end], start=start))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--detector',help='Omit for no-detector controls; S=0 with a detector still retreats/resamples')
    parser.add_argument('--catalog',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--protocol',default=str(PROJECT_ROOT/'configs/evaluation_can.json'))
    parser.add_argument('--rollouts',type=int)
    parser.add_argument('--episode-start',type=int,default=0,help='First global catalog index in this shard')
    parser.add_argument('--episode-end',type=int,help='Exclusive global catalog index; full rollout count stays unchanged')
    parser.add_argument('--max-steps',type=int,help='Diagnostic override; recorded as a smoke run')
    parser.add_argument('--scale',type=float,default=0.)
    parser.add_argument('--negative-prompt-mode',choices=('full_plan','repeat_last_action'),default='full_plan')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--video',action='store_true',help='Save concatenated two-camera MP4s')
    parser.add_argument('--disable-action-conditioning',action='store_true',
                        help='Keep all weights and bypass prompt projection; recovery requires S=0')
    parser.add_argument('--threshold-override',type=float,help='Smoke tests only, never part of paper-comparable results')
    args=parser.parse_args()
    spec=json.loads(Path(args.protocol).read_text())
    if not np.isfinite(args.scale) or args.scale<0:parser.error('scale must be finite and nonnegative')
    if args.scale not in spec['guidance_scales']:parser.error('Scale not in the published sweep')
    if not args.detector and args.scale!=0:parser.error('Guidance requires a detector/failed history')
    if args.disable_action_conditioning and args.scale!=0:
        parser.error('Action-conditioning ablation requires S=0; negative guidance needs the projection')
    count=args.rollouts if args.rollouts is not None else spec['rollouts_per_checkpoint']
    if count<1:parser.error('rollouts must be positive')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads(Path(args.catalog).read_text())
    if count>len(catalog['episodes']):parser.error('Catalog has too few common initial states')
    try:selected=select_episodes(catalog,count,args.episode_start,args.episode_end)
    except ValueError as error:parser.error(str(error))
    policy=Policy(args.checkpoint,args.device,spec['denoising_steps'],
                  disable_action_conditioning=args.disable_action_conditioning)
    if args.detector and not policy.conditional:raise ValueError('Recovery requires a conditional checkpoint')
    metadata=policy.saved.get('metadata') or catalog['metadata']
    if metadata_digest(metadata)!=catalog['metadata_sha256']:raise ValueError('Environment metadata differs from common catalog')
    detector=ErrorDetector(args.detector,policy.device) if args.detector else None
    if args.threshold_override is not None:
        if detector is None or not np.isfinite(args.threshold_override):parser.error('Need a detector and finite override')
        detector.threshold=args.threshold_override
    settings=dict(spec,max_steps=args.max_steps if args.max_steps is not None else spec['horizons'][policy.saved['dataset_type']],
                  scale=args.scale,negative_prompt_mode=args.negative_prompt_mode)
    if settings['max_steps']<1:parser.error('max-steps must be positive')
    run=dict(checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=checkpoint_digest(args.checkpoint,output.parent/'checkpoint_hash_cache'),
        checkpoint_step=policy.payload['step'],training_seed=policy.saved['seed'],
        policy_mode=policy.saved['policy_mode'],policy_dataset=policy.saved['dataset_type'],
        action_conditioning_enabled=policy.action_conditioning_enabled,
        task_prompt=policy.saved['prompt'],
        detector=str(Path(args.detector).resolve()) if args.detector else None,
        detector_sha256=digest(args.detector) if args.detector else None,
        threshold=detector.threshold if detector else None,scale=args.scale,
        negative_prompt_mode=args.negative_prompt_mode,
        catalog_sha256=digest(args.catalog),protocol=spec,rollouts=count,max_steps=settings['max_steps'],
        smoke=bool(args.threshold_override is not None or args.max_steps is not None or count!=spec['rollouts_per_checkpoint']),
        exact_paper_reproduction=False,implementation_sha256=implementation_digest(),video=args.video,
        policy_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        normalization_sha256=hashlib.sha256(json.dumps(policy.stats,sort_keys=True).encode()).hexdigest(),
        torch_version=torch.__version__,numpy_version=np.__version__)
    if len(selected)!=count:
        run['episode_range']=[selected[0][0],selected[-1][0]+1]
    identity=run_identity(run)
    manifest=output/'run.json'
    attempts=[]
    if manifest.exists():
        if not args.resume:raise FileExistsError('Output already exists; use a new folder or --resume')
        previous=json.loads(manifest.read_text())
        if previous['fingerprint']!=identity:raise ValueError('Cannot resume changed evaluation inputs')
        attempts=previous.get('execution_attempts',[])
    attempts.append(dict(timestamp=time.time(),pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    manifest.write_text(json.dumps(dict(run,fingerprint=identity,execution_attempts=attempts),indent=2,allow_nan=False))
    completed=[]
    results_file=output/'episodes.jsonl'
    if results_file.exists():
        for line in results_file.read_text().splitlines():completed.append(json.loads(line))
    done_ids={r['episode_id'] for r in completed}
    if len(done_ids)!=len(completed):raise ValueError('Duplicate evaluation episodes')
    if not done_ids <= {e['id'] for _,e in selected}:raise ValueError('Unexpected completed episode IDs')
    with Simulation(metadata,output/'simulator.log') as sim:
        if sim.info['environment']!=spec['environment']:raise ValueError('Wrong task')
        for field,key in [('robosuite','robosuite_version'),('robomimic','robomimic_version'),('mujoco','mujoco_version'),('control_freq','control_freq')]:
            if sim.info[field]!=spec[key]:raise ValueError(f'Unexpected simulator {field}: {sim.info[field]}')
        (output/'environment.json').write_text(json.dumps(sim.info,indent=2))
        for i,initial in selected:
            if initial['id'] in done_ids:continue
            trace=[]
            seed=policy.saved['seed']*1_000_000+initial['seed']
            writer=None
            if args.video:
                import imageio.v2 as imageio
                writer=imageio.get_writer(output/f'episode_{i:03d}.mp4',fps=20,macro_block_size=1)
            try:
                callback=(lambda obs:writer.append_data(np.concatenate([obs['agentview_image'],obs['robot0_eye_in_hand_image']],axis=1))) if writer else None
                result=rollout(sim,policy,detector,initial,settings,seed,trace,callback)
            finally:
                if writer:writer.close()
            row=dict(result,episode_id=initial['id'],environment_seed=initial['seed'])
            (output/f'episode_{i:03d}_trace.json').write_text(json.dumps(trace,allow_nan=False))
            with results_file.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            completed.append(row)
            print(f'Episode {i+1}/{count}: success={row["success"]}, steps={row["physical_steps"]}, '
                  f'recoveries={row["recoveries"]}, reason={row["termination"]}',flush=True)
    summary=dict(fingerprint=identity,episodes=len(completed),successes=sum(r['success'] for r in completed),
        success_rate=sum(r['success'] for r in completed)/len(completed),
        mean_steps=float(np.mean([r['physical_steps'] for r in completed])),
        mean_recoveries=float(np.mean([r['recoveries'] for r in completed])),
        smoke=run['smoke'],run=str(manifest.resolve()))
    if 'episode_range' in run:
        summary.update(episode_range=run['episode_range'],target_rollouts=count,complete_evaluation=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

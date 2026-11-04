#!/usr/bin/env python3
"""Plan, execute on disjoint GPUs, and aggregate the checkpoint/seed sweep."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from checkpoint_utils import PROJECT_ROOT
from dataset_paths import dataset_paths
from evaluate_can import implementation_digest


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def build(args):
    spec=json.loads((PROJECT_ROOT/'configs/evaluation_can.json').read_text())
    jobs=[];unavailable=[]
    for kind in args.datasets:
        for seed in args.seeds:
            for mode in ('baseline','conditional'):
                folder=args.artifacts/f'policies/can_{kind}/{mode}/seed_{seed}'
                if not (folder/'final_model.pt').exists():
                    unavailable.append(dict(dataset=kind,seed=seed,mode=mode));continue
                checkpoints=sorted((folder/'checkpoints').glob('step_*.pt')) if args.checkpoints=='all' else []
                if args.checkpoints=='all' and (folder/'best_validation.pt').is_file():
                    checkpoints.append(folder/'best_validation.pt')
                checkpoints.append(folder/'final_model.pt')
                detectors=[(None,0.)]
                if mode=='conditional':
                    for detector in (['ph','mh','paired'] if kind=='paired' else ['mh','ph']):
                        detectors.extend((detector,scale) for scale in spec['guidance_scales'])
                for checkpoint in checkpoints:
                    for detector,scale in detectors:
                        detector_root=getattr(args,'detector_artifacts',args.artifacts)
                        detector_path=detector_root/f'detectors/can_{detector}/seed_{seed}/detector.pt' if detector else None
                        name=f'{kind}_{mode}_ed_{detector or "none"}_S{scale}_seed_{seed}_{checkpoint.stem}'
                        output=args.output/'cells'/name
                        command=[os.path.abspath(sys.executable),str(PROJECT_ROOT/'scripts/evaluate_can.py'),
                            '--checkpoint',str(checkpoint),'--catalog',str(args.catalog),'--output',str(output),
                            '--scale',str(scale),'--resume']
                        if detector_path:command+=['--detector',str(detector_path)]
                        job=dict(id=name,dataset=kind,seed=seed,mode=mode,detector_dataset=detector,scale=scale,
                            checkpoint=str(checkpoint),detector=str(detector_path) if detector_path else None,
                            output=str(output),command=command,implementation_sha256=implementation_digest(),
                            protocol_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())
                        job['fingerprint']=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
                        jobs.append(job)
    return dict(protocol=spec,checkpoint_scope=args.checkpoints,requested_training_seeds=args.seeds,
                unavailable_models=unavailable,catalog=str(args.catalog),jobs=jobs)


def execute(plan,output,gpus,wait_seconds):
    if plan['unavailable_models']:raise ValueError('Some requested models are not trained; explicitly select available --seeds')
    pending=list(plan['jobs']);active={};completed=[];failed=[]
    lock=threading.Lock();stop=threading.Event();started=time.monotonic()
    status=output/'status.json'
    def save():atomic(status,dict(pid=os.getpid(),pending=len(pending),active=active,
        completed=completed,failed=failed,status='failed' if failed else ('completed' if not pending and not active else 'running')))
    def worker(gpu):
        while not stop.is_set():
            job=None
            with lock:
                if not pending:return
                for candidate in pending:
                    if Path(candidate['checkpoint']).exists() and (not candidate['detector'] or Path(candidate['detector']).exists()) and Path(plan['catalog']).exists():
                        job=candidate;pending.remove(candidate);break
                if job is None:
                    if time.monotonic()-started>=wait_seconds:
                        failed.append(dict(error='Timed out waiting for catalog/detector artifacts'));stop.set();save();return
                else:
                    active[gpu]=dict(id=job['id'],status='starting');save()
            if job is None:time.sleep(5);continue
            folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
            marker=folder/'job_record.json';summary=folder/'summary.json'
            if marker.exists() and summary.exists() and json.loads(marker.read_text())['fingerprint']==job['fingerprint']:
                saved=json.loads(summary.read_text())
                if not saved['smoke'] and saved['episodes']==plan['protocol']['rollouts_per_checkpoint']:
                    with lock:completed.append(job['id']);active.pop(gpu);save()
                    continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,EVAL_RENDER_DEVICE=gpu,MUJOCO_EGL_DEVICE_ID=gpu,
                JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'driver.log').open('a') as log:
                process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,cwd=PROJECT_ROOT,
                                         env=env,start_new_session=True)
                with lock:active[gpu].update(pid=process.pid,status='running');save()
                code=process.wait()
            with lock:
                active.pop(gpu)
                if code or not summary.exists():
                    failed.append(dict(id=job['id'],returncode=code));stop.set()
                else:atomic(marker,job);completed.append(job['id'])
                save()
    save()
    def interrupt(signum,frame):
        stop.set()
        with lock:
            for entry in active.values():
                if 'pid' in entry:
                    try:os.killpg(entry['pid'],signal.SIGTERM)
                    except ProcessLookupError:pass
        raise KeyboardInterrupt
    old_handler=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,interrupt)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,gpu) for gpu in gpus]
            try:
                for future in futures:future.result()
            except BaseException:
                interrupt(signal.SIGTERM,None)
    finally:
        signal.signal(signal.SIGTERM,old_handler)
    return not failed


def summarize(plan,output):
    groups={};missing=[];accepted=set();expected={}
    for job in plan['jobs']:
        key=(job['dataset'],job['mode'],job['detector_dataset'],job['scale'])
        expected.setdefault(key,set()).add(job['id'])
        folder=Path(job['output']);summary=folder/'summary.json';record=folder/'job_record.json'
        if not summary.exists() or not record.exists():missing.append(job['id']);continue
        if json.loads(record.read_text())['fingerprint']!=job['fingerprint']:raise ValueError('Evaluation provenance mismatch')
        result=json.loads(summary.read_text())
        if result['smoke'] or result['episodes']!=plan['protocol']['rollouts_per_checkpoint']:
            missing.append(job['id']);continue
        accepted.add(job['id'])
        groups.setdefault(key,{}).setdefault(job['seed'],[]).append(dict(checkpoint=job['checkpoint'],**result))
    import numpy as np
    rows=[]
    for key,seeds in groups.items():
        per_seed=[]
        for seed,values in sorted(seeds.items()):
            best=max(values,key=lambda v:v['success_rate'])
            final=next((v['success_rate'] for v in values if Path(v['checkpoint']).name=='final_model.pt'),None)
            per_seed.append(dict(seed=seed,best_checkpoint=best['checkpoint'],best_success_rate=best['success_rate'],
                                 final_success_rate=final,evaluated_checkpoints=len(values),
                                 mean_available_checkpoint_success=float(np.mean([v['success_rate'] for v in values])),
                                 checkpoint_results=values))
        complete=set(seeds)==set(plan['protocol']['training_seeds'])
        sweep_complete=expected[key]<=accepted
        publish=complete and sweep_complete
        bests=[v['best_success_rate'] for v in per_seed]
        rows.append(dict(dataset=key[0],mode=key[1],detector_dataset=key[2],scale=key[3],per_seed=per_seed,
            complete_three_seeds=complete,complete_checkpoint_sweep=sweep_complete,
            paper_caption_mean_seed_best=float(np.mean(bests)) if publish else None,
            population_std=float(np.std(bests)) if publish else None,sample_std=float(np.std(bests,ddof=1)) if publish else None,
            paper_prose_max_across_seeds=max(bests) if publish else None,
            mean_final_success=float(np.mean([v['final_success_rate'] for v in per_seed]))
                if complete and all(v['final_success_rate'] is not None for v in per_seed) else None))
    report=dict(rows=rows,missing_jobs=missing,unavailable_models=plan['unavailable_models'],
                checkpoint_scope=plan['checkpoint_scope'],exact_paper_reproduction=False,
                note='Incomplete seed sets have no three-seed aggregate. Raw results, final and best checkpoints are retained; caption/prose aggregation differ in REACH.')
    atomic(output/'aggregate.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run','summarize'])
    parser.add_argument('--datasets',nargs='+',choices=['paired','mh'],default=['paired'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--checkpoints',choices=['all','final'],default='all')
    parser.add_argument('--gpus',nargs='+',default=[str(i) for i in range(8)])
    parser.add_argument('--wait-seconds',type=float,default=0)
    parser.add_argument('--artifacts',type=Path,default=PROJECT_ROOT/'outputs/pi05_full_sft_physical7_v4')
    parser.add_argument('--detector-artifacts',type=Path,default=PROJECT_ROOT/'outputs/paper_can',
                        help='Reuse already trained independent AEs; explicitly select a new root if retraining them')
    parser.add_argument('--catalog',type=Path,default=PROJECT_ROOT/'outputs/evaluation/can_initial_states_50.json')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/evaluation/pi05_full_sft_physical7_v4_paired_sweep')
    args=parser.parse_args()
    args.output=args.output.expanduser().resolve()
    args.artifacts=args.artifacts.expanduser().resolve()
    args.detector_artifacts=args.detector_artifacts.expanduser().resolve()
    args.catalog=args.catalog.expanduser().resolve()
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or any(not g.isdigit() for g in args.gpus):parser.error('Need unique numeric GPU IDs')
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or args.wait_seconds<0:parser.error('Invalid seeds or wait duration')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/'matrix.json'
    if args.action=='summarize':
        report=summarize(json.loads(manifest.read_text()),args.output)
        print(f'{len(report["rows"])} aggregate rows; {len(report["missing_jobs"])} pending cells');return
    plan=build(args)
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError('Output contains a different matrix; choose another folder')
    atomic(manifest,plan)
    print(f'{len(plan["jobs"])} checkpoint/configuration cells; {len(plan["unavailable_models"])} unavailable models',flush=True)
    if args.action=='plan':return
    import fcntl
    with (args.output/'driver.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        success=execute(plan,args.output,args.gpus,args.wait_seconds)
    summarize(plan,args.output)
    if not success:sys.exit(1)


if __name__=='__main__':main()

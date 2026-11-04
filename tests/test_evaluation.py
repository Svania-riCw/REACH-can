"""Check action units, physical rollback, failed history and equal step budgets."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from contextlib import contextmanager
import tempfile
import json
import threading
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from eval_policy import Policy,denormalize_actions,normalize_action_prompt
from eval_recovery import invert_se3,pose_error,retreat_command,failed_prompt
from evaluate_can import build_recovery_prompt,rollout,run_identity
from openpi_robomimic_adapter import normalize
from evaluation_matrix import summarize,execute,build


CONTROLLER=dict(output_min=[-.05]*3+[-.5]*3,output_max=[.05]*3+[.5]*3,input_min=-1,input_max=1)


class FakeSim:
    info=CONTROLLER
    def __init__(self,freeze=False):self.reset_count=0;self.freeze=freeze
    def obs(self):return dict(eef_pose=self.pose.copy(),success=False)
    def reset(self,seed,state):
        self.reset_count+=1;self.pose=np.eye(4);self.actions=[]
        return dict(observation=self.obs())
    def step(self,action):
        self.actions.append(np.asarray(action).copy())
        if not self.freeze or np.asarray(action)[0]>0:
            self.pose[:3,3]+=np.asarray(action[:3])*.05
            self.pose[:3,:3]=Rotation.from_rotvec(np.asarray(action[3:6])*.5).as_matrix()@self.pose[:3,:3]
        return dict(observation=self.obs())


class FakePolicy:
    config=SimpleNamespace(action_horizon=50)
    def __init__(self,first_chunk=None):self.prompts=[];self.observations=[];self.first_chunk=first_chunk;self.predictions=[]
    def predict(self,obs,seed,failed,scale):
        self.prompts.append(failed)
        self.observations.append(obs['eef_pose'].copy())
        result=np.zeros((50,7))
        if len(self.prompts)==1:
            if self.first_chunk is None:
                result[:,0]=.9;result[0,0]=.2;result[0,-1]=.7
            else:result[:len(self.first_chunk)]=self.first_chunk
        self.predictions.append(result.copy())
        return dict(action=np.clip(result,-1,1),action_pred=result,
                    clipped_predictions=int(np.count_nonzero(np.abs(result)>1)))


class OnceDetector:
    threshold=.5
    def __init__(self):self.calls=0
    def __call__(self,obs):
        self.calls+=1
        return dict(score=torch.tensor(float(self.calls==2)))


def settings():
    return dict(max_steps=16,executed_steps=4,scale=0.,rollback_scope='last_executed_chunk',negative_prompt_source='last_full_action_pred',
                rollback_max_control_steps=2,rollback_position_tolerance=.001,rollback_rotation_tolerance=.001)


class EvaluationTests(unittest.TestCase):
    def test_resume_can_change_gpu_but_not_weights_or_protocol(self):
        run=dict(checkpoint_sha256='weights-A',protocol=dict(executed_steps=8),policy_cuda_visible_devices='0')
        original=run_identity(run)
        self.assertEqual(original,run_identity(dict(run,policy_cuda_visible_devices='3')))
        self.assertNotEqual(original,run_identity(dict(run,checkpoint_sha256='weights-B')))
        self.assertNotEqual(original,run_identity(dict(run,protocol=dict(executed_steps=4))))
        self.assertNotEqual(run_identity(dict(run,action_conditioning_enabled=True)),
                            run_identity(dict(run,action_conditioning_enabled=False)))

    def test_se3_inverse_and_world_osc_convention(self):
        current=np.eye(4);current[:3,:3]=Rotation.from_euler('xyz',[.3,.2,-.1]).as_matrix()
        current[:3,3]=[.2,.3,.4]
        target=current.copy();target[:3,3]+=[.01,-.02,.03]
        target[:3,:3]=Rotation.from_rotvec([0,0,.1]).as_matrix()@current[:3,:3]
        relative=invert_se3(current)@target
        np.testing.assert_allclose(relative@invert_se3(relative),np.eye(4),atol=1e-12)
        np.testing.assert_allclose(target@invert_se3(relative),current,atol=1e-12)
        command=retreat_command(current,target,CONTROLLER,.7)
        np.testing.assert_allclose(command,[.2,-.4,.6,0,0,.2,.7],atol=1e-12)
        self.assertNotEqual(float(invert_se3(relative)[0,3]),-float(relative[0,3]))

    def test_normalization_roundtrip_and_prompt_padding(self):
        stats=dict(min=[-2,1,2,0,-1,-3,-1],max=[2,5,2,6,1,3,1])
        raw=np.array([[.5,2,2,1,.3,1,-.7]],dtype=np.float32)
        normalized=np.pad(normalize(raw,stats),((0,0),(0,25)))
        np.testing.assert_allclose(denormalize_actions(normalized,stats),raw,atol=1e-6)
        hint=failed_prompt(raw,50)
        self.assertEqual(hint.shape,(50,7))
        np.testing.assert_allclose(hint,np.repeat(raw,50,axis=0))

    def test_recovery_prompt_modes_use_cached_plan_or_last_executed_action(self):
        plan=np.arange(350,dtype=np.float32).reshape(50,7)
        executed=np.array([.2,-.3,.4,-.5,.6,-.7,1.],dtype=np.float32)
        full_plan=build_recovery_prompt('full_plan',plan,executed)
        repeated=build_recovery_prompt('repeat_last_action',plan,executed)
        np.testing.assert_array_equal(full_plan,plan)
        np.testing.assert_array_equal(repeated,np.repeat(executed[None,:],50,axis=0))
        self.assertIsNot(full_plan,plan)
        with self.assertRaisesRegex(ValueError,'Unknown negative prompt mode'):
            build_recovery_prompt('invalid',plan,executed)

    def test_whole_chunk_retreats_before_replanning_with_full_prediction_prompt(self):
        sim=FakeSim();policy=FakePolicy();trace=[]
        result=rollout(sim,policy,OnceDetector(),dict(seed=1,state={}),settings(),123,trace)
        self.assertEqual(sim.reset_count,1)
        self.assertEqual(result['physical_steps'],16)
        self.assertEqual(result['recoveries'],1)
        self.assertEqual(result['decisions'],3)
        self.assertEqual(len(sim.actions),16)
        detections=[event for event in trace if event['phase']=='detection']
        self.assertEqual(len(detections),3)
        self.assertEqual(detections[0]['step'],4)
        self.assertTrue(detections[0]['triggered'])
        self.assertEqual(detections[0]['executed_chunk_steps'],4)
        self.assertFalse(detections[1]['triggered'])
        self.assertEqual(detections[1]['step'],8)
        self.assertTrue(detections[1]['after_retreat'])
        self.assertTrue(detections[1]['previous_triggered'])
        self.assertEqual(detections[2]['step'],12)
        self.assertTrue(any(e['phase']=='retreat' for e in trace))
        self.assertFalse(any(e.get('guided',False) for e in trace))
        self.assertTrue(any(e.get('recovery_replan',False) for e in trace))
        np.testing.assert_allclose(policy.prompts[1],policy.predictions[0],atol=1e-7)
        self.assertIsNot(policy.prompts[1],policy.predictions[0])
        self.assertIsNone(policy.prompts[2])
        np.testing.assert_allclose(sim.actions[4],[-.9,0,0,0,0,0,0])
        np.testing.assert_allclose(sim.actions[7],[-.2,0,0,0,0,0,0],atol=1e-12)
        np.testing.assert_allclose(policy.observations[1],np.eye(4),atol=1e-12)
        targets=[event for event in trace if event['phase']=='retreat_result']
        self.assertEqual([event['rollback_action_index'] for event in targets],[3,2,1,0])
        self.assertTrue(all(event['reached'] for event in targets))
        start=next(event for event in trace if event['phase']=='retreat_start')
        self.assertEqual(start['chunk_steps'],4)
        self.assertEqual(start['negative_prompt_steps'],50)
        self.assertEqual(start['negative_prompt_source'],'last_normal_action_pred')
        np.testing.assert_allclose(start['negative_prompt'],policy.predictions[0],atol=1e-7)

    def test_eight_step_chunk_restores_noncommuting_poses_and_gripper(self):
        actions=np.array([[.2,.1,0,.6,0,0,.7],[0,.3,0,0,.4,0,-.6],
                          [.1,0,.4,0,0,.5,.2],[-.2,.1,0,.3,.2,0,.8],
                          [.3,-.2,.1,0,.4,.2,-.4],[0,0,-.2,-.3,0,.2,.1],
                          [.1,.3,0,.2,-.4,0,.5],[0,-.1,.2,0,.3,-.2,-.7]])
        sim=FakeSim();policy=FakePolicy(actions);trace=[]
        spec=dict(settings(),executed_steps=8,max_steps=24)
        result=rollout(sim,policy,OnceDetector(),dict(seed=1,state={}),spec,123,trace)
        self.assertEqual(result['physical_steps'],24)
        self.assertEqual(result['decisions'],2)
        self.assertEqual(sim.reset_count,1)
        np.testing.assert_allclose(policy.observations[1],np.eye(4),atol=1e-12)
        np.testing.assert_allclose(policy.prompts[1],policy.predictions[0],atol=1e-7)
        np.testing.assert_allclose(np.asarray(sim.actions)[8:16,-1],np.r_[0.,actions[:-1,-1]][::-1])
        targets=[event for event in trace if event['phase']=='retreat_result']
        self.assertEqual([event['rollback_action_index'] for event in targets],list(range(7,-1,-1)))
        self.assertTrue(all(event['reached'] for event in targets))

    def test_full_prediction_prompt_keeps_unexecuted_unclipped_suffix(self):
        prediction=np.zeros((50,7));prediction[:4,0]=.2
        prediction[4:,0]=np.linspace(1.1,2.,46)
        sim=FakeSim();policy=FakePolicy(prediction)
        rollout(sim,policy,OnceDetector(),dict(seed=1,state={}),settings(),123,[])
        np.testing.assert_allclose(policy.prompts[1],prediction,atol=1e-7)
        self.assertGreater(policy.prompts[1][-1,0],1.)
        self.assertTrue(np.all(np.abs(np.asarray(sim.actions))<=1))

    def test_repeat_last_action_prompt_uses_last_executed_action(self):
        sim=FakeSim();policy=FakePolicy();trace=[];spec=settings()
        spec['negative_prompt_mode']='repeat_last_action'
        rollout(sim,policy,OnceDetector(),dict(seed=1,state={}),spec,123,trace)
        start=next(event for event in trace if event['phase']=='retreat_start')
        expected=np.repeat(np.asarray(sim.actions[3])[None,:],50,axis=0)
        np.testing.assert_allclose(start['negative_prompt'],expected,atol=1e-7)
        self.assertEqual(start['negative_prompt_mode'],'repeat_last_action')
        self.assertEqual(start['negative_prompt_source'],'last_normal_action_repeated')

    def test_full_prompt_normalizes_each_timestep_and_zero_pads_unused_channels(self):
        stats=dict(q01=[-2.]*7,q99=[4.]*7)
        prediction=np.arange(350,dtype=np.float32).reshape(50,7)/100
        normalized=normalize_action_prompt(prediction,stats,50,32)
        self.assertEqual(normalized.shape,(50,32))
        expected=2*(prediction.astype(np.float64)+2)/(6+1e-6)-1
        np.testing.assert_allclose(normalized[:,:7],expected,atol=1e-7)
        np.testing.assert_array_equal(normalized[:,7:],0)
        with self.assertRaises(ValueError):normalize_action_prompt(prediction[:8],stats,50,32)
        prediction[0,0]=np.nan
        with self.assertRaises(ValueError):normalize_action_prompt(prediction,stats,50,32)

    def test_policy_returns_raw_prediction_separately_from_clipped_execution(self):
        policy=Policy.__new__(Policy)
        policy.conditional=True;policy.action_conditioning_enabled=True;policy.device=torch.device('cpu')
        policy.config=SimpleNamespace(action_horizon=50,action_dim=32)
        policy.stats={'action':dict(min=[-2.]*7,max=[2.]*7)}
        policy.base=torch.zeros(1,1,32);policy.num_steps=1
        sampled=torch.zeros(1,50,32);sampled[0,:,0]=2.
        policy.model=SimpleNamespace(sample_actions=lambda *args,**kwargs:sampled)
        with patch.object(policy,'observation',return_value=None):prediction=policy.predict({},123)
        self.assertEqual(prediction['action_pred'].shape,(50,7))
        np.testing.assert_allclose(prediction['action_pred'][:,0],4.)
        np.testing.assert_allclose(prediction['action'][:,0],1.)
        self.assertEqual(prediction['clipped_predictions'],50)

    def test_policy_guidance_receives_normalized_full_plan_with_zero_padding(self):
        policy=Policy.__new__(Policy)
        policy.conditional=True;policy.action_conditioning_enabled=True;policy.device=torch.device('cpu')
        policy.config=SimpleNamespace(action_horizon=50,action_dim=32)
        policy.stats={'action':dict(q01=[-2.]*7,q99=[4.]*7)}
        policy.base=torch.zeros(1,1,32);policy.num_steps=1
        conditions=[]
        @contextmanager
        def condition_scope(condition):
            conditions.append(condition.clone())
            yield
        policy.model=SimpleNamespace(_condition_scope=condition_scope,
                                     denoise_step=lambda *args:torch.zeros_like(args[-2]))
        failed=np.arange(350,dtype=np.float32).reshape(50,7)/100
        with patch.object(policy,'observation',return_value=None),patch('eval_policy.build_prefix',return_value=()):
            policy.predict({},123,failed,2.5)
        self.assertEqual(len(conditions),2)
        torch.testing.assert_close(conditions[0],torch.zeros(1,50,32))
        self.assertEqual(conditions[1].shape,(1,50,32))
        expected=2*(failed.astype(np.float64)+2)/(6+1e-6)-1
        np.testing.assert_allclose(conditions[1][0,:,:7].numpy(),expected,atol=1e-7)
        torch.testing.assert_close(conditions[1][0,:,7:],torch.zeros(50,25))

    def test_author_previous_flag_replans_without_refreshing_normal_cache(self):
        class RepeatedDetector(OnceDetector):
            def __call__(self,obs):
                self.calls+=1
                return dict(score=torch.tensor(float(self.calls>1)))
        policy=FakePolicy();trace=[]
        result=rollout(FakeSim(),policy,RepeatedDetector(),dict(seed=1,state={}),
                       dict(settings(),max_steps=20,scale=2.5),123,trace)
        self.assertEqual(result['recoveries'],1)
        self.assertEqual(len(policy.prompts),4)
        np.testing.assert_allclose(policy.prompts[1],policy.predictions[0],atol=1e-7)
        for prompt in policy.prompts[2:]:
            np.testing.assert_allclose(prompt,policy.predictions[0],atol=1e-7)
        self.assertFalse(np.array_equal(policy.prompts[2],policy.predictions[1]))
        detections=[event for event in trace if event['phase']=='detection']
        self.assertTrue(detections[1]['after_retreat'])
        self.assertTrue(detections[1]['triggered'])
        self.assertEqual(sum(event['phase']=='retreat_start' for event in trace),1)

    def test_initial_detection_flag_selects_author_replan_without_retreat(self):
        class InitialDetector(OnceDetector):
            def __call__(self,obs):
                self.calls+=1
                return dict(score=torch.tensor(float(self.calls==1)))
        policy=FakePolicy();trace=[]
        result=rollout(FakeSim(),policy,InitialDetector(),dict(seed=1,state={}),
                       dict(settings(),max_steps=12),123,trace)
        self.assertEqual(result['recoveries'],0)
        np.testing.assert_allclose(policy.prompts[1],policy.predictions[0],atol=1e-7)
        self.assertIsNone(policy.prompts[2])
        self.assertFalse(any(event['phase']=='retreat' for event in trace))

    def test_normal_forward_refreshes_cache_after_recovery(self):
        class SequenceDetector(OnceDetector):
            def __call__(self,obs):
                self.calls+=1
                return dict(score=torch.tensor(float(self.calls in (2,5))))
        policy=FakePolicy();trace=[]
        result=rollout(FakeSim(),policy,SequenceDetector(),dict(seed=1,state={}),
                       dict(settings(),max_steps=28),123,trace)
        self.assertEqual(result['recoveries'],2)
        self.assertIsNone(policy.prompts[2])
        np.testing.assert_allclose(policy.prompts[3],policy.predictions[2],atol=1e-7)
        self.assertFalse(np.array_equal(policy.prompts[3],policy.predictions[0]))

    def test_timeout_protocol_is_twenty_ticks_without_relaxing_pose_tolerance(self):
        protocol=json.loads((Path(__file__).resolve().parents[1]/'configs/evaluation_can.json').read_text())
        self.assertEqual(protocol['rollback_max_control_steps'],20)
        self.assertEqual(protocol['rollback_position_tolerance'],.005)
        self.assertEqual(protocol['rollback_rotation_tolerance'],.05)
        self.assertEqual(protocol['recovery_cache_update'],'normal_forward_only')

    def test_chunk_retreat_stops_at_episode_budget_without_replanning(self):
        sim=FakeSim();policy=FakePolicy();trace=[]
        spec=dict(settings(),max_steps=6)
        result=rollout(sim,policy,OnceDetector(),dict(seed=1,state={}),spec,123,trace)
        self.assertEqual(result['physical_steps'],6)
        self.assertEqual(result['termination'],'budget')
        self.assertEqual(result['decisions'],1)
        self.assertEqual(len(policy.prompts),1)
        self.assertEqual(sim.reset_count,1)
        self.assertEqual(sum(event['phase']=='retreat' for event in trace),2)

    def test_unreachable_retreat_fails_instead_of_restoring_simulator(self):
        sim=FakeSim(freeze=True)
        result=rollout(sim,FakePolicy(),OnceDetector(),dict(seed=1,state={}),settings(),123,[])
        self.assertEqual(sim.reset_count,1)
        self.assertEqual(result['termination'],'rollback_failed')
        self.assertEqual(result['physical_steps'],6)

    def test_aggregation_keeps_both_paper_readings_and_refuses_partial_sweeps(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs=[]
            for seed in range(3):
                for final in (False,True):
                    name=f'{seed}_{final}';out=root/name;out.mkdir()
                    success=([.4,.5,.6] if not final else [.3,.4,.2])[seed]
                    (out/'summary.json').write_text(json.dumps(dict(smoke=False,episodes=50,success_rate=success)))
                    (out/'job_record.json').write_text(json.dumps(dict(fingerprint=name)))
                    jobs.append(dict(id=name,output=str(out),fingerprint=name,dataset='paired',mode='baseline',
                        detector_dataset=None,scale=0.,seed=seed,checkpoint=str(out/('final_model.pt' if final else 'step_000500.pt'))))
            plan=dict(jobs=jobs,protocol=dict(rollouts_per_checkpoint=50,training_seeds=[0,1,2]),
                      unavailable_models=[],checkpoint_scope='all')
            row=summarize(plan,root)['rows'][0]
            self.assertAlmostEqual(row['paper_caption_mean_seed_best'],.5)
            self.assertAlmostEqual(row['paper_prose_max_across_seeds'],.6)
            self.assertAlmostEqual(row['mean_final_success'],.3)
            # One seed still has its final model, but the checkpoint sweep is incomplete.
            (root/'2_False/summary.json').unlink()
            row=summarize(plan,root)['rows'][0]
            self.assertTrue(row['complete_three_seeds'])
            self.assertFalse(row['complete_checkpoint_sweep'])
            self.assertIsNone(row['paper_caption_mean_seed_best'])
            self.assertAlmostEqual(row['mean_final_success'],.3)
            # A diagnostic run must never be counted as a formal episode set.
            (root/'2_False/summary.json').write_text(json.dumps(dict(smoke=True,episodes=50,success_rate=1.)))
            report=summarize(plan,root)
            self.assertIn('2_False',report['missing_jobs'])

    def test_parallel_driver_assigns_gpus_and_reuses_completed_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);catalog=root/'catalog.json';catalog.touch()
            checkpoint=root/'checkpoint.pt';checkpoint.touch()
            jobs=[];calls=[];guard=threading.Lock()
            for i in range(4):
                out=root/f'job{i}'
                jobs.append(dict(id=str(i),output=str(out),checkpoint=str(checkpoint),detector=None,
                                 command=['fake','--output',str(out)],fingerprint=str(i)))
            plan=dict(jobs=jobs,unavailable_models=[],catalog=str(catalog),protocol=dict(rollouts_per_checkpoint=50))
            class Process:
                def __init__(self,command,env,**kwargs):
                    self.output=Path(command[command.index('--output')+1]);self.pid=999999
                    with guard:calls.append(env['CUDA_VISIBLE_DEVICES'])
                def wait(self):
                    time.sleep(.02)
                    (self.output/'summary.json').write_text(json.dumps(dict(smoke=False,episodes=50)))
                    return 0
            with patch('evaluation_matrix.subprocess.Popen',Process):
                self.assertTrue(execute(plan,root,['0','1'],0))
                self.assertTrue(execute(plan,root,['0','1'],0))
            self.assertEqual(len(calls),4)
            self.assertEqual(set(calls),{'0','1'})
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'completed')

    def test_matrix_python_path_is_stable_across_shell_and_direct_entrypoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for mode in ('baseline','conditional'):
                path=root/f'policies/can_paired/{mode}/seed_0/final_model.pt'
                path.parent.mkdir(parents=True);path.touch()
            args=SimpleNamespace(datasets=['paired'],seeds=[0],artifacts=root,checkpoints='final',
                                 output=root/'results',catalog=root/'catalog.json')
            with patch('evaluation_matrix.sys.executable','/tmp/venv/../venv/bin/python'):
                first=build(args)
            with patch('evaluation_matrix.sys.executable','/tmp/venv/bin/python'):
                second=build(args)
            self.assertEqual(first,second)
            for mode in ('baseline','conditional'):
                directory=root/f'policies/can_paired/{mode}/seed_0'
                (directory/'best_validation.pt').touch()
                (directory/'checkpoints').mkdir()
                (directory/'checkpoints/step_000500.pt').touch()
            args.checkpoints='all'
            args.detector_artifacts=root/'independent_detectors'
            plan=build(args)
            self.assertEqual(len(plan['jobs']),42)  # Three weights times fourteen configurations.
            self.assertEqual({Path(j['checkpoint']).name for j in plan['jobs']},
                             {'step_000500.pt','best_validation.pt','final_model.pt'})
            self.assertTrue(all(str(args.detector_artifacts) in j['detector']
                                for j in plan['jobs'] if j['detector']))


if __name__=='__main__':unittest.main()

#!/usr/bin/env python3
"""Robosuite 1.4.1 JSON pipe worker; run with the existing simulation Python."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import base64
import json
import sys
import traceback

import h5py
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    args=parser.parse_args()
    channel=sys.stdout
    sys.stdout=sys.stderr  # Third-party startup prints must not corrupt RPC.
    import robosuite
    assert robosuite.__version__=='1.4.1',robosuite.__version__
    with h5py.File(args.metadata,'r') as f:
        metadata=json.loads(f['data'].attrs['env_args'])
    if metadata['env_name']!='PickPlaceCan':
        raise ValueError('Evaluation must use the original PickPlaceCan environment')
    kwargs=dict(metadata['env_kwargs'])
    kwargs.update(has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,
        camera_names=['agentview','robot0_eye_in_hand'],camera_heights=84,camera_widths=84,
        ignore_done=True,reward_shaping=False,
        render_gpu_device_id=int(os.environ.get('EVAL_RENDER_DEVICE','0')))
    env=robosuite.make(metadata['env_name'],**kwargs)

    def observation(raw=None):
        raw=env._get_observations(force_update=True) if raw is None else raw
        result={k:np.asarray(raw[k]).tolist() for k in
            ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
        result['images']={k:base64.b64encode(np.ascontiguousarray(raw[k][::-1]).tobytes()).decode()
                         for k in ('agentview_image','robot0_eye_in_hand_image')}
        pose=np.eye(4)
        from robosuite.utils.transform_utils import quat2mat
        pose[:3,:3]=quat2mat(np.asarray(raw['robot0_eef_quat']))
        pose[:3,3]=raw['robot0_eef_pos']
        result['eef_pose']=pose.tolist()
        result['success']=bool(env._check_success())
        return result

    def state():
        return dict(model=env.sim.model.get_xml(),states=env.sim.get_state().flatten().tolist())

    try:
        for line in sys.stdin:
            try:
                request=json.loads(line)
                op=request['op']
                if op=='close':
                    response={'closed':True}
                elif op=='info':
                    import importlib.metadata
                    ctrl=env.robots[0].controller
                    response=dict(environment=metadata['env_name'],robosuite=robosuite.__version__,
                        controller=metadata['env_kwargs']['controller_configs'],
                        input_min=np.asarray(ctrl.input_min).tolist(),input_max=np.asarray(ctrl.input_max).tolist(),
                        output_min=np.asarray(ctrl.output_min).tolist(),output_max=np.asarray(ctrl.output_max).tolist(),
                        image_shape=[84,84,3],image_convention='robomimic_vertical_flip',
                        egl_device=os.environ.get('MUJOCO_EGL_DEVICE_ID'),python=sys.executable,
                        mujoco=importlib.metadata.version('mujoco'),robomimic=importlib.metadata.version('robomimic'),
                        control_freq=kwargs['control_freq'])
                elif op=='reset':
                    np.random.seed(request['seed'])
                    env.reset()
                    saved=request.get('state')
                    if saved is not None:
                        env.reset_from_xml_string(env.edit_model_xml(saved['model']))
                        env.sim.reset()
                        env.sim.set_state_from_flattened(np.asarray(saved['states']))
                        env.sim.forward()
                    response=dict(observation=observation(),state=state())
                elif op=='step':
                    action=np.asarray(request['action'],dtype=np.float64)
                    if action.shape!=(7,) or not np.isfinite(action).all() or np.abs(action).max()>1+1e-6:
                        raise ValueError('Expected bounded finite 7D environment action')
                    raw,reward,done,info=env.step(action)
                    response=dict(observation=observation(raw),reward=float(reward),done=bool(done))
                else:
                    raise ValueError(f'Unknown operation {op}')
                channel.write(json.dumps(dict(ok=True,result=response),allow_nan=False)+'\n');channel.flush()
                if op=='close':break
            except Exception as exc:
                traceback.print_exc()
                channel.write(json.dumps(dict(ok=False,error=repr(exc)))+'\n');channel.flush()
    finally:
        env.close()


if __name__=='__main__':main()

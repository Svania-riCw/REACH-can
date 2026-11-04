"""Keep the policy and simulator in their already-validated Python environments."""
import base64
import json
import os
from pathlib import Path
import select
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SIM_PYTHON=ROOT.parent/'reach_simulation_env/venv/bin/python'


class Simulation:
    def __init__(self,metadata,log,python=SIM_PYTHON,timeout=180):
        self.timeout=timeout
        self.log=Path(log).open('w')
        # This host exposes one EGL graphics device even though CUDA exposes
        # eight compute GPUs. Robosuite 1.4.1 also reads CUDA_VISIBLE_DEVICES as
        # an EGL index. Isolate its selector from the policy's CUDA assignment.
        sim_cuda=os.environ.get('SIM_CUDA_VISIBLE_DEVICES','0')
        sim_egl=os.environ.get('SIM_EGL_DEVICE_ID','0')
        self.process=subprocess.Popen([str(python),str(ROOT/'scripts/eval_sim_worker.py'),
            '--metadata',str(metadata)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,bufsize=1,env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',JAX_PLATFORMS='cpu',
                CUDA_VISIBLE_DEVICES=sim_cuda,MUJOCO_EGL_DEVICE_ID=sim_egl,EVAL_RENDER_DEVICE=sim_egl))
        try:self.info=self.request('info')
        except BaseException:
            self.close();raise

    def request(self,op,**kwargs):
        self.process.stdin.write(json.dumps(dict(op=op,**kwargs),allow_nan=False)+'\n')
        self.process.stdin.flush()
        readable,_,_=select.select([self.process.stdout],[],[],self.timeout)
        if not readable:raise TimeoutError(f'Simulator timeout for {op}; see simulator log')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError(f'Simulator exited {self.process.poll()}; see simulator log')
        reply=json.loads(line)
        if not reply['ok']:raise RuntimeError(reply['error'])
        result=reply['result']
        if 'observation' in result:
            obs=result['observation']
            for key in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','eef_pose'):
                obs[key]=np.asarray(obs[key],dtype=np.float64)
            for key,value in obs.pop('images').items():
                obs[key]=np.frombuffer(base64.b64decode(value),dtype=np.uint8).reshape(84,84,3).copy()
        return result

    def reset(self,seed,state=None):return self.request('reset',seed=seed,state=state)
    def step(self,action):return self.request('step',action=np.asarray(action).tolist())

    def close(self):
        if getattr(self,'process',None) is not None and self.process.poll() is None:
            try:
                self.request('close')
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if getattr(self,'log',None):self.log.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

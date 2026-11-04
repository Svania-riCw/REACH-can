"""Integration check against real image data, controller, and EGL rendering."""
from pathlib import Path
import sys
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h5py
import numpy as np
from eval_sim_bridge import Simulation
from eval_recovery import pose_error,retreat_command


root=Path(__file__).resolve().parents[1]
out=root/'outputs/evaluation/environment_validation'
out.mkdir(parents=True,exist_ok=True)
with h5py.File(root/'data/paper/can_paired_image.hdf5','r') as f:
    demo=f['data/demo_0']
    state=dict(model=demo.attrs['model_file'],states=demo['states'][0].tolist())
    expected={k:demo['obs'][k][0] for k in ('agentview_image','robot0_eye_in_hand_image')}
    expected_pos=demo['obs/robot0_eef_pos'][0]
report={}
with Simulation(root/'data/robomimic/can/paired/low_dim_v141.hdf5',out/'simulator.log') as sim:
    first=sim.reset(100000,state)['observation']
    report['dataset_state_position_error']=float(np.linalg.norm(first['robot0_eef_pos']-expected_pos))
    report['dataset_image_mae']={k:float(np.abs(first[k].astype(float)-expected[k]).mean()) for k in expected}
    assert report['dataset_state_position_error']<1e-6,report
    assert max(report['dataset_image_mae'].values())<.5,report
    second=sim.reset(100000,state)['observation']
    assert all(np.array_equal(first[k],second[k]) for k in expected)
    before=second['eef_pose'];obs=second
    obs=sim.step(np.array([.3,0,0,0,0,0,0]))['observation']
    report['forward_pose_error']=pose_error(obs['eef_pose'],before)
    for count in range(10):
        obs=sim.step(retreat_command(obs['eef_pose'],before,sim.info,0))['observation']
        distance,angle=pose_error(obs['eef_pose'],before)
        if distance<.005 and angle<.05:break
    report.update(retreat_position_error=distance,retreat_rotation_error=angle,retreat_steps=count+1,
                  environment=sim.info)
    assert distance<.005 and angle<.05,report
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

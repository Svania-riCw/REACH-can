#!/usr/bin/env python3
"""Check both Python environments, metadata, initial states and ready artifacts."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from checkpoint_utils import PROJECT_ROOT
import numpy as np
import torch
from eval_sim_bridge import Simulation
from eval_catalog import metadata_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=PROJECT_ROOT/'outputs/evaluation/can_initial_states_50.json')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/evaluation/environment_report.json')
    parser.add_argument('--artifacts',type=Path,default=PROJECT_ROOT/'outputs/pi05_full_sft_physical7_v4')
    parser.add_argument('--detector-artifacts',type=Path,default=PROJECT_ROOT/'outputs/paper_can')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    catalog=json.loads(args.catalog.read_text())
    assert len(catalog['episodes'])>=50
    assert len({e['seed'] for e in catalog['episodes']})==len(catalog['episodes'])
    assert metadata_digest(catalog['metadata'])==catalog['metadata_sha256']
    with Simulation(catalog['metadata'],args.output.with_suffix('.simulator.log')) as sim:
        first=sim.reset(catalog['episodes'][0]['seed'],catalog['episodes'][0]['state'])['observation']
        second=sim.reset(catalog['episodes'][0]['seed'],catalog['episodes'][0]['state'])['observation']
        for key in ('agentview_image','robot0_eye_in_hand_image'):
            assert first[key].shape==(84,84,3) and first[key].dtype==np.uint8
            assert np.array_equal(first[key],second[key])
        info=sim.info
    artifacts={}
    for mode in ('baseline','conditional'):
        p=args.artifacts/f'policies/can_paired/{mode}/seed_0/final_model.pt'
        artifacts[f'policy_{mode}']=dict(path=str(p),ready=p.is_file())
    for kind in ('ph','mh','paired'):
        p=args.detector_artifacts/f'detectors/can_{kind}/seed_0/detector.pt'
        artifacts[f'detector_{kind}']=dict(path=str(p),ready=p.is_file())
    report=dict(policy_python=sys.executable,torch=torch.__version__,numpy=np.__version__,
        imageio_ffmpeg=importlib.metadata.version('imageio-ffmpeg'),cuda_available=torch.cuda.is_available(),
        simulator=info,catalog=str(args.catalog),common_initial_states=len(catalog['episodes']),
        deterministic_reset_and_camera_layout=True,artifacts=artifacts,
        all_seed0_artifacts_ready=all(a['ready'] for a in artifacts.values()),
        environment_ready=True,exact_paper_reproduction=False)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()

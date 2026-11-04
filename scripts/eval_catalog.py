#!/usr/bin/env python3
"""Persist randomized initial states shared across every evaluation setting."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
from eval_sim_bridge import Simulation


def metadata_digest(path):
    with h5py.File(path,'r') as f:metadata=json.loads(f['data'].attrs['env_args'])
    return hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--seed-start',type=int,default=100000)
    args=parser.parse_args()
    if args.count<1:parser.error('count must be positive')
    output=Path(args.output)
    if output.exists():raise FileExistsError(f'Refusing to replace a shared evaluation catalog: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    catalog=dict(metadata=str(Path(args.metadata).resolve()),metadata_sha256=metadata_digest(args.metadata),
                 initial_state_source='fresh benchmark environment resets; not demonstration states',episodes=[])
    with Simulation(args.metadata,output.with_suffix('.simulator.log')) as sim:
        catalog['environment']=sim.info
        for i in range(args.count):
            seed=args.seed_start+i
            result=sim.reset(seed)
            catalog['episodes'].append(dict(id=i,seed=seed,state=result['state']))
            print(f'Initial states {i+1}/{args.count}',flush=True)
    temporary=output.with_suffix('.tmp')
    temporary.write_text(json.dumps(catalog,sort_keys=True))
    temporary.replace(output)
    print(f'Ready: {output}',flush=True)


if __name__=='__main__':main()

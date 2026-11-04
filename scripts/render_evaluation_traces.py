#!/usr/bin/env python3
"""Replay saved Can action traces and render two-camera MP4s."""
import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from eval_sim_bridge import Simulation


def action_events(trace, expected_steps):
    events=[event for event in trace if event['phase'] in ('forward','retreat')]
    if len(events)!=expected_steps or [event['step'] for event in events]!=list(range(1,expected_steps+1)):
        raise ValueError('Trace actions do not match the recorded physical step count')
    return events


def video_filename(episode_id,success):
    outcome='success' if success else 'failed'
    return f'episode_{episode_id:03d}_{outcome}.mp4'


def normalize_video_name(output_dir,episode_id,success):
    output_dir=Path(output_dir)
    target=output_dir/video_filename(episode_id,success)
    legacy=output_dir/f'episode_{episode_id:03d}.mp4'
    if legacy.exists() and not target.exists():
        legacy.replace(target)
    return target


def render_shard(shard,catalog_path,output_dir,overwrite=False):
    shard=Path(shard);catalog=json.loads(Path(catalog_path).read_text())
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (shard/'episodes.jsonl').read_text().splitlines()]
    rows.sort(key=lambda row:row['episode_id'])
    simulator_log=output_dir/f'simulator_{shard.parent.name}_{shard.name}.log'
    with Simulation(catalog['metadata'],simulator_log) as sim:
        for row in rows:
            episode_id=row['episode_id']
            video=normalize_video_name(output_dir,episode_id,row['success'])
            if video.exists() and not overwrite:
                print(f'Skip existing video for episode {episode_id}',flush=True);continue
            trace_path=shard/f'episode_{episode_id:03d}_trace.json'
            trace=json.loads(trace_path.read_text())
            events=action_events(trace,row['physical_steps'])
            initial=catalog['episodes'][episode_id]
            reset=sim.reset(initial['seed'],initial['state'])
            temporary=video.with_name(f'.{video.name}.{os.getpid()}.tmp.mp4')
            try:
                with imageio.get_writer(temporary,fps=20,macro_block_size=1) as writer:
                    observation=reset['observation']
                    writer.append_data(np.concatenate((observation['agentview_image'],
                                                       observation['robot0_eye_in_hand_image']),axis=1))
                    for event in events:
                        observation=sim.step(event['action'])['observation']
                        writer.append_data(np.concatenate((observation['agentview_image'],
                                                           observation['robot0_eye_in_hand_image']),axis=1))
                temporary.replace(video)
            finally:
                temporary.unlink(missing_ok=True)
            print(f'Rendered episode {episode_id}: {video}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard-dir',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    render_shard(args.shard_dir,args.catalog,args.output_dir,args.overwrite)


if __name__=='__main__':main()

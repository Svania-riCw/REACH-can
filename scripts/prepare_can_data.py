#!/usr/bin/env python3
"""Download official v1.4.1 Can data and render stored states without replay drift.

Run with the simulation venv, not the policy venv. Existing original extraction
jobs and files are left alone. New renders are published atomically only when
all episodes, cameras and masks are complete.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import fcntl
import json
from pathlib import Path
import urllib.request

import h5py
import numpy as np
from data_splits import demo_names, scene_groups
from dataset_paths import PROJECT_ROOT, dataset_paths


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with h5py.File(path, 'r') as file:
            print(f'Using existing {path}: {len(file["data"])} episodes', flush=True)
        return
    partial = path.with_suffix('.download')
    print(f'Downloading {url}', flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as file:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            file.write(block)
    with h5py.File(partial, 'r') as file:
        if not demo_names(file):
            raise ValueError('Empty downloaded dataset')
    partial.replace(path)


def render(source, destination, kind, limit_episodes=None, limit_frames=None):
    import robosuite as suite
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    partial = destination.with_suffix('.partial.hdf5')
    with h5py.File(source, 'r') as file:
        metadata = json.loads(file['data'].attrs['env_args'])
        names = demo_names(file)
        if limit_episodes:
            names = names[:limit_episodes]
        groups = scene_groups(file, names, kind)
        kwargs = dict(metadata['env_kwargs'])
        kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                      camera_names=['agentview', 'robot0_eye_in_hand'], camera_heights=84, camera_widths=84)
        env = suite.make(metadata['env_name'], **kwargs)
        try:
            with h5py.File(partial, 'w') as out:
                out.attrs.update(complete=False, dataset_type=kind, source_path=str(source.resolve()),
                                 robosuite_version=suite.__version__, image_convention='robomimic_vertical_flip',
                                 smoke_subset=bool(limit_episodes or limit_frames))
                data = out.create_group('data')
                for key, value in file['data'].attrs.items():
                    data.attrs[key] = value
                total = 0
                for i, name in enumerate(names):
                    src = file[f'data/{name}']
                    dst = data.create_group(name)
                    for key, value in src.attrs.items():
                        dst.attrs[key] = value
                    if kind == 'paired':
                        dst.attrs['pair_id'] = groups[name]
                    length = min(len(src['actions']), limit_frames) if limit_frames else len(src['actions'])
                    if len(src['states']) < length:
                        raise ValueError(f'Missing stored simulator states in {name}')
                    dst.attrs['num_samples'] = length
                    for key in ('actions', 'states', 'rewards', 'dones'):
                        if key in src:
                            dst.create_dataset(key, data=src[key][:length])
                    obs = dst.create_group('obs')
                    for key in src['obs']:
                        if 'image' not in key:
                            obs.create_dataset(key, data=src['obs'][key][:length])
                    cameras = {key: obs.create_dataset(key, shape=(length, 84, 84, 3), dtype=np.uint8,
                                                      chunks=(1, 84, 84, 3), compression='lzf')
                               for key in ('agentview_image', 'robot0_eye_in_hand_image')}
                    env.reset()
                    xml = src.attrs.get('model_file')
                    if xml is None:
                        raise ValueError(f'Missing original scene XML for {name}')
                    if isinstance(xml, bytes):
                        xml = xml.decode()
                    env.reset_from_xml_string(env.edit_model_xml(xml))
                    env.sim.reset()
                    for t in range(length):
                        env.sim.set_state_from_flattened(src['states'][t])
                        env.sim.forward()
                        current = env._get_observations(force_update=True)
                        for key, camera in cameras.items():
                            # Match robomimic EnvRobosuite.get_observation.
                            camera[t] = current[key][::-1]
                    total += length
                    out.flush()
                    print(f'{kind}: {i+1}/{len(names)} episodes, {total} frames', flush=True)
                if 'mask' in file:
                    included = set(names)
                    filtered = {key: [x for x in file[f'mask/{key}'][:] if x.decode() in included]
                                for key in file['mask']}
                    # Tiny smoke subsets may contain no official validation pair.
                    if not (limit_episodes and (not filtered.get('train') or not filtered.get('valid'))):
                        masks = out.create_group('mask')
                        for key, values in filtered.items():
                            masks.create_dataset(key, data=np.asarray(values, dtype='S'))
                data.attrs['total'] = total
                out.attrs['complete'] = True
            partial.replace(destination)
            print(f'Ready: {destination}', flush=True)
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-types', nargs='+', choices=['ph', 'mh', 'paired'], default=['paired'])
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--output', type=Path, help='Only valid with one dataset type')
    parser.add_argument('--limit-episodes', type=int, help='Smoke testing only; requires --output')
    parser.add_argument('--limit-frames', type=int, help='Smoke testing only; requires --output')
    args = parser.parse_args()
    if any(value is not None and value < 1 for value in (args.limit_episodes, args.limit_frames)):
        parser.error('Smoke limits must be positive')
    if args.output and len(args.dataset_types) != 1:
        parser.error('--output requires exactly one dataset type')
    if (args.limit_episodes or args.limit_frames) and args.output is None:
        parser.error('Smoke limits require an explicit --output; never publish a subset as the full dataset')
    for kind in args.dataset_types:
        paths = dataset_paths(kind)
        paths['metadata'].parent.mkdir(parents=True, exist_ok=True)
        if args.download_only and paths['metadata'].exists():
            download(paths['url'], paths['metadata'])
            continue
        with paths['metadata'].with_suffix('.prepare.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            download(paths['url'], paths['metadata'])
            if args.download_only:
                continue
            destination = args.output or paths['images']
            if destination.exists():
                with h5py.File(destination, 'r') as file:
                    if not file.attrs.get('complete', False) or file.attrs.get('smoke_subset', False) != bool(args.limit_episodes or args.limit_frames):
                        raise ValueError(f'Existing render is incomplete or incompatible: {destination}')
                print(f'Using completed render: {destination}', flush=True)
                continue
            render(paths['metadata'], destination, kind, args.limit_episodes, args.limit_frames)


if __name__ == '__main__':
    main()

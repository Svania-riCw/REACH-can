#!/usr/bin/env python3
"""Reuse completed legacy images: flip vertically and restore official metadata.

This does not re-render legacy images from the original per-episode XML. The
conversion records that provenance instead of claiming a fresh render.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from data_splits import demo_names, make_episode_split
from dataset_paths import dataset_paths

CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')
STATE_KEYS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')


def convert(legacy_path, metadata_path, output_path, dataset_type='ph', block_size=32):
    legacy_path, metadata_path, output_path = map(lambda path: Path(path).resolve(),
                                                 (legacy_path, metadata_path, output_path))
    if output_path in (legacy_path, metadata_path):
        raise ValueError('Conversion must use a different output file')
    if block_size < 1:
        raise ValueError('block_size must be positive')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix('.conversion.json')
    with output_path.with_suffix('.conversion.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output_path.exists():
            raise FileExistsError(f'Refusing to overwrite {output_path}')
        partial = output_path.with_suffix('.conversion.partial.hdf5')
        expected_hash = hashlib.sha256()
        total_frames = 0
        with h5py.File(legacy_path, 'r') as legacy, h5py.File(metadata_path, 'r') as metadata:
            names = demo_names(metadata)
            if set(names) != set(demo_names(legacy)):
                raise ValueError('Episode names differ between legacy images and original metadata')
            if 'mask/train' not in metadata or 'mask/valid' not in metadata:
                raise ValueError('Original metadata has no official train/valid masks')
            with h5py.File(partial, 'w') as out:
                env_args = json.loads(metadata['data'].attrs['env_args'])
                out.attrs.update(complete=False, smoke_subset=False, dataset_type=dataset_type,
                                 source_path=str(metadata_path), source_image_path=str(legacy_path),
                                 robosuite_version=env_args.get('env_version', 'unknown'),
                                 image_convention='robomimic_vertical_flip',
                                 preparation_method='legacy_vertical_flip_and_metadata_restore',
                                 original_scene_xml_rerendered=False)
                data = out.create_group('data')
                for key, value in metadata['data'].attrs.items():
                    data.attrs[key] = value
                for i, name in enumerate(names):
                    source, previous = metadata[f'data/{name}'], legacy[f'data/{name}']
                    length = len(source['actions'])
                    if not length or not np.array_equal(source['actions'][:], previous['actions'][:]):
                        raise ValueError(f'{name}: original and legacy actions differ')
                    for key in STATE_KEYS:
                        if not np.array_equal(source[f'obs/{key}'][:], previous[f'obs/{key}'][:]):
                            raise ValueError(f'{name}: original and legacy state differs: {key}')
                    # Copy from the original, preserving states, XML, rewards,
                    # dones, next_obs and all episode attributes exactly.
                    metadata.copy(source, data, name=name)
                    dst = data[name]
                    for camera in CAMERAS:
                        images = previous[f'obs/{camera}']
                        if images.shape != (length, 84, 84, 3) or images.dtype != np.uint8:
                            raise ValueError(f'{name}/{camera}: wrong image shape or dtype')
                        target = dst['obs'].create_dataset(camera, shape=images.shape, dtype=np.uint8,
                                  chunks=(1,84,84,3), compression='lzf')
                        for start in range(0, length, block_size):
                            block = np.ascontiguousarray(images[start:start+block_size][:, ::-1, :, :])
                            target[start:start+len(block)] = block
                            expected_hash.update(block.tobytes())
                    total_frames += length
                    if (i+1) % 20 == 0 or i+1 == len(names):
                        out.flush()
                        print(f'Converted {i+1}/{len(names)} episodes; {total_frames} frames', flush=True)
                metadata.copy('mask', out)
                data.attrs['total'] = total_frames
                out.attrs['converted_image_sha256'] = expected_hash.hexdigest()
                out.attrs['complete'] = True
        # Verify the full written image payload, not just sampled frames.
        actual_hash = hashlib.sha256()
        with h5py.File(partial, 'r') as out, h5py.File(metadata_path, 'r') as metadata:
            for name in names:
                source, target = metadata[f'data/{name}'], out[f'data/{name}']
                for key in ('actions', 'states', 'rewards', 'dones'):
                    if key in source and not np.array_equal(source[key][:], target[key][:]):
                        raise ValueError(f'Written {name}/{key} differs from original')
                for camera in CAMERAS:
                    images = target[f'obs/{camera}']
                    for start in range(0, len(images), block_size):
                        actual_hash.update(images[start:start+block_size].tobytes())
            for key in metadata['mask']:
                if not np.array_equal(metadata[f'mask/{key}'][:], out[f'mask/{key}'][:]):
                    raise ValueError(f'Written official mask differs: {key}')
        if expected_hash.hexdigest() != actual_hash.hexdigest():
            raise ValueError('Converted image payload failed readback verification')
        split = make_episode_split(partial, dataset_type, metadata_path=metadata_path)
        report = dict(timestamp=datetime.now(timezone.utc).isoformat(), dataset_type=dataset_type,
                      source_images=str(legacy_path), source_metadata=str(metadata_path), output=str(output_path),
                      episodes=len(names), frames=total_frames, train_episodes=len(split['train']),
                      valid_episodes=len(split['val']), split_protocol=split['protocol'],
                      image_sha256=actual_hash.hexdigest(), full_image_readback_verified=True,
                      original_actions_states_rewards_dones_verified=True, all_official_masks_verified=True,
                      image_convention='robomimic_vertical_flip', original_scene_xml_rerendered=False,
                      provenance_note='Reused legacy renderer pixels; restored original metadata and flipped images. '
                                      'Sample re-render comparisons were close, but this is not an exact per-episode XML re-render.')
        report_partial = report_path.with_suffix('.json.tmp')
        report_partial.write_text(json.dumps(report, indent=2))
        partial.replace(output_path)
        report_partial.replace(report_path)
        print(json.dumps(report, indent=2), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', required=True, type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dataset-type', choices=['ph', 'mh'], default='ph')
    args = parser.parse_args()
    paths = dataset_paths(args.dataset_type)
    convert(args.legacy, args.metadata or paths['metadata'], args.output or paths['images'], args.dataset_type)


if __name__ == '__main__':
    main()

"""Robomimic splits shared by policies and detectors, with Paired scene isolation."""
import hashlib
from pathlib import Path
import re

import h5py
import numpy as np


def demo_names(file):
    return sorted(file['data'], key=lambda name: int(re.search(r'\d+$', name).group()))


def scene_groups(file, names, dataset_type):
    if dataset_type != 'paired':
        return {name: name for name in names}
    groups = {}
    for name in names:
        demo = file[f'data/{name}']
        if 'pair_id' in demo.attrs:
            groups[name] = str(demo.attrs['pair_id'])
        elif 'states' in demo:
            # First element is simulator time, not part of the initial scene.
            state = np.round(np.asarray(demo['states'][0, 1:], dtype='<f8'), 6)
            groups[name] = hashlib.sha256(state.tobytes()).hexdigest()
        else:
            raise ValueError('Paired split requires pair_id or original states; provide --metadata')
    counts = {}
    for group in groups.values():
        counts[group] = counts.get(group, 0) + 1
    if len(counts) < 2 or any(count != 2 for count in counts.values()):
        raise ValueError('Paired data must contain at least two initial scenes, each with exactly two demonstrations')
    return groups


def make_episode_split(path, dataset_type='ph', val_ratio=.1, seed=0, split=None, metadata_path=None):
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    with h5py.File(path, 'r') as file:
        names = demo_names(file)
    if len(names) < 2:
        raise ValueError('At least two episodes are required')
    with h5py.File(metadata_path or path, 'r') as file:
        if set(demo_names(file)) != set(names):
            raise ValueError('Metadata and image dataset episode names differ')
        groups = scene_groups(file, names, dataset_type)
        if split is None and 'mask/train' in file and 'mask/valid' in file:
            decode = lambda values: [value.decode() if isinstance(value, bytes) else str(value) for value in values]
            split = {'train': decode(file['mask/train'][:]), 'val': decode(file['mask/valid'][:]), 'protocol': 'official_masks'}
    if split is None:
        unique = sorted(set(groups.values()))
        if len(unique) < 2:
            raise ValueError('At least two independent scenes are needed')
        shuffled = np.random.default_rng(seed).permutation(unique).tolist()
        n_val = min(len(unique) - 1, max(1, int(len(unique) * val_ratio)))
        val_groups = set(shuffled[:n_val])
        split = {'train': [name for name in names if groups[name] not in val_groups],
                 'val': [name for name in names if groups[name] in val_groups],
                 'protocol': 'scene_groups' if dataset_type == 'paired' else 'episode_random'}
    train, val = split['train'], split['val']
    if (not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val)
            or set(train) & set(val) or set(train) | set(val) != set(names)):
        raise ValueError('Episode split does not match this dataset')
    if {groups[name] for name in train} & {groups[name] for name in val}:
        raise ValueError('Initial scene leaks between training and validation')
    return split

"""Inspect the three paper datasets; render them with prepare_can_data.sh first."""
import argparse
from pathlib import Path
import h5py
from openpi_robomimic_adapter import validate_data
from data_splits import make_episode_split
from dataset_paths import PROJECT_ROOT, dataset_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data/paper')
    args = parser.parse_args()
    ready = []
    for kind in ('ph', 'mh', 'paired'):
        paths = dataset_paths(kind)
        image_path = args.data_dir / f'can_{kind}_image.hdf5'
        try:
            with h5py.File(paths['metadata'], 'r') as file:
                expected = len(file['data'])
            names = validate_data(image_path, expected)
            with h5py.File(image_path, 'r') as file:
                if not file.attrs.get('complete', False) or file.attrs.get('smoke_subset', False):
                    raise ValueError('Not a completed full paper render')
            split = make_episode_split(image_path, kind, metadata_path=paths['metadata'])
        except (OSError, KeyError, ValueError) as error:
            print(f'Can({kind}): not ready: {error}')
            continue
        print(f'Can({kind}): ready, {len(names)} episodes, split={split["protocol"]}')
        ready.append(kind)
    print(f'Ready paper datasets: {ready}')


if __name__ == '__main__':
    main()

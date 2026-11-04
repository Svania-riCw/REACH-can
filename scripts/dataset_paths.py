"""Keep paper datasets separate from earlier, potentially incompatible renders."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dataset_paths(kind, root=PROJECT_ROOT):
    root = Path(root)
    if kind not in {'ph', 'mh', 'paired'}:
        raise ValueError(kind)
    legacy = root.parent / f'reach_simulation_env/data/robomimic/can/{kind}/low_dim_v141.hdf5'
    local = root / f'data/robomimic/can/{kind}/low_dim_v141.hdf5'
    return {'images': root / f'data/paper/can_{kind}_image.hdf5',
            'metadata': legacy if kind != 'paired' and legacy.exists() else local,
            'url': f'https://downloads.cs.stanford.edu/downloads/rt_benchmark/can/{kind}/low_dim_v141.hdf5'}

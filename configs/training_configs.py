"""Optional single-job presets; paper_can.json is the canonical experiment matrix."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get_config(name):
    choices = {f'can_{kind}_{mode}': (kind, mode)
               for kind in ('paired', 'mh', 'ph') for mode in ('baseline', 'conditional')}
    if name not in choices:
        raise ValueError(f'Unknown preset: {name}; choose from {list(choices)}')
    kind, mode = choices[name]
    spec = json.loads((ROOT / 'configs/paper_can.json').read_text())
    return dict(dataset_type=kind, policy_mode=mode, data=str(ROOT / f'data/paper/can_{kind}_image.hdf5'),
                epochs=spec['training']['epochs'], batch_size=spec['training']['global_batch_size'],
                lr=spec['training']['learning_rate'], warmup_steps=spec['training']['warmup_steps'],
                min_lr=spec['training']['min_lr'], weight_decay=spec['training']['weight_decay'],
                beta1=spec['training']['betas'][0], beta2=spec['training']['betas'][1],
                precision=spec['training']['precision'],
                gradient_accumulation_steps=spec['training']['gradient_accumulation_steps'])

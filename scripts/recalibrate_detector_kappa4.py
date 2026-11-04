"""Derive a detector threshold from saved held-out Gaussian statistics."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from evaluate_can import digest
import torch


def recalibrate(source, output, kappa=4.):
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError('kappa must be finite and nonnegative')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(source, map_location='cpu', weights_only=False)
    original = copy.deepcopy(payload['calibration'])
    if original['method'] != 'gaussian' or original['calibration_source'] != 'held_out_id_validation':
        raise ValueError('Expected saved held-out Gaussian calibration')
    calibration = dict(original, kappa=kappa, threshold=original['mean'] + kappa * original['std'])
    provenance = dict(source=str(source), source_sha256=digest(source),
                      original_calibration=original, method='reuse_saved_held_out_statistics',
                      ae_weights_changed=False)
    payload['calibration'] = calibration
    payload['args'] = dict(payload['args'], kappa=kappa, output=str(output.parent))
    payload['recalibration'] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(output)
    for name, value in (('calibration.json', calibration), ('config.json', payload['args']),
                        ('recalibration.json', provenance)):
        (output.parent / name).write_text(json.dumps(value, indent=2) + '\n')
    split = source.parent / 'episode_split.json'
    if split.exists():
        shutil.copy2(split, output.parent / split.name)
    return calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kappa', type=float, default=4.)
    args = parser.parse_args()
    print(json.dumps(recalibrate(args.source, args.output, args.kappa)), flush=True)


if __name__ == '__main__':
    main()

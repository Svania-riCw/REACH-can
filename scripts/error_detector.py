"""Feedforward image AE and calibrated reconstruction-error detector (REACH §3.2).

Layer sizes and multi-view aggregation are explicit reproduction choices: the
paper does not publish these architecture details. Inputs are RGB in [0,1].
"""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')


class ImageAutoencoder(nn.Module):
    def __init__(self, image_size=84, hidden_dim=512, latent_dim=128, num_cameras=2):
        super().__init__()
        self.config = dict(image_size=image_size, hidden_dim=hidden_dim,
                           latent_dim=latent_dim, num_cameras=num_cameras)
        width = num_cameras * 3 * image_size ** 2
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(width, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, width), nn.Sigmoid())

    def forward(self, images):
        return self.decoder(self.encoder(images)).reshape_as(images)

    def score(self, images):
        return (self(images) - images).square().flatten(1).mean(1)


def calibrate(scores, method='gaussian', kappa=4., alpha=.05):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError('Calibration needs a nonempty finite vector of ID scores')
    result = dict(method=method, n_samples=len(scores), mean=float(scores.mean()),
                  std=float(scores.std()), calibration_source='held_out_id_validation')
    if method == 'gaussian':
        if kappa < 0 or not math.isfinite(kappa):
            raise ValueError('kappa must be finite and nonnegative')
        result.update(kappa=kappa, threshold=result['mean'] + kappa * result['std'])
    elif method == 'conformal':
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        order = math.ceil((len(scores) + 1) * (1 - alpha))
        if order > len(scores):
            raise ValueError('Too few calibration samples for the requested conformal alpha')
        result.update(alpha=alpha, threshold=float(np.sort(scores)[order - 1]))
    else:
        raise ValueError(f'Unknown threshold method: {method}')
    return result


def pack_images(images, image_size=84, device='cpu', camera_keys=CAMERAS):
    packed = []
    for key in camera_keys:
        image = torch.as_tensor(images[key], device=device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f'{key}: expected RGB [B,C,H,W] or [B,H,W,C]')
        image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Detector images must be uint8 or floats in [0,1], not Pi0.5 [-1,1] tensors')
        packed.append(F.interpolate(image, (image_size, image_size), mode='bilinear', align_corners=False))
    return torch.cat(packed, dim=1)


class ErrorDetector:
    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = ImageAutoencoder(**payload['model_config']).to(self.device)
        self.model.load_state_dict(payload['model_state_dict'], strict=True)
        self.model.eval()
        self.calibration = payload['calibration']
        self.camera_keys = tuple(payload['camera_keys'])
        self.threshold = float(self.calibration['threshold'])

    @torch.no_grad()
    def __call__(self, images):
        inputs = pack_images(images, self.model.config['image_size'], self.device, self.camera_keys)
        score = self.model.score(inputs)
        return {'score': score, 'is_ood': score > self.threshold, 'threshold': self.threshold}

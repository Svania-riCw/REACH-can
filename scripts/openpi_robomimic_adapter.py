"""Episode-level splits, training-only normalization and Robomimic action chunks."""
from pathlib import Path
import json

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from data_splits import demo_names, make_episode_split

STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
CAMERAS = {"base_0_rgb": "agentview_image", "left_wrist_0_rgb": "robot0_eye_in_hand_image"}


def safe_h5py_open(path, mode="r", **kwargs):
    # Never bypass a live writer's lock or assume it enabled SWMR.
    return h5py.File(path, mode, **kwargs)


def validate_data(path, expected_demos=None):
    """Validate every episode, including the last one a writer may be creating."""
    with safe_h5py_open(path) as file:
        if "complete" in file.attrs and not file.attrs["complete"]:
            raise ValueError(f"Render is not complete: {path}")
        names = demo_names(file)
        if len(names) < (expected_demos or 2):
            raise ValueError(f"Only {len(names)} episodes in {path}")
        for name in names:
            demo = file[f"data/{name}"]
            if "actions" not in demo or "obs" not in demo:
                raise ValueError(f"Incomplete episode: {name}")
            length = len(demo["actions"])
            if length == 0 or demo["actions"].shape != (length, 7):
                raise ValueError(f"Expected nonempty [T,7] actions in {name}")
            for key in (*STATE_KEYS, *CAMERAS.values()):
                if key not in demo["obs"] or len(demo["obs"][key]) != length:
                    raise ValueError(f"Missing/misaligned {name}/obs/{key}")
            for key, width in zip(STATE_KEYS, (3, 4, 2)):
                state = demo[f"obs/{key}"]
                if state.shape != (length, width) or not np.isfinite(state[:]).all():
                    raise ValueError(f"Invalid state values/shape in {name}/{key}")
            if not np.isfinite(demo["actions"][:]).all():
                raise ValueError(f"Non-finite actions in {name}")
            for key in CAMERAS.values():
                images = demo["obs"][key]
                if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
                    raise ValueError(f"Expected uint8 [T,H,W,3] images in {name}/{key}")
        return names


def compute_action_stats(hdf5_path, episodes=None, state_keys=STATE_KEYS):
    """Fit action and proprioception scales on training episodes only."""
    values = {"action": [], "state": []}
    with safe_h5py_open(hdf5_path) as file:
        for name in episodes if episodes is not None else demo_names(file):
            demo = file[f"data/{name}"]
            values["action"].append(demo["actions"][:].astype(np.float32))
            values["state"].append(np.concatenate([demo[f"obs/{key}"][:].reshape(len(demo["actions"]), -1)
                                                    for key in state_keys], axis=-1))
    result = {"schema_version": 2, "normalization": "quantile", "state_tokenization": "real_dimensions",
              "state_keys": list(state_keys), "fit_source": "training_episodes_only"}
    for key, arrays in values.items():
        array = np.concatenate(arrays)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite {key} values")
        result[key] = {"min": array.min(0).tolist(), "max": array.max(0).tolist(),
                       "mean": array.mean(0).tolist(), "std": array.std(0).tolist(),
                       "q01": np.quantile(array, .01, axis=0).tolist(),
                       "q99": np.quantile(array, .99, axis=0).tolist()}
    return result


def normalize(value, stats):
    if 'q01' in stats:
        low, high = np.asarray(stats['q01']), np.asarray(stats['q99'])
        return ((np.asarray(value) - low) / (high - low + 1e-6) * 2 - 1).astype(np.float32)
    # Old checkpoints retain their original min/max interpretation.
    low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
    span = high - low
    return np.where(span > 1e-6, 2 * (value - low) / np.maximum(span, 1e-6) - 1, 0).astype(np.float32)


def denormalize(value, stats):
    if 'q01' in stats:
        low, high = np.asarray(stats['q01']), np.asarray(stats['q99'])
        return (np.asarray(value) + 1) / 2 * (high - low + 1e-6) + low
    low, high = np.asarray(stats['min']), np.asarray(stats['max'])
    return (np.asarray(value) + 1) / 2 * (high - low) + low


def encode_state(state, stats, action_dim, tokenizer=None, prompt=''):
    """OpenPI tokenizes actual robot DOFs before tensor padding; preserve old checkpoints."""
    state = normalize(state, stats['state']) if stats else np.asarray(state, dtype=np.float32)
    if len(state) > action_dim:
        raise ValueError(f'State dimension {len(state)} exceeds model width {action_dim}')
    padded = np.pad(state, (0, action_dim - len(state))).astype(np.float32)
    real_dimensions = stats is None or stats.get('state_tokenization') == 'real_dimensions'
    token_state = state if real_dimensions else np.clip(padded, -1, 1)
    tokens, mask = tokenizer.tokenize(prompt, token_state) if tokenizer is not None else (None, None)
    return padded, tokens, mask


class RobomimicHDF5Dataset(Dataset):
    def __init__(self, hdf5_path, episodes=None, action_stats=None, action_horizon=50,
                 action_dim=32, tokenizer=None, prompt="Pick up the can and place it in the bin."):
        self.path = str(hdf5_path)
        self.stats = action_stats
        self.action_horizon, self.action_dim = action_horizon, action_dim
        self.tokenizer, self.prompt = tokenizer, prompt
        self.index_map = []
        with safe_h5py_open(self.path) as file:
            self.demo_names = episodes if episodes is not None else demo_names(file)
            for name in self.demo_names:
                self.index_map.extend((name, t) for t in range(len(file[f"data/{name}/actions"])))
        if action_dim < 7 or action_horizon < 1:
            raise ValueError("Model action_dim must be >=7 and action_horizon >=1")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        name, timestep = self.index_map[index]
        # One reader per sample is fork-safe, including on shared filesystems.
        with safe_h5py_open(self.path) as file:
            demo = file[f"data/{name}"]
            images = {target: torch.from_numpy(demo[f"obs/{source}"][timestep].copy()).permute(2, 0, 1).float() / 127.5 - 1
                      for target, source in CAMERAS.items()}
            images["right_wrist_0_rgb"] = torch.zeros_like(images["left_wrist_0_rgb"])
            state_keys = self.stats.get("state_keys", STATE_KEYS) if self.stats else STATE_KEYS
            state = np.concatenate([demo[f"obs/{key}"][timestep].reshape(-1) for key in state_keys])
            actions = demo["actions"][timestep:min(timestep + self.action_horizon, len(demo["actions"]))]
        # Repeat the final action at episode boundaries; never cross episodes.
        actions = np.concatenate([actions, np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)])
        if self.stats:
            actions = normalize(actions, self.stats["action"])
        state, tokens, mask = encode_state(state, self.stats, self.action_dim, self.tokenizer, self.prompt)
        actions = np.pad(actions, ((0, 0), (0, self.action_dim - 7))).astype(np.float32)
        observation = {"image": images, "image_mask": {key: torch.tensor(key != "right_wrist_0_rgb") for key in images},
                       "state": torch.from_numpy(state)}
        if self.tokenizer is not None:
            observation["tokenized_prompt"] = torch.from_numpy(tokens.astype(np.int64))
            observation["tokenized_prompt_mask"] = torch.from_numpy(mask.astype(bool))
        return observation, torch.from_numpy(actions)


class ValidationSampler(Sampler):
    """Partition validation without padding duplicate samples across ranks."""
    def __init__(self, dataset, rank, world_size):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def create_robomimic_dataloaders(hdf5_path, batch_size=64, val_ratio=0.1, num_workers=4,
                                seed=42, action_horizon=50, action_dim=32, tokenizer=None,
                                prompt="Pick up the can and place it in the bin.",
                                rank=0, world_size=1, stats=None, split=None, dataset_type="ph", metadata_path=None):
    validate_data(hdf5_path)
    split = make_episode_split(hdf5_path, dataset_type, val_ratio, seed, split, metadata_path)
    stats = stats if stats is not None else compute_action_stats(hdf5_path, split["train"])
    kwargs = dict(action_stats=stats, action_horizon=action_horizon, action_dim=action_dim,
                  tokenizer=tokenizer, prompt=prompt)
    train = RobomimicHDF5Dataset(hdf5_path, split["train"], **kwargs)
    val = RobomimicHDF5Dataset(hdf5_path, split["val"], **kwargs)
    sampler = DistributedSampler(train, world_size, rank, shuffle=True, seed=seed, drop_last=True)
    if len(sampler) < batch_size:
        raise ValueError("Training split is too small for one global batch")
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers:
        loader_kwargs["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train, sampler=sampler, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val, sampler=ValidationSampler(val, rank, world_size), **loader_kwargs)
    train_loader.normalization_stats, train_loader.episode_split = stats, split
    return train_loader, val_loader


def save_action_stats(hdf5_path, output_path, episodes=None):
    Path(output_path).write_text(json.dumps(compute_action_stats(hdf5_path, episodes), indent=2))

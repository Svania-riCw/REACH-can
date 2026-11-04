"""Fixed-input flow and steering probes; all distances use the 7 robot DOFs."""
from contextlib import contextmanager
import math

from checkpoint_utils import restore_rng, rng_state
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


@contextmanager
def diagnostic_mode(model=None):
    """Keep probes out of the training random stream and restore train/eval mode."""
    state = rng_state()
    was_training = model.training if model is not None else None
    try:
        if model is not None:
            model.eval()
        yield
    finally:
        restore_rng(state)
        if model is not None:
            model.train(was_training)


class FixedProbe:
    def __init__(self, dataset, samples, seed, device, prepare_batch):
        if samples < 1 or not len(dataset):
            raise ValueError("A fixed probe needs positive samples and a nonempty held-out dataset")
        self.seed = seed
        self.indices = torch.linspace(0, len(dataset) - 1, min(samples, len(dataset))).long().tolist()
        self.sample_ids = [list(dataset.index_map[i]) if hasattr(dataset, "index_map") else i
                           for i in self.indices]
        with diagnostic_mode():
            self.observation, self.actions = prepare_batch(default_collate([dataset[i] for i in self.indices]), device)
            generator = torch.Generator(device=device).manual_seed(seed)
            self.noise = torch.randn(self.actions.shape, generator=generator, device=device)
            # Stratified times cover the flow interval even for a small fixed panel.
            count = self.actions.shape[0]
            self.time = (torch.arange(count, device=device) + .5) / count

    def metadata(self):
        return dict(seed=self.seed, sample_indices=self.indices, sample_ids=self.sample_ids,
                    flow_times=self.time.tolist(), physical_action_dims=7,
                    distance_space="normalized training action space")


def checked(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {label}")
    return value


def action_mse(first, second):
    return checked((first[..., :7] - second[..., :7]).square(), "action distance").mean().item()


def flow_metrics(flow, actions):
    velocity = checked(flow["velocity"][..., :7], "flow velocity")
    target = checked(flow["target_velocity"][..., :7], "target velocity")
    residual = velocity - target
    endpoint = flow["x_t"][..., :7] - flow["time"][:, None, None] * velocity
    return dict(mse=checked(residual.square().mean(), "flow loss").item(),
                velocity_cosine=F.cosine_similarity(velocity.flatten(1), target.flatten(1), dim=1).mean().item(),
                relative_velocity_l2=(residual.flatten(1).norm(dim=1) /
                                      target.flatten(1).norm(dim=1).clamp_min(1e-8)).mean().item(),
                endpoint_estimate_mse=action_mse(endpoint, actions))


@torch.no_grad()
def monitor_branches(model, probe, base_action):
    """One panel, identical x_t/noise/time, no image augmentation for all branches."""
    result, velocities = {}, {}
    conditions = {"baseline": None} if base_action is None else {
        "base": base_action.expand_as(probe.actions), "gt": probe.actions}
    if base_action is not None and len(probe.actions) > 1:
        conditions["foreign"] = probe.actions.roll(1, dims=0)
    with diagnostic_mode(model):
        for name, condition in conditions.items():
            kwargs = {} if condition is None else {"action_condition": condition}
            _, flow = model(probe.observation, probe.actions, noise=probe.noise.clone(),
                            time=probe.time.clone(), preprocess_train=False, return_flow=True, **kwargs)
            result[name] = flow_metrics(flow, probe.actions)
            velocities[name] = flow["velocity"].detach()
        if base_action is not None:
            result["condition_effect"] = dict(velocity_mse=action_mse(velocities["gt"], velocities["base"]),
                                               gt_loss_gain=result["base"]["mse"] - result["gt"]["mse"])
    return result


def build_prefix(model, observation):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    images, masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
    embeddings, pad_masks, attention_masks = model.embed_prefix(images, masks, tokens, token_masks)
    mask = model._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, attention_masks))
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=mask, position_ids=torch.cumsum(pad_masks, dim=1) - 1,
        past_key_values=None, inputs_embeds=[embeddings, None], use_cache=True)
    return state, pad_masks, cache


def euler_sample(velocity, noise, base, target=None, scale=0., direction="away", num_steps=10):
    """REACH-style velocity guidance on the same x,t; dt is negative (1 -> 0)."""
    if num_steps < 1 or direction not in {"away", "toward"} or not math.isfinite(scale) or scale < 0:
        raise ValueError("Invalid integration steps, direction, or guidance scale")
    x = noise.clone()
    dt = torch.tensor(-1. / num_steps, device=x.device, dtype=torch.float32)
    time = torch.tensor(1., device=x.device, dtype=torch.float32)
    # Match upstream floating-point time accumulation, not a different time grid.
    for _ in range(num_steps):
        expanded_time = time.expand(len(x))
        v_base = velocity(x, expanded_time, base)
        if target is not None:
            v_target = velocity(x, expanded_time, target)
            delta = v_base - v_target if direction == "away" else v_target - v_base
            v_base = v_base + scale * delta
        x = checked(x + dt * v_base, "Euler state")
        time = time + dt
    return x


def assert_close(first, second, tolerance, label):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Consistency tolerance must be finite and positive")
    error = checked((first - second).abs(), label).max().item()
    if error > tolerance:
        raise AssertionError(f"{label}: max error {error:.6g} exceeds {tolerance:.6g}")
    return error


@torch.no_grad()
def steering_probe(model, probe, base_action, scales=(0., .2, 2.5, 4.5), num_steps=10, tolerance=1e-4):
    if base_action is None:
        raise ValueError("Steering requires a conditional policy")
    base_condition = base_action.expand_as(probe.actions)
    result = dict(consistency={}, follow={}, guidance=[])
    with diagnostic_mode(model):
        prefix = build_prefix(model, probe.observation)

        def velocity(x, time, condition):
            with model._condition_scope(condition):
                return model.denoise_step(*prefix, x, time)

        def sample(condition):
            return euler_sample(velocity, probe.noise, condition, num_steps=num_steps)

        base = sample(base_condition)
        official = model.sample_actions(probe.actions.device, probe.observation, noise=probe.noise.clone(),
                                        num_steps=num_steps, action_condition=base_condition)
        result["consistency"]["custom_vs_upstream_max_abs"] = assert_close(
            base, official, tolerance, "Custom/upstream sampler mismatch")
        # Action history contains 7 physical channels; padding is always zero,
        # even when the model's unsupervised output channels are nonzero.
        self_prompt = base.clone()
        self_prompt[..., 7:] = 0
        targets = {"gt": probe.actions, "self": self_prompt}
        if len(probe.actions) > 1:
            targets["foreign"] = probe.actions.roll(1, dims=0)
        for name, target in targets.items():
            followed = sample(target)
            before, after = action_mse(base, target), action_mse(followed, target)
            result["follow"][name] = dict(base_target_mse=before, followed_target_mse=after,
                                          follow_gain=before - after, output_change_mse=action_mse(followed, base))
        # These are diagnostic prompts, not the observed failed-action history of a rollout.
        for name, target, direction in (("negative_self", self_prompt, "away"),
                                        ("negative_gt", probe.actions, "away"),
                                        ("positive_gt", probe.actions, "toward")):
            for scale in sorted(set([0., *scales])):
                guided = euler_sample(velocity, probe.noise, base_condition, target, scale, direction, num_steps)
                entry = dict(family=name, scale=scale, target_mse=action_mse(guided, target),
                             base_mse=action_mse(guided, base), gt_mse=action_mse(guided, probe.actions))
                if scale == 0:
                    entry["zero_scale_max_abs"] = assert_close(base, guided, tolerance, "S=0/base mismatch")
                result["guidance"].append(entry)
    return result

"""Real multi-GPU full-SFT update and reference parity; base weights remain untouched."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from checkpoint_utils import PROJECT_ROOT, load_model, rng_state, restore_rng
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi_robomimic_adapter import create_robomimic_dataloaders, STATE_KEYS, CAMERAS
from train_pi05 import configure_full_sft, setup_distributed, train_update, prepare_batch, cosine_warmup_lambda
from eval_policy import Policy


def main():
    started = time.time()
    micro_batch = int(os.environ.get('SFT_SMOKE_MICRO_BATCH', '8'))
    accumulation = int(os.environ.get('SFT_SMOKE_ACCUMULATION', '1'))
    torch.set_num_threads(4)
    device, rank, world = setup_distributed('cuda')
    torch.manual_seed(2026 + rank)
    base = PROJECT_ROOT / 'checkpoints/pi05_base_pytorch/model.safetensors'
    model, config, _ = load_model(base, 'pi05_robomimic', device, precision='float32')
    configure_full_sft(model)
    model.gradient_checkpointing_enable()
    tokenizer = PaligemmaTokenizer(config.max_token_len)
    train, val = create_robomimic_dataloaders(PROJECT_ROOT / 'data/paper/can_paired_image.hdf5',
        batch_size=micro_batch, num_workers=0, tokenizer=tokenizer, seed=0, rank=rank, world_size=world,
        dataset_type='paired', metadata_path=PROJECT_ROOT / 'data/robomimic/can/paired/low_dim_v141.hdf5')
    stats = train.normalization_stats
    # Same real sample must produce identical training/inference model inputs.
    name, step = train.dataset.index_map[0]
    with h5py.File(train.dataset.path) as f:
        raw = {key: f[f'data/{name}/obs/{key}'][step] for key in (*STATE_KEYS, *CAMERAS.values())}
    policy = object.__new__(Policy)
    policy.stats, policy.config, policy.tokenizer, policy.device = stats, config, tokenizer, device
    policy.saved = {'prompt': train.dataset.prompt}
    inference_obs = policy.observation(raw)
    training_obs, reference_actions = prepare_batch(default_collate([train.dataset[0]]), device)
    torch.testing.assert_close(inference_obs.state, training_obs.state, rtol=0, atol=0)
    torch.testing.assert_close(inference_obs.tokenized_prompt, training_obs.tokenized_prompt, rtol=0, atol=0)
    for key in training_obs.images:
        torch.testing.assert_close(inference_obs.images[key], training_obs.images[key], rtol=0, atol=0)
    # Compare actual baseline computation to the upstream-equivalent base class,
    # including identical stochastic image augmentation and full32 target loss.
    model.eval()
    noise = torch.randn(reference_actions.shape, device=device,
                        generator=torch.Generator(device=device).manual_seed(2026))
    flow_time = torch.tensor([.5], device=device)
    saved_rng = rng_state()
    with torch.no_grad():
        current_losses = model(training_obs, reference_actions, noise=noise, time=flow_time, preprocess_train=True)
        restore_rng(saved_rng)
        upstream_losses = PI0Pytorch.forward(model, training_obs, reference_actions, noise=noise, time=flow_time)
    torch.testing.assert_close(current_losses, upstream_losses, rtol=0, atol=0)
    assert torch.count_nonzero(reference_actions[..., 7:]) == 0
    assert torch.equal((noise-reference_actions)[..., 7:], noise[..., 7:])
    reference_parity = dict(max_absolute_loss_difference=float((current_losses-upstream_losses).abs().max()),
        full32_loss=float(current_losses.mean()), physical7_loss=float(current_losses[..., :7].mean()),
        padded25_loss=float(current_losses[..., 7:].mean()),
        zero_padded_actions_have_noise_velocity_target=True,
        reference='PI0Pytorch.forward; same weights, real sample, fixed noise/time and identical image augmentation')
    tracked = {name: parameter for name, parameter in model.named_parameters() if
               name.endswith('vision_model.embeddings.patch_embedding.weight') or
               name.endswith('paligemma.model.language_model.layers.0.self_attn.q_proj.weight') or
               name == 'action_out_proj.weight'}
    assert len(tracked) == 3, list(tracked)
    before = {name: parameter.detach().clone() for name, parameter in tracked.items()}
    wrapper = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
                                     gradient_as_bucket_view=True, broadcast_buffers=False) if world > 1 else model
    planned_steps = 50 * (len(train) // accumulation)
    lr = 2.5e-5 * cosine_warmup_lambda(0, 1000, planned_steps, .1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                 betas=(.9, .95), weight_decay=1e-10, foreach=False)
    iterator = iter(train)
    wrapper.train()
    loss = train_update(wrapper, [next(iterator) for _ in range(accumulation)], optimizer, device)
    changes = {name: {'changed_elements': int(torch.count_nonzero(parameter.detach() != before[name])),
                       'delta_norm': float((parameter.detach() - before[name]).norm()),
                       'gradient_norm': float(parameter.grad.norm())}
               for name, parameter in tracked.items()}
    assert all(value['changed_elements'] > 0 for value in changes.values()), changes
    assert model.action_out_proj.weight.grad[:7].norm() > 0
    assert model.action_out_proj.weight.grad[7:].norm() == 0
    checksums = [None] * world
    checksum = {name: parameter.detach().double().sum().item() for name, parameter in tracked.items()}
    if world > 1:
        dist.all_gather_object(checksums, checksum)
        assert all(value == checksums[0] for value in checksums), checksums
    else:
        checksums = [checksum]
    rank_report = dict(rank=rank, loss=float(loss), peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                       forward_reference_parity=reference_parity)
    ranks = [None] * world
    if world > 1:
        dist.all_gather_object(ranks, rank_report)
    else:
        ranks = [rank_report]
    result = dict(status='passed', base=str(base), long_training_started=False,
        training_objective='mean flow loss on physical action dimensions0:7; padding dimensions7:32 masked',
        world_size=world, effective_batch=world*micro_batch*accumulation,
        accumulation_steps=accumulation, micro_batch=micro_batch,
        planned_50_epoch_optimizer_steps=planned_steps,
        loss=float(loss), precision=config.dtype, lr=lr,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        vlm_trainable_parameters=sum(p.numel() for p in model.paligemma_with_expert.paligemma.parameters() if p.requires_grad),
        train_episodes=len(train.dataset.demo_names), validation_episodes=len(val.dataset.demo_names),
        normalization=stats['normalization'], state_tokenization=stats['state_tokenization'],
        training_inference_inputs_identical=True, parameter_updates=changes, rank_checksums=checksums,
        forward_reference_parity=reference_parity, per_rank=ranks,
        padded_output_gradient_norm=float(model.action_out_proj.weight.grad[7:].norm()),
        peak_gpu_gib=max(value['peak_gpu_gib'] for value in ranks), seconds=time.time()-started)
    if rank == 0:
        output = PROJECT_ROOT / 'outputs/validation/full_sft_physical7_v4'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'gpu_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
        (output / f'gpu_smoke_micro{micro_batch}_accum{accumulation}.json').write_text(json.dumps(result, indent=2) + '\n')
        (output / f'gpu_smoke_world{world}_micro{micro_batch}_accum{accumulation}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

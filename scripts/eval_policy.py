"""Inference uses the checkpoint's exact prompt, state scales, and image layout."""
import json
from pathlib import Path

from checkpoint_utils import load_model, load_payload, resolve_checkpoint
import numpy as np
import torch
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi_robomimic_adapter import CAMERAS, normalize, denormalize, encode_state
from policy_diagnostics import build_prefix, euler_sample
from train_pi05 import parse_base_action, prepare_batch


def denormalize_actions(values,stats):
    values=np.asarray(values)[...,:7]
    return denormalize(values,stats)


def normalize_action_prompt(values,stats,horizon,action_dim):
    values=np.asarray(values,dtype=np.float32)
    if values.shape!=(horizon,7):raise ValueError('Negative prompt must contain the complete predicted [horizon,7] plan')
    if not np.isfinite(values).all():raise ValueError('Negative prompt contains nonfinite actions')
    if action_dim<7:raise ValueError('Action dimension must include seven physical controls')
    normalized=normalize(values,stats)
    return np.pad(normalized,((0,0),(0,action_dim-7)))


class Policy:
    def __init__(self,checkpoint,device='cuda',num_steps=10,disable_action_conditioning=False):
        path=resolve_checkpoint(checkpoint)
        config_path=(path.parent.parent if path.parent.name=='checkpoints' else path.parent)/'config.json'
        saved=json.loads(config_path.read_text()) if config_path.exists() else load_payload(path)['args']
        self.conditional=saved['policy_mode']=='conditional'
        if disable_action_conditioning and not self.conditional:
            raise ValueError('Action-conditioning ablation requires a conditional checkpoint')
        self.action_conditioning_enabled=self.conditional and not disable_action_conditioning
        self.device=torch.device(device)
        if self.device.type=='cuda':torch.cuda.set_device(self.device.index or 0)
        self.model,self.config,self.payload=load_model(path,'pi05_robomimic',self.device,enable_ac=self.conditional)
        self.model.eval()
        self.saved=self.payload['args']
        if self.saved['policy_mode']!=saved['policy_mode']:raise ValueError('Checkpoint mode/config mismatch')
        self.stats=self.payload['normalization_stats']
        self.tokenizer=PaligemmaTokenizer(self.config.max_token_len)
        self.base=parse_base_action(self.saved['base_action'],self.config.action_dim,self.device)
        self.num_steps=num_steps

    def observation(self,raw):
        state=np.concatenate([raw[k] for k in self.stats['state_keys']])
        state,tokens,mask=encode_state(state,self.stats,self.config.action_dim,self.tokenizer,self.saved['prompt'])
        images={target:torch.from_numpy(raw[source].copy()).permute(2,0,1).float()[None]/127.5-1
                for target,source in CAMERAS.items()}
        images['right_wrist_0_rgb']=torch.zeros_like(images['left_wrist_0_rgb'])
        obs=dict(image=images,image_mask={k:torch.tensor([k!='right_wrist_0_rgb']) for k in images},
                 state=torch.from_numpy(state)[None],tokenized_prompt=torch.from_numpy(tokens.astype(np.int64))[None],
                 tokenized_prompt_mask=torch.from_numpy(mask.astype(bool))[None])
        return prepare_batch((obs,torch.zeros(1,self.config.action_horizon,self.config.action_dim)),self.device)[0]

    @torch.no_grad()
    def predict(self,raw,noise_seed,failed=None,scale=0.):
        if failed is not None and not self.action_conditioning_enabled and scale!=0:
            raise ValueError('Failed prompts require enabled action conditioning')
        observation=self.observation(raw)
        shape=(1,self.config.action_horizon,self.config.action_dim)
        noise=torch.randn(shape,device=self.device,generator=torch.Generator(device=self.device).manual_seed(noise_seed))
        base=self.base.expand(shape) if self.action_conditioning_enabled else None
        if failed is None or scale==0:
            kwargs={} if base is None else dict(action_condition=base)
            actions=self.model.sample_actions(self.device,observation,noise=noise,num_steps=self.num_steps,**kwargs)
        else:
            normalized=normalize_action_prompt(failed,self.stats['action'],self.config.action_horizon,self.config.action_dim)
            negative=torch.as_tensor(normalized,device=self.device)[None]
            prefix=build_prefix(self.model,observation)
            def velocity(x,time,condition):
                with self.model._condition_scope(condition):
                    return self.model.denoise_step(*prefix,x,time)
            actions=euler_sample(velocity,noise,base,negative,scale,'away',self.num_steps)
        raw_actions=denormalize_actions(actions[0].float().cpu().numpy(),self.stats['action'])
        if not np.isfinite(raw_actions).all():raise FloatingPointError('Nonfinite sampled actions')
        return dict(action=np.clip(raw_actions,-1,1),action_pred=raw_actions,
                    clipped_predictions=int(np.count_nonzero(np.abs(raw_actions)>1)))

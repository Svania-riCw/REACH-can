"""Architecture compatible with official pi0.5; pad Robomimic's 7 actions to 32."""
from openpi.models.pi0_config import Pi0Config


def get_config(name="pi05_robomimic"):
    if name not in {"pi05_robomimic", "pi05_robomimic_ph", "pi05_robomimic_mh"}:
        raise ValueError(f"Unknown Robomimic model config: {name}")
    return Pi0Config(pi05=True, action_dim=32, action_horizon=50,
                     max_token_len=200, pytorch_compile_mode=None)

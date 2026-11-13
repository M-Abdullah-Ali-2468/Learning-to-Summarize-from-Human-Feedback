
import sys
sys.path.insert(0, "./starter")

import torch
import torch.nn as nn

from pathlib import Path

from starter.models import load_tokenizer, load_causal_lm


# -----------------------------
# 1. Device
# -----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 2. Reward Model Architecture
# -----------------------------

class RewardModel(nn.Module):

    def __init__(self, base_model):

        super().__init__()

        self.transformer = base_model.model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1).to(DEVICE)

        self.config = base_model.config


# -----------------------------
# 3. Load All Models
# -----------------------------

def load_all_models():

    print("Loading tokenizer...")

    tok = load_tokenizer()

    print("Loading SFT model...")
    sft_model = load_causal_lm(name_or_path="checkpoints/sft", device=DEVICE)

    print("Loading PPO model...")
    ppo_model = load_causal_lm(name_or_path="checkpoints/ppo", device=DEVICE)

    print("Loading DPO model...")
    dpo_model = load_causal_lm(name_or_path="checkpoints/dpo", device=DEVICE)

    print("Loading Reward Model...")

    reward_checkpoint = torch.load("checkpoints/reward/reward_model.pt", map_location=DEVICE, weights_only=True)

    base_model_path = reward_checkpoint["base_model_path"]

    reward_base = load_causal_lm(name_or_path=base_model_path, device=DEVICE)

    reward_model = RewardModel(reward_base)

    reward_model.load_state_dict(reward_checkpoint["model_state_dict"])

    del reward_base

    # Evaluation mode

    sft_model.eval()
    ppo_model.eval()
    dpo_model.eval()
    reward_model.eval()

    # Freeze all models

    sft_model.requires_grad_(False)
    ppo_model.requires_grad_(False)
    dpo_model.requires_grad_(False)
    reward_model.requires_grad_(False)

    print("\nAll models loaded successfully!")

    return {
        "tokenizer": tok,
        "sft": sft_model,
        "ppo": ppo_model,
        "dpo": dpo_model,
        "reward": reward_model,
        "device": DEVICE
    }

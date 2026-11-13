
import sys
sys.path.insert(0, "./starter")

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader

from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_sft_splits
from starter.batching import collate_prompts


# -----------------------------
# 1. Configuration
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_EPSILON = 0.2
BETA_KL = 0.05
VALUE_COEF = 0.1
ENTROPY_COEF = 0.01
GAMMA = 1.0

PPO_EPOCHS = 4
LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 1.0

ROLLOUT_SIZE = 16
UPDATE_MICRO_BATCH_SIZE = 8
MAX_NEW_TOKENS = 48
TOTAL_ITERATIONS = 200

SFT_CHECKPOINT = "checkpoints/sft"
REWARD_CHECKPOINT = Path("checkpoints/reward")
PPO_CHECKPOINT = Path("checkpoints/ppo")


# -----------------------------
# 2. Reward Model Architecture
# -----------------------------

class RewardModel(nn.Module):

    def __init__(self, base_model):

        super().__init__()

        self.transformer = base_model.model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1).to(device)

        self.config = base_model.config


# -----------------------------
# 3. Load Reward Model
# -----------------------------

def load_reward_model():

    base_model = load_causal_lm(name_or_path=SFT_CHECKPOINT, device=device)

    reward_model = RewardModel(base_model)

    del base_model

    checkpoint = torch.load(REWARD_CHECKPOINT / "reward_model.pt", map_location=device, weights_only=True)

    # Support both old and new checkpoint formats

    if "model_state_dict" in checkpoint:

        reward_model.load_state_dict(checkpoint["model_state_dict"])

    else:

        reward_model.transformer.load_state_dict(checkpoint["transformer"])
        reward_model.reward_head.load_state_dict(checkpoint["reward_head"])

    reward_model.to(device)
    reward_model.requires_grad_(False)
    reward_model.eval()

    return reward_model


# -----------------------------
# 4. Load Tokenizer and Dataset
# -----------------------------

print("Loading tokenizer and dataset...")

tok = load_tokenizer()

sft_train, sft_val = load_sft_splits(tok)

prompt_loader = DataLoader(sft_train, batch_size=ROLLOUT_SIZE, shuffle=True, collate_fn=partial(collate_prompts, pad_id=tok.pad_token_id))

prompt_iterator = iter(prompt_loader)

print("Training prompts:", len(sft_train))


# -----------------------------
# 5. Load Models
# -----------------------------

print("\nLoading Policy Model...")

policy_model = load_causal_lm(name_or_path=SFT_CHECKPOINT, device=device)

hidden_size = policy_model.config.hidden_size

value_head = nn.Linear(hidden_size, 1).to(device)

print("Policy Model loaded!")
print("Value Head created!")


print("\nLoading Reference Model...")

reference_model = load_causal_lm(name_or_path=SFT_CHECKPOINT, device=device)

reference_model.requires_grad_(False)
reference_model.eval()

print("Reference Model loaded and frozen!")


print("\nLoading Reward Model...")

reward_model = load_reward_model()

print("Reward Model loaded and frozen!")


# -----------------------------
# 6. Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(list(policy_model.parameters()) + list(value_head.parameters()), lr=LEARNING_RATE)


# -----------------------------
# 7. Position IDs
# -----------------------------

def get_position_ids(attention_mask):

    position_ids = attention_mask.long().cumsum(dim=-1) - 1

    position_ids.masked_fill_(attention_mask == 0, 1)

    return position_ids


# -----------------------------
# 8. Generate Rollout
# -----------------------------

def generate_rollout(prompts):

    input_ids = prompts["input_ids"].to(device)
    attention_mask = prompts["attention_mask"].to(device)

    policy_model.eval()

    with torch.no_grad():

        outputs = policy_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    generated_ids = outputs[:, input_ids.shape[1]:]

    generated_mask = torch.ones_like(generated_ids)

    for i in range(generated_ids.size(0)):

        eos_positions = (generated_ids[i] == tok.eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:

            end_position = eos_positions[0].item() + 1

            generated_mask[i, end_position:] = 0

    full_ids = torch.cat([input_ids, generated_ids], dim=1)

    full_mask = torch.cat([attention_mask, generated_mask], dim=1)

    return input_ids, attention_mask, generated_ids, generated_mask, full_ids, full_mask


# -----------------------------
# 9. Calculate Policy Statistics
# -----------------------------

def get_policy_stats(model, full_ids, full_mask, generated_ids, prompt_width, value_head=None, calculate_entropy=False):

    gen_length = generated_ids.size(1)

    position_ids = get_position_ids(full_mask)

    outputs = model.model(input_ids=full_ids, attention_mask=full_mask, position_ids=position_ids, use_cache=False)

    hidden_states = outputs.last_hidden_state

    # Hidden state immediately before each generated token

    action_hidden = hidden_states[:, prompt_width - 1:prompt_width + gen_length - 1, :]

    logits = model.lm_head(action_hidden)

    log_probs = F.log_softmax(logits.float(), dim=-1)

    token_log_probs = log_probs.gather(dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)

    values = None

    if value_head is not None:

        values = value_head(action_hidden.to(value_head.weight.dtype)).squeeze(-1)

    entropy = None

    if calculate_entropy:

        probabilities = log_probs.exp()

        entropy = -(probabilities * log_probs).sum(dim=-1)

    return token_log_probs, values, entropy


# -----------------------------
# 10. Calculate Reward Scores
# -----------------------------

def get_reward_scores(full_ids, full_mask):

    position_ids = get_position_ids(full_mask)

    with torch.no_grad():

        outputs = reward_model.transformer(input_ids=full_ids, attention_mask=full_mask, position_ids=position_ids, use_cache=False)

        hidden_states = outputs.last_hidden_state

        positions = torch.arange(full_mask.size(1), device=device).unsqueeze(0)

        last_indices = (positions * full_mask.long()).max(dim=1).values

        batch_indices = torch.arange(hidden_states.size(0), device=device)

        last_hidden = hidden_states[batch_indices, last_indices]

        scores = reward_model.reward_head(last_hidden.to(reward_model.reward_head.weight.dtype))

    return scores.squeeze(-1)


# -----------------------------
# 11. Collect Rollout Statistics
# -----------------------------

def collect_rollout_statistics(generated_ids, generated_mask, full_ids, full_mask, prompt_width):

    old_log_probs_list = []
    reference_log_probs_list = []
    old_values_list = []

    policy_model.eval()
    reference_model.eval()

    with torch.no_grad():

        for start in range(0, generated_ids.size(0), UPDATE_MICRO_BATCH_SIZE):

            end = start + UPDATE_MICRO_BATCH_SIZE

            batch_ids = full_ids[start:end]
            batch_mask = full_mask[start:end]
            batch_generated = generated_ids[start:end]

            old_log_probs, old_values, _ = get_policy_stats(policy_model, batch_ids, batch_mask, batch_generated, prompt_width, value_head=value_head)

            reference_log_probs, _, _ = get_policy_stats(reference_model, batch_ids, batch_mask, batch_generated, prompt_width)

            old_log_probs_list.append(old_log_probs)
            old_values_list.append(old_values)
            reference_log_probs_list.append(reference_log_probs)

    old_log_probs = torch.cat(old_log_probs_list, dim=0)
    old_values = torch.cat(old_values_list, dim=0)
    reference_log_probs = torch.cat(reference_log_probs_list, dim=0)

    reward_scores = get_reward_scores(full_ids, full_mask)

    return old_log_probs, reference_log_probs, old_values, reward_scores


# -----------------------------
# 12. Calculate Rewards, Returns and Advantages
# -----------------------------

def calculate_returns_and_advantages(old_log_probs, reference_log_probs, old_values, reward_scores, generated_mask):

    # KL penalty for every generated token

    token_rewards = -BETA_KL * (old_log_probs - reference_log_probs)

    token_rewards = token_rewards * generated_mask

    # Add reward model score to the final valid token

    lengths = generated_mask.sum(dim=1).long()

    last_indices = lengths - 1

    batch_indices = torch.arange(generated_mask.size(0), device=device)

    token_rewards[batch_indices, last_indices] += reward_scores

    # Calculate discounted returns

    returns = torch.zeros_like(token_rewards)

    running_return = torch.zeros(token_rewards.size(0), device=device)

    for t in reversed(range(token_rewards.size(1))):

        running_return = token_rewards[:, t] + GAMMA * running_return

        running_return = running_return * generated_mask[:, t]

        returns[:, t] = running_return

    # Calculate advantages

    advantages = returns - old_values

    # Normalize across all valid generated tokens

    valid_advantages = advantages[generated_mask.bool()]

    advantage_mean = valid_advantages.mean()
    advantage_std = valid_advantages.std(unbiased=False)

    advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

    advantages = advantages * generated_mask

    return token_rewards, returns, advantages


# -----------------------------
# 13. PPO Update
# -----------------------------

def ppo_update(full_ids, full_mask, generated_ids, generated_mask, prompt_width, old_log_probs, old_values, returns, advantages):

    total_valid_tokens = generated_mask.sum().clamp(min=1)

    policy_model.train()
    value_head.train()

    final_policy_loss = 0.0
    final_value_loss = 0.0
    final_entropy = 0.0
    final_clip_fraction = 0.0

    for epoch in range(PPO_EPOCHS):

        optimizer.zero_grad()

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        clipped_tokens = 0.0

        for start in range(0, generated_ids.size(0), UPDATE_MICRO_BATCH_SIZE):

            end = start + UPDATE_MICRO_BATCH_SIZE

            batch_ids = full_ids[start:end]
            batch_mask = full_mask[start:end]
            batch_generated = generated_ids[start:end]
            batch_generated_mask = generated_mask[start:end]

            batch_old_log_probs = old_log_probs[start:end]
            batch_returns = returns[start:end]
            batch_advantages = advantages[start:end]

            current_log_probs, current_values, entropy = get_policy_stats(policy_model, batch_ids, batch_mask, batch_generated, prompt_width, value_head=value_head, calculate_entropy=True)

            # Probability ratio

            ratios = torch.exp(current_log_probs - batch_old_log_probs)

            # PPO clipped objective

            unclipped = ratios * batch_advantages

            clipped = torch.clamp(ratios, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages

            policy_loss_tokens = -torch.minimum(unclipped, clipped)

            # Value loss

            value_loss_tokens = (current_values - batch_returns) ** 2

            # Entropy bonus

            entropy_tokens = entropy

            mask = batch_generated_mask

            policy_loss_sum = policy_loss_sum + (policy_loss_tokens * mask).sum().item()

            value_loss_sum = value_loss_sum + (value_loss_tokens * mask).sum().item()

            entropy_sum = entropy_sum + (entropy_tokens * mask).sum().item()

            clipped_tokens += (((ratios < 1.0 - CLIP_EPSILON) | (ratios > 1.0 + CLIP_EPSILON)).float() * mask).sum().item()

            # Divide by total rollout tokens so both halves accumulate to the full-batch mean

            loss = ((policy_loss_tokens * mask).sum() + VALUE_COEF * (value_loss_tokens * mask).sum() - ENTROPY_COEF * (entropy_tokens * mask).sum()) / total_valid_tokens

            loss.backward()

        # One optimizer update after both halves

        torch.nn.utils.clip_grad_norm_(list(policy_model.parameters()) + list(value_head.parameters()), max_norm=MAX_GRAD_NORM)

        optimizer.step()

        final_policy_loss = policy_loss_sum / total_valid_tokens.item()
        final_value_loss = value_loss_sum / total_valid_tokens.item()
        final_entropy = entropy_sum / total_valid_tokens.item()
        final_clip_fraction = clipped_tokens / total_valid_tokens.item()

    return final_policy_loss, final_value_loss, final_entropy, final_clip_fraction


# -----------------------------
# 14. Get Next Prompt Batch
# -----------------------------

def get_next_prompt_batch():

    global prompt_iterator

    try:

        prompts = next(prompt_iterator)

    except StopIteration:

        prompt_iterator = iter(prompt_loader)

        prompts = next(prompt_iterator)

    return prompts


# -----------------------------
# 15. Main PPO Training Loop
# -----------------------------

def main():

    print("\nStarting PPO Training...\n")

    for iteration in range(1, TOTAL_ITERATIONS + 1):

        # Step 1: Get 16 prompts

        prompts = get_next_prompt_batch()

        # Step 2: Generate summaries

        input_ids, attention_mask, generated_ids, generated_mask, full_ids, full_mask = generate_rollout(prompts)

        prompt_width = input_ids.size(1)

        # Step 3: Collect old policy, reference and value predictions

        old_log_probs, reference_log_probs, old_values, reward_scores = collect_rollout_statistics(generated_ids, generated_mask, full_ids, full_mask, prompt_width)

        # Step 4: Calculate rewards, returns and advantages

        token_rewards, returns, advantages = calculate_returns_and_advantages(old_log_probs, reference_log_probs, old_values, reward_scores, generated_mask)

        # Step 5: PPO optimization for 4 epochs

        policy_loss, value_loss, entropy, clip_fraction = ppo_update(full_ids, full_mask, generated_ids, generated_mask, prompt_width, old_log_probs, old_values, returns, advantages)

        # Step 6: Calculate metrics

        mean_reward = reward_scores.mean().item()

        mean_kl = ((old_log_probs - reference_log_probs) * generated_mask).sum(dim=1).mean().item()

        mean_length = generated_mask.sum(dim=1).float().mean().item()

        print(f"\nPPO Iteration: {iteration}/{TOTAL_ITERATIONS}")
        print(f"Mean Reward Score: {mean_reward:.4f}")
        print(f"Mean KL to Reference: {mean_kl:.4f}")
        print(f"Mean Summary Length: {mean_length:.2f}")
        print(f"Clip Fraction: {clip_fraction:.4f}")
        print(f"Policy Loss: {policy_loss:.4f}")
        print(f"Value Loss: {value_loss:.4f}")
        print(f"Entropy: {entropy:.4f}")

    # -----------------------------
    # Save Final Policy
    # -----------------------------

    print("\nSaving final PPO Policy...")

    PPO_CHECKPOINT.mkdir(parents=True, exist_ok=True)

    policy_model.save_pretrained(PPO_CHECKPOINT, safe_serialization=True)

    tok.save_pretrained(PPO_CHECKPOINT)

    print("PPO Policy saved successfully!")
    print(f"Checkpoint location: {PPO_CHECKPOINT}")


if __name__ == "__main__":
    main()

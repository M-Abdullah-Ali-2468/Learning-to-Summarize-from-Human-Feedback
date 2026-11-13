
import torch


# -----------------------------
# 1. Generation Configuration
# -----------------------------

MAX_NEW_TOKENS = 48


# -----------------------------
# 2. Prepare Input Prompt
# -----------------------------

def prepare_prompt(post):

    post = post.strip()

    if not post.endswith("TL;DR:"):
        post = post + "\nTL;DR:"

    return post


# -----------------------------
# 3. Generate Summary
# -----------------------------

def generate_summary(model, prompt, tok, device):

    model.eval()

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=448).to(device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():

        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    generated_ids = outputs[0, input_length:]

    # Stop at first EOS token

    eos_positions = (generated_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]

    if len(eos_positions) > 0:

        generated_ids = generated_ids[:eos_positions[0].item() + 1]

    summary = tok.decode(generated_ids, skip_special_tokens=True)

    return summary.strip(), generated_ids


# -----------------------------
# 4. Calculate Reward Score
# -----------------------------

def calculate_reward(reward_model, prompt, generated_ids, tok, device):

    reward_model.eval()

    prompt_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=448)["input_ids"].to(device)

    generated_ids = generated_ids.unsqueeze(0).to(device)

    full_ids = torch.cat([prompt_ids, generated_ids], dim=1)

    attention_mask = torch.ones_like(full_ids)

    with torch.no_grad():

        outputs = reward_model.transformer(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)

        hidden_states = outputs.last_hidden_state

        last_hidden = hidden_states[:, -1, :]

        reward = reward_model.reward_head(last_hidden.to(reward_model.reward_head.weight.dtype))

    return reward.squeeze().item()


# -----------------------------
# 5. Compare All Three Models
# -----------------------------

def compare_models(post, models):

    tok = models["tokenizer"]
    device = models["device"]

    prompt = prepare_prompt(post)

    results = {}

    for model_name in ["sft", "ppo", "dpo"]:

        print(f"Generating {model_name.upper()} summary...")

        model = models[model_name]

        summary, generated_ids = generate_summary(model, prompt, tok, device)

        reward_score = calculate_reward(models["reward"], prompt, generated_ids, tok, device)

        results[model_name] = {
            "summary": summary,
            "reward_score": reward_score
        }

    return results

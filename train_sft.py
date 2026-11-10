
import sys
sys.path.insert(0, "./starter")

import torch
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader

from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_sft_splits
from starter.batching import collate_sft
from common import token_log_probs


# -----------------------------
# 1. Load Dataset
# -----------------------------

print("Loading dataset...")

tok = load_tokenizer()
sft_train, sft_val = load_sft_splits(tok)

pad = tok.pad_token_id

print("Dataset loaded successfully!")
print(f"Training examples: {len(sft_train)}")
print(f"Validation examples: {len(sft_val)}")


# -----------------------------
# 2. Create DataLoaders
# -----------------------------

BATCH_SIZE = 8  # Use 8 for debugging; assignment requires 16

sft_train_loader = DataLoader(
    sft_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=partial(collate_sft, pad_id=pad)
)

sft_val_loader = DataLoader(
    sft_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(collate_sft, pad_id=pad)
)

print("DataLoaders created successfully!")


# -----------------------------
# 3. Validation Function
# -----------------------------

def evaluate_sft(model, loader):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():

        for step, batch in enumerate(loader, start=1):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)

            token_log_probs, summary_mask = token_log_probs_fn(
                model,
                input_ids,
                attention_mask,
                prompt_lens
            )

            batch_loss = -(token_log_probs * summary_mask).sum()

            total_loss += batch_loss.item()
            total_tokens += summary_mask.sum().item()

    return total_loss / total_tokens


# -----------------------------
# 4. Training Function
# -----------------------------

def train_one_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0.0
    total_tokens = 0

    for step, batch in enumerate(loader, start=1):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)

        optimizer.zero_grad()

        token_log_probs, summary_mask = token_log_probs_fn(
            model,
            input_ids,
            attention_mask,
            prompt_lens
        )

        loss = -(token_log_probs * summary_mask).sum() / summary_mask.sum()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item() * summary_mask.sum().item()
        )

        total_tokens += summary_mask.sum().item()

        if step % 25 == 0 or step == 1:
            print(
                f"Training progress: {step}/{len(loader)} batches completed"
            )

        if step % 250 == 0:

            print("\nChecking model performance...")

            val_loss = evaluate_sft(model, sft_val_loader)

            print(f"Training Loss: {total_loss / total_tokens:.4f}")
            print(f"Validation Loss: {val_loss:.4f}\n")

            model.train()

    return total_loss / total_tokens


# -----------------------------
# 5. Main Function
# -----------------------------

def main():

    global device, token_log_probs_fn

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    token_log_probs_fn = token_log_probs

    print("\nLoading model...")

    model = load_causal_lm(device=device)

    print("Model loaded successfully!")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5
    )

    print("\nStarting SFT training...\n")

    train_loss = train_one_epoch(
        model,
        sft_train_loader,
        optimizer
    )

    print("\nTraining completed successfully!")
    print(f"Final Training Loss: {train_loss:.4f}")

    print("\nSaving model...")

    save_path = Path("checkpoints/sft")
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)

    print("Model saved successfully!")
    print(f"Checkpoint location: {save_path}")


if __name__ == "__main__":
    main()

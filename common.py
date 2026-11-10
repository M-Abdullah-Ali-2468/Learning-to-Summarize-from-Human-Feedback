import torch
import torch.nn.functional as F


def token_log_probs(model, input_ids, attention_mask, prompt_lens):

    # 1. Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    logits = outputs.logits

    # 2. Shift logits and targets
    shift_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    # 3. Convert logits into log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # 4. Extract actual target token log probabilities
    token_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    # 5. Create summary mask
    positions = torch.arange(
        1,
        input_ids.size(1),
        device=input_ids.device
    )

    summary_mask = (
        (positions.unsqueeze(0) >= prompt_lens.unsqueeze(1))
        & attention_mask[:, 1:].bool()
    )

    return token_log_probs, summary_mask
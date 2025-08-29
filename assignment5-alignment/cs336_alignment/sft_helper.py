import torch

from transformers import PreTrainedTokenizer


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer):
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    """
    # Encode the prompt and output strings
    prompt_ids = [tokenizer.encode(prompt_str, add_special_tokens=False) for prompt_str in prompt_strs]
    output_ids = [tokenizer.encode(output_str, add_special_tokens=False) for output_str in output_strs]
    # Concatenate the prompt and output strings and padding to the same length
    prompt_and_output_ids = [prompt_id + output_id for prompt_id, output_id in zip(prompt_ids, output_ids)]
    max_len = max(len(ids) for ids in prompt_and_output_ids)
    prompt_and_output_ids = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in prompt_and_output_ids]
    # Slice off the final token
    input_ids = [prompt_and_output_id[:-1] for prompt_and_output_id in prompt_and_output_ids]
    # Construct the labels
    labels = [prompt_and_output_id[1:] for prompt_and_output_id in prompt_and_output_ids]
    # Construct the response mask
    response_mask = [
        [0] * (len(prompt_ids[i]))
        + [1] * (len(output_ids[i]))
        + [0] * (max_len - len(prompt_ids[i]) - len(output_ids[i]))
        for i in range(len(prompt_ids))
    ]
    response_mask = [mask[1:] for mask in response_mask]
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "response_mask": torch.tensor(response_mask),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = torch.log_softmax(logits, dim=-1)  # (B, L, V)
    probs = torch.exp(log_probs)  # (B, L, V)
    return -torch.sum(probs * log_probs, dim=-1)  # (B, L)


def get_response_log_probs(
    model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool
) -> torch.Tensor:
    """
    Get the conditional log-probs of the response given the prompt,
    and optionally the entropy of the next token predictions.
    """
    logits = model(input_ids).logits  # (B, L, V)

    log_softmax = torch.log_softmax(logits, dim=-1)  # (B, L, V)
    # labels: (B, L)
    labels_log_probs = log_softmax.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

    result = {"log_probs": labels_log_probs}
    if return_token_entropy:
        entropy = compute_entropy(logits)  # (B, L)
        result["token_entropy"] = entropy

    return result


def masked_normalize(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None, normalize_constant: float = 1.0
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.
    """
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return torch.sum(masked_tensor, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    masked_normalize_probs = masked_normalize(policy_log_probs, response_mask, -1, normalize_constant)
    loss = -masked_normalize_probs.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {"policy_log_probs_grad": policy_log_probs.grad}

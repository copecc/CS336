import torch

from typing import Callable, Literal


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    """
    raw_rewards = [
        reward_fn(response, ground_truth)["reward"]
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(raw_rewards)

    raw_rewards_grouped = raw_rewards.view(-1, group_size)
    group_means = raw_rewards_grouped.mean(dim=1, keepdim=True)
    advantages = raw_rewards_grouped - group_means

    if normalize_by_std:
        group_stds = raw_rewards_grouped.std(dim=1, keepdim=True)
        advantages = advantages / (group_stds + advantage_eps)

    advantages = advantages.view(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
    }
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute policy gradient loss using either raw rewards or advantages.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor, policy_log_probs: torch.Tensor, old_log_probs: torch.Tensor, cliprange: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO-Clip loss.
    """
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_loss = clipped_ratio * advantages
    unclipped_loss = ratio * advantages

    loss = -torch.min(clipped_loss, unclipped_loss)
    is_clipped = (clipped_loss > unclipped_loss).float()

    metadata = {
        "is_clipped": is_clipped,
        "ratio": ratio,
        "unclipped_loss": unclipped_loss,
        "clipped_loss": clipped_loss,
    }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    match loss_type:
        case "no_baseline":
            assert raw_rewards is not None, "raw_rewards is required for no_baseline"
            return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
        case "reinforce_with_baseline":
            assert advantages is not None, "advantages is required for reinforce_with_baseline"
            return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
        case "grpo_clip":
            assert advantages is not None, "advantages is required for grpo_clip"
            assert old_log_probs is not None, "old_log_probs is required for grpo_clip"
            assert cliprange is not None, "cliprange is required for grpo_clip"
            return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        case _:
            raise ValueError(f"Invalid loss type: {loss_type}")


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """
    Compute the mean of the tensor along a dimension, considering only the elements with mask value 1.
    """
    mask = mask.to(dtype=tensor.dtype)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    loss, meta = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    loss = masked_mean(loss, response_mask)
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, meta

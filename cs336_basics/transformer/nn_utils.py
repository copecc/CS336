import torch

from einops import einsum, rearrange
from jaxtyping import Float
from math import sqrt
from torch import Tensor


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Sigmoid Linear Unit (SiLU) activation function.

    Mathematical formula:
        SiLU(x) = x * sigmoid(x)
        where sigmoid(x) = 1 / (1 + exp(-x))

    This function is differentiable and smooth, making it suitable for deep learning models.
    """
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of a tensor along a specified dimension.

    Mathematical formula:
        softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    Numerical stability optimization:
        softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    This prevents overflow when x_i is large, since exp(x_i - max(x)) ≤ 1.
    """
    x = x - x.max(dim=dim, keepdim=True).values
    x = x.exp()
    return x / x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Compute the scaled dot-product attention.

    Mathematical formula:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Where:
        - Q: Query matrix [batch, heads, seq_len, d_k]
        - K: Key matrix [batch, heads, seq_len, d_k]
        - V: Value matrix [batch, heads, seq_len, d_v]
        - d_k: Dimension of keys/queries (for scaling)

    Scaling rationale:
        Without scaling, dot products grow with √d_k, pushing softmax into saturation regions with small gradients.
        Scaling by 1/√d_k normalizes the variance of dot products to be ~1.
    """
    d_k = K.size(-1)
    # QK^T gives similarity between each query and each key
    scores = einsum(Q, K, " ... q d_k, ... k d_k -> ... q k") / sqrt(d_k)

    # set masked positions to -inf so they become 0 after softmax
    if mask is not None:
        scores.masked_fill_(mask == 0, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, V, " ... q k, ... k d_v -> ... q d_v")


def cross_entropy(inputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Compute the cross-entropy loss between the inputs and targets.

    Mathematical derivation:
        Cross-entropy loss = -log(P(target_class))
        Where P(target_class) = softmax(logits)[target_class]

        So: loss = -log(softmax(logits)[target])
                 = -log(exp(logits[target]) / Σ_j exp(logits[j]))
                 = -log(exp(logits[target])) + log(Σ_j exp(logits[j]))
                 = -logits[target] + log(Σ_j exp(logits[j]))
                 = -logits[target] + logsumexp(logits)
                 = -(logits[target] - c) + logsumexp(x) - c
                 = -(logits[target] - c) + logsumexp(x - c)

    Key optimization: Log/Exp cancellation
        Instead of computing: -log(softmax(x)) = -log(exp(x)/sum(exp(x)))
        We directly compute: -x + logsumexp(x)
        This avoids the intermediate exp/log operations and is more numerically stable.

    Numerical stability techniques:
        1. Subtract max(logits) before computing logsumexp to prevent overflow
        2. Use logsumexp instead of log(sum(exp(x))) for better precision
    """
    # reshape inputs to 2D for easier processing
    inputs = rearrange(inputs, "b ... v -> (b ...) v")
    targets = rearrange(targets, "b ... -> (b ...)")
    # doesn't change the final result due to the mathematical properties
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.logsumexp(inputs, dim=-1, keepdim=True)
    log_softmax = inputs - log_sum_exp

    batch_indices = torch.arange(len(targets), device=targets.device)
    # advanced indexing to select log_softmax[i, targets[i]] for each i
    target_log_probs = log_softmax[batch_indices, targets]

    return -target_log_probs.mean()

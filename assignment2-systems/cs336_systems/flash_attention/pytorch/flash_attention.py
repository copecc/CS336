import torch

from torch import Tensor
from jaxtyping import Float, Bool, Int

from .forward import flash_attention_forward
from .backward import standard_attention_backward, flash_attention_backward, flash_attention_backward_two_passes


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " ... Nq d"],
        K: Float[Tensor, " ... Nk d"],
        V: Float[Tensor, " ... Nk d"],
        is_causal: Bool = False,
    ):
        Bq, Bk = 16, 16
        Nq, Nk = Q.shape[-2], K.shape[-2]
        # For elements that are masked out, add the constant value of -1e6
        mask = torch.triu(torch.full((Nq, Nk), -1e6, device=Q.device), diagonal=1) if is_causal else None

        O, L = flash_attention_forward(Q, K, V, Bq, Bk, is_causal, mask)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.Q_TILE_SIZE = Bq
        ctx.K_TILE_SIZE = Bk
        ctx.is_causal = is_causal
        ctx.mask = mask

        return O

    def backward(ctx, dO: Float[Tensor, " ... Nq d"]):
        L, Q, K, V, O = ctx.saved_tensors
        Bq, Bk = ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE
        is_causal, mask = ctx.is_causal, ctx.mask

        # implement the backward pass using the standard way, flash-attention, or the two-pass method.

        # return torch.compile(standard_attention_backward)(L, Q, K, V, O, dO, is_causal, mask)
        # return flash_attention_backward_two_passes(L, Q, K, V, O, dO, Bq, Bk, is_causal, mask)
        return flash_attention_backward(L, Q, K, V, O, dO, Bq, Bk, is_causal, mask)

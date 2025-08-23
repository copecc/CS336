import math
import torch
import triton

from torch import Tensor
from jaxtyping import Float, Bool

from .forward import flash_attention_forward_kernel
from .backward import flash_attention_backward_kernel


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " batch_size Nq d"],
        K: Float[Tensor, " batch_size Nk d"],
        V: Float[Tensor, " batch_size Nk d"],
        is_causal: Bool = False,
    ):
        Bq, Bk = 16, 16
        batch_size, Nq, d = Q.shape
        Nk = K.shape[-2]

        O: Float[Tensor, " batch_size Nq d"] = torch.empty_like(Q)
        L: Float[Tensor, " batch_size Nq"] = torch.empty(Q.shape[:-1], device=Q.device)

        mask: Float[Tensor, " Nq Nk"] = (
            torch.triu(torch.full((Nq, Nk), -1e6, device=Q.device), diagonal=1) if is_causal else None
        )

        # Launch kernel
        flash_attention_forward_kernel[(triton.cdiv(Nq, Bq), batch_size)](
            # fmt:off
            Q, K, V, O, L, mask,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            mask.stride(0) if is_causal else 0, mask.stride(1) if is_causal else 0,
            Nq, Nk, 1 / math.sqrt(d),
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.Q_TILE_SIZE = Bq
        ctx.K_TILE_SIZE = Bk
        ctx.is_causal = is_causal
        ctx.mask = mask

        return O

    # fmt:on

    @staticmethod
    def backward(ctx, dO: Float[Tensor, " batch_size Nq d"]):
        L, Q, K, V, O = ctx.saved_tensors
        Bq, Bk = ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE
        is_causal, mask = ctx.is_causal, ctx.mask

        batch_size, Nk, d = K.shape
        Nq = Q.shape[-2]

        # Important initialization since dQ will accumulate gradients
        dQ: Float[Tensor, " batch_size Nq d"] = torch.zeros_like(Q)
        # No initialization here, so remember to zero out gradients in the main loop
        dK: Float[Tensor, " batch_size Nk d"] = torch.empty_like(K)
        dV: Float[Tensor, " batch_size Nk d"] = torch.empty_like(V)

        # Compute D = rowsum(O * dO), O is never used later
        D: Float[Tensor, " batch_size Nq"] = torch.sum(O * dO, dim=-1)

        # Launch kernel
        flash_attention_backward_kernel[(triton.cdiv(Nk, Bk), batch_size)](
            # fmt:off
            Q, K, V, D, L, dO, dQ, dK, dV, mask,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            mask.stride(0) if is_causal else 0, mask.stride(1) if is_causal else 0,
            Nq, Nk, 1 / math.sqrt(d),
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
            IS_CAUSAL=is_causal,
        )

        return dQ, dK, dV, None


# fmt:on

import math
import torch
import torch.nn.functional as F

from torch import Tensor
from einops import einsum
from jaxtyping import Float, Bool, Int


def flash_attention_forward_inner(
    Q_i: Float[Tensor, " ... Bq d"],
    K: Float[Tensor, " ... Nk d"],
    V: Float[Tensor, " ... Nk d"],
    Bk: int,
    is_causal: Bool = False,
    mask_i: Float[Tensor, " Bq Nk"] = None,
):
    Nk, d = K.shape[-2], K.shape[-1]
    Tk = math.ceil(Nk / Bk)
    # Initialize O_i^(0) = 0 ∈ R^{B_q × d}, l_i = 0 ∈ R^{B_q}, m_i = -∞ ∈ R^{B_q}
    O_i: Float[Tensor, " ... Bq d"] = torch.zeros_like(Q_i)
    l_i: Float[Tensor, " ... Bq"] = torch.zeros(Q_i.shape[:-1], device=Q_i.device)
    m_ij: Float[Tensor, " ... Bq"] = torch.full(Q_i.shape[:-1], float("-inf"), device=Q_i.device)

    for j in range(Tk):
        start, end = j * Bk, min((j + 1) * Bk, Nk)
        need_padding = Bk - (end - start)
        # Load K_j and V_j
        K_j: Float[Tensor, " ... Bk d"] = K[..., start:end, :]
        V_j: Float[Tensor, " ... Bk d"] = V[..., start:end, :]
        mask_j: Float[Tensor, " Bq Bk"] = mask_i[..., start:end] if is_causal else None
        if need_padding > 0:  # Ensure K_j and V_j have shape (..., B_k, d)
            K_j = F.pad(K_j, (0, 0, 0, need_padding), value=0)
            V_j = F.pad(V_j, (0, 0, 0, need_padding), value=0)
            mask_j = F.pad(mask_j, (0, need_padding), value=0) if is_causal else None
        # Compute tile of pre-softmax attention scores: S_i^{(j)} = Q_i (K_j)^T / sqrt(d) ∈ R^{Bq × Bk}
        S_i = einsum(Q_i, K_j, "... Bq d, ... Bk d -> ... Bq Bk") / (math.sqrt(d))
        # Apply mask
        S_i = S_i + mask_j.to(S_i.dtype) if is_causal else S_i
        # m_i^{(j)} = max(m_i^{(j-1)}, rowmax(S_i^{(j)})) ∈ R^{Bq}
        m_i = torch.max(m_ij, S_i.amax(dim=-1))
        # P_i^{(j)} = exp(S_i^{(j)} - m_i^{(j)}) ∈ R^{Bq × Bk}
        P_i: Float[Tensor, " ... Bq Bk"] = torch.exp(S_i - m_i.unsqueeze(-1))
        # l_i^{(j)} = exp(m_i^{(j-1)} - m_i^{(j)}) * l_i^{(j-1)} + rowsum(P_i^{(j)}) ∈ R^{Bq}
        l_i = torch.exp(m_ij - m_i) * l_i + P_i.sum(dim=-1)
        # O_i^{(j)} = diag(exp(m_i^{(j-1)} - m_i^{(j)})) O_i^{(j-1)} + P_i^{(j)} V_j
        scale: Float[Tensor, " ... Bq 1"] = torch.exp(m_ij - m_i).unsqueeze(-1)
        O_i = scale * O_i + einsum(P_i, V_j, "... Bq Bk, ... Bk d -> ... Bq d")

        m_ij = m_i
    # O_i = diag(l_i^{(T_k)})^{-1} O_i^{(T_k)}
    O_i = O_i / l_i.unsqueeze(-1)
    # L_i = m_i^{(T_k)} + log(l_i^{(T_k)})
    L_i = m_ij + torch.log(l_i)

    return O_i, L_i


def flash_attention_forward(
    Q: Float[Tensor, " ... Nq d"],
    K: Float[Tensor, " ... Nk d"],
    V: Float[Tensor, " ... Nk d"],
    Bq: Int = 16,
    Bk: Int = 16,
    is_causal: Bool = False,
    mask: Float[Tensor, " Nq Nk"] = None,
):
    Nq = Q.shape[-2]
    Tq = math.ceil(Nq / Bq)

    O: Float[Tensor, " ... Nq d"] = torch.zeros_like(Q)
    L: Float[Tensor, " ... Nq"] = torch.empty(Q.shape[:-1], device=Q.device)

    for i in range(Tq):
        start, end = i * Bq, min((i + 1) * Bq, Nq)
        need_padding = Bq - (end - start)
        # Load Q_i
        Q_i: Float[Tensor, " ... Bq d"] = Q[..., start:end, :]
        mask_i: Float[Tensor, " Bq Nk"] = mask[start:end] if is_causal else None
        if need_padding > 0:
            # Ensure Q_i has shape (..., Bq, d), mask_i has shape (Bq, Nk)
            Q_i = F.pad(Q_i, (0, 0, 0, need_padding), value=0)
            mask_i = F.pad(mask_i, (0, 0, 0, need_padding), value=0) if is_causal else None
        # Compute inner flash attention
        O_i, L_i = flash_attention_forward_inner(Q_i, K, V, Bk, is_causal, mask_i)

        O[..., start:end, :] = O_i[..., : end - start, :]
        L[..., start:end] = L_i[..., : end - start]

    return O, L

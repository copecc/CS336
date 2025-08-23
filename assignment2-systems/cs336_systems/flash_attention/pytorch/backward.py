import math
import torch
import torch.nn.functional as F

from torch import Tensor
from einops import einsum
from jaxtyping import Float, Bool, Int


def standard_attention_backward(
    L: Float[Tensor, " ... Nq"],
    Q: Float[Tensor, " ... Nq d"],
    K: Float[Tensor, " ... Nk d"],
    V: Float[Tensor, " ... Nk d"],
    O: Float[Tensor, " ... Nq d"],
    dO: Float[Tensor, " ... Nq d"],
    is_causal: Bool = False,
    mask: Float[Tensor, " Nq Nk"] = None,
):
    d = Q.shape[-1]
    # D = rowsum(O * dO), equal to rowsum(P * dP) since
    # P dP^T = P (dO V^T)^T = (P V) dO^T = O dO^T, [rowsum(A * B) = diag(AB^T)]
    # so D = rowsum(P * dP) = rowsum(O * dO)
    D: Float[Tensor, " ... Nq"] = torch.sum(O * dO, dim=-1)
    # recompute attention score: S = Q K^T / sqrt(d)
    S = einsum(Q, K, "... Nq d, ... Nk d -> ... Nq Nk") / (math.sqrt(d))
    # Apply mask, consistent with flash_attention_forward
    S = S + mask.to(S.dtype) if is_causal else S
    # P_ij = exp(S_ij - L_i)
    P: Float[Tensor, " ... Nq Nk"] = torch.exp(S - L.unsqueeze(-1))
    # dV = P^T dO
    dV = einsum(P, dO, "... Nq Nk, ... Nq d -> ... Nk d")
    # dP = dO V^T
    dP = einsum(dO, V, "... Nq d, ... Nk d -> ... Nq Nk")
    # dS_ij = P_ij ◦ (dP_ij - D_i)
    dS: Float[Tensor, " ... Nq Nk"] = P * (dP - D.unsqueeze(-1))
    # dQ = dS K / sqrt(d)
    dQ = einsum(dS, K, "... Nq Nk, ... Nk d -> ... Nq d") / (math.sqrt(d))
    # dK = dS^T Q / sqrt(d)
    dK = einsum(dS, Q, "... Nq Nk, ... Nq d -> ... Nk d") / (math.sqrt(d))
    return dQ, dK, dV, None  # None for is_causal since we don't need it in backward


def flash_attention_backward_inner(
    D: Float[Tensor, " ... Nq"],
    L: Float[Tensor, " ... Nq"],
    Q: Float[Tensor, " ... Nq d"],
    K_j: Float[Tensor, " ... Bk d"],
    V_j: Float[Tensor, " ... Bk d"],
    dO: Float[Tensor, " ... Nq d"],
    dQ: Float[Tensor, " ... Nq d"],
    dK_j: Float[Tensor, " ... Bk d"],
    dV_j: Float[Tensor, " ... Bk d"],
    Bq: Int,
    is_causal: Bool = False,
    mask_j: Float[Tensor, " Nq Bk"] = None,
):
    Nq, d = Q.shape[-2], Q.shape[-1]
    Tq = math.ceil(Nq / Bq)

    for i in range(Tq):
        start, end = i * Bq, min((i + 1) * Bq, Nq)
        need_padding = Bq - (end - start)
        # Load Qi, Oi, dOi, dQi from global memory
        Q_i: Float[Tensor, " ... Bq d"] = Q[..., start:end, :]
        dO_i: Float[Tensor, " ... Bq d"] = dO[..., start:end, :]
        dQ_i: Float[Tensor, " ... Bq d"] = dQ[..., start:end, :]
        L_i: Float[Tensor, " ... Bq"] = L[..., start:end]
        D_i: Float[Tensor, " ... Bq"] = D[..., start:end]
        mask_i: Float[Tensor, " ... Bq Bk"] = mask_j[start:end] if is_causal else None
        if need_padding > 0:
            Q_i = F.pad(Q_i, (0, 0, 0, need_padding), value=0)
            dO_i = F.pad(dO_i, (0, 0, 0, need_padding), value=0)
            dQ_i = F.pad(dQ_i, (0, 0, 0, need_padding), value=0)
            L_i = F.pad(L_i, (0, need_padding), value=0)
            D_i = F.pad(D_i, (0, need_padding), value=0)
            mask_i = F.pad(mask_i, (0, 0, 0, need_padding), value=0) if mask_i is not None else None
        # Compute tile of pre-softmax attention scores: S_i^{(j)} = Q_i (K_j)^T / sqrt(d) ∈ R^{Bq × Bk}
        S_i = einsum(Q_i, K_j, "... Bq d, ... Bk d -> ... Bq Bk") / (math.sqrt(d))
        # Apply mask
        S_i = S_i + mask_i.to(S_i.dtype) if is_causal else S_i
        # Compute attention probabilities: P_i^{(j)} = exp(S_i^{(j)} - L_i) ∈ R^{Bq × Bk}
        P_i = torch.exp(S_i - L_i.unsqueeze(-1))
        # Compute dV_j: dV_j += (P_i^{(j)})^T dO_i ∈ R^{Bk × d}
        dV_j += einsum(P_i, dO_i, "... Bq Bk, ... Bq d -> ... Bk d")
        # Compute dP_i^{(j)}: dP_i^{(j)} = dO_i (V_j)^T ∈ R^{Bq × Bk}
        dP_i = einsum(dO_i, V_j, "... Bq d, ... Bk d -> ... Bq Bk")
        # Compute dS_i^{(j)}: dS_i^{(j)} = P_i^{(j)} * (dP_i^{(j)} - D_i) / sqrt(d) ∈ R^{Bq × Bk}
        dS_i = P_i * (dP_i - D_i.unsqueeze(-1)) / (math.sqrt(d))
        # Load dQ_i from global memory, then update: dQ_i += dS_i^{(j)} K_j ∈ R^{Bq × d}
        # !!! Must be atomic for correctness!
        dQ_i += einsum(dS_i, K_j, "... Bq Bk, ... Bk d -> ... Bq d")
        # Compute dK_j: dK_j += (dS_i^{(j)})^T Q_i ∈ R^{Bk × d}
        dK_j += einsum(dS_i, Q_i, "... Bq Bk, ... Bq d -> ... Bk d")

        dQ[..., start:end, :] = dQ_i[..., : end - start, :]

    return dK_j, dV_j


def flash_attention_backward(
    L: Float[Tensor, " ... Nq"],
    Q: Float[Tensor, " ... Nq d"],
    K: Float[Tensor, " ... Nk d"],
    V: Float[Tensor, " ... Nk d"],
    O: Float[Tensor, " ... Nq d"],
    dO: Float[Tensor, " ... Nq d"],
    Bq: Int,
    Bk: Int,
    is_causal: Bool = False,
    mask: Float[Tensor, " Nq Nk"] = None,
):
    Nk = K.shape[-2]
    Tk = math.ceil(Nk / Bk)
    # Important initialization since dQ will accumulate gradients
    dQ: Float[Tensor, " ... Nq d"] = torch.zeros_like(Q)
    # No initialization here, so remember to zero out gradients in the main loop
    dK: Float[Tensor, " ... Nk d"] = torch.empty_like(K)
    dV: Float[Tensor, " ... Nk d"] = torch.empty_like(V)

    # Compute D = rowsum(O * dO), O is never used later
    D: Float[Tensor, " ... Nq"] = torch.sum(O * dO, dim=-1)

    for j in range(Tk):
        start, end = j * Bk, min((j + 1) * Bk, Nk)
        need_padding = Bk - (end - start)
        # Load K_j, V_j
        K_j: Float[Tensor, " ... Bk d"] = K[..., start:end, :]
        V_j: Float[Tensor, " ... Bk d"] = V[..., start:end, :]
        dK_j: Float[Tensor, " ... Bk d"] = dK[..., start:end, :]
        dV_j: Float[Tensor, " ... Bk d"] = dV[..., start:end, :]
        # Initialize dK(j) = dV(j) = 0 ∈ R^{Bk×d}
        dK_j = torch.zeros_like(dK_j)
        dV_j = torch.zeros_like(dV_j)
        mask_j: Float[Tensor, " Nq Bk"] = mask[..., start:end] if is_causal else None
        if need_padding > 0:
            K_j = F.pad(K_j, (0, 0, 0, need_padding), value=0)
            V_j = F.pad(V_j, (0, 0, 0, need_padding), value=0)
            dK_j = F.pad(dK_j, (0, 0, 0, need_padding), value=0)
            dV_j = F.pad(dV_j, (0, 0, 0, need_padding), value=0)
            mask_j = F.pad(mask_j, (0, need_padding), value=0) if mask_j is not None else None
        # Compute inner backward flash attention
        dK_j, dV_j = flash_attention_backward_inner(D, L, Q, K_j, V_j, dO, dQ, dK_j, dV_j, Bq, is_causal, mask_j)
        dK[..., start:end, :] = dK_j[..., : end - start, :]
        dV[..., start:end, :] = dV_j[..., : end - start, :]

    return dQ, dK, dV, None  # None for is_causal since we don't need it in backward


def flash_attention_backward_inner_dKV(
    D: Float[Tensor, " ... Nq"],
    L: Float[Tensor, " ... Nq"],
    Q: Float[Tensor, " ... Nq d"],
    K_j: Float[Tensor, " ... Bk d"],
    V_j: Float[Tensor, " ... Bk d"],
    dO: Float[Tensor, " ... Nq d"],
    dK_j: Float[Tensor, " ... Bk d"],
    dV_j: Float[Tensor, " ... Bk d"],
    Bq: Int,
    is_causal: Bool = False,
    mask_j: Float[Tensor, " Nq Bk"] = None,
):
    Nq, d = Q.shape[-2], Q.shape[-1]
    Tq = math.ceil(Nq / Bq)

    for i in range(Tq):
        start, end = i * Bq, min((i + 1) * Bq, Nq)
        need_padding = Bq - (end - start)
        # Load Qi, Oi, dOi, dQi from global memory
        Q_i: Float[Tensor, " ... Bq d"] = Q[..., start:end, :]
        dO_i: Float[Tensor, " ... Bq d"] = dO[..., start:end, :]
        L_i: Float[Tensor, " ... Bq"] = L[..., start:end]
        D_i: Float[Tensor, " ... Bq"] = D[..., start:end]
        mask_i: Float[Tensor, " ... Bq Bk"] = mask_j[start:end] if is_causal else None
        if need_padding > 0:
            Q_i = F.pad(Q_i, (0, 0, 0, need_padding), value=0)
            dO_i = F.pad(dO_i, (0, 0, 0, need_padding), value=0)
            L_i = F.pad(L_i, (0, need_padding), value=0)
            D_i = F.pad(D_i, (0, need_padding), value=0)
            mask_i = F.pad(mask_i, (0, 0, 0, need_padding), value=0) if mask_i is not None else None
        # Compute tile of pre-softmax attention scores: S_i^{(j)} = Q_i (K_j)^T / sqrt(d) ∈ R^{Bq × Bk}
        S_i = einsum(Q_i, K_j, "... Bq d, ... Bk d -> ... Bq Bk") / (math.sqrt(d))
        # Apply mask
        S_i = S_i + mask_i.to(S_i.dtype) if is_causal else S_i
        # Compute attention probabilities: P_i^{(j)} = exp(S_i^{(j)} - L_i) ∈ R^{Bq × Bk}
        P_i = torch.exp(S_i - L_i.unsqueeze(-1))
        # Compute dV_j: dV_j += (P_i^{(j)})^T dO_i ∈ R^{Bk × d}
        dV_j += einsum(P_i, dO_i, "... Bq Bk, ... Bq d -> ... Bk d")
        # Compute dP_i^{(j)}: dP_i^{(j)} = dO_i (V_j)^T ∈ R^{Bq × Bk}
        dP_i = einsum(dO_i, V_j, "... Bq d, ... Bk d -> ... Bq Bk")
        # Compute dS_i^{(j)}: dS_i^{(j)} = P_i^{(j)} * (dP_i^{(j)} - D_i) / sqrt(d) ∈ R^{Bq × Bk}
        dS_i = P_i * (dP_i - D_i.unsqueeze(-1)) / (math.sqrt(d))
        # Compute dK_j: dK_j += (dS_i^{(j)})^T Q_i ∈ R^{Bk × d}
        dK_j += einsum(dS_i, Q_i, "... Bq Bk, ... Bq d -> ... Bk d")

    return dK_j, dV_j


def flash_attention_backward_inner_dQ(
    D: Float[Tensor, " ... Nq"],
    L: Float[Tensor, " ... Nq"],
    Q: Float[Tensor, " ... Nq d"],
    K_j: Float[Tensor, " ... Bk d"],
    V_j: Float[Tensor, " ... Bk d"],
    dO: Float[Tensor, " ... Nq d"],
    dQ: Float[Tensor, " ... Nq d"],
    Bq: Int,
    is_causal: Bool = False,
    mask_j: Float[Tensor, " Nq Bk"] = None,
):
    Nq, d = Q.shape[-2], Q.shape[-1]
    Tq = math.ceil(Nq / Bq)

    for i in range(Tq):
        start, end = i * Bq, min((i + 1) * Bq, Nq)
        need_padding = Bq - (end - start)
        # Load Qi, Oi, dOi, dQi from global memory
        Q_i: Float[Tensor, " ... Bq d"] = Q[..., start:end, :]
        dO_i: Float[Tensor, " ... Bq d"] = dO[..., start:end, :]
        dQ_i: Float[Tensor, " ... Bq d"] = dQ[..., start:end, :]
        L_i: Float[Tensor, " ... Bq"] = L[..., start:end]
        D_i: Float[Tensor, " ... Bq"] = D[..., start:end]
        mask_i: Float[Tensor, " ... Bq Bk"] = mask_j[start:end] if is_causal else None
        if need_padding > 0:
            Q_i = F.pad(Q_i, (0, 0, 0, need_padding), value=0)
            dO_i = F.pad(dO_i, (0, 0, 0, need_padding), value=0)
            dQ_i = F.pad(dQ_i, (0, 0, 0, need_padding), value=0)
            L_i = F.pad(L_i, (0, need_padding), value=0)
            D_i = F.pad(D_i, (0, need_padding), value=0)
            mask_i = F.pad(mask_i, (0, 0, 0, need_padding), value=0) if mask_i is not None else None
        # Compute tile of pre-softmax attention scores: S_i^{(j)} = Q_i (K_j)^T / sqrt(d) ∈ R^{Bq × Bk}
        S_i = einsum(Q_i, K_j, "... Bq d, ... Bk d -> ... Bq Bk") / (math.sqrt(d))
        # Apply mask
        S_i = S_i + mask_i.to(S_i.dtype) if is_causal else S_i
        # Compute attention probabilities: P_i^{(j)} = exp(S_i^{(j)} - L_i) ∈ R^{Bq × Bk}
        P_i = torch.exp(S_i - L_i.unsqueeze(-1))
        # Compute dP_i^{(j)}: dP_i^{(j)} = dO_i (V_j)^T ∈ R^{Bq × Bk}
        dP_i = einsum(dO_i, V_j, "... Bq d, ... Bk d -> ... Bq Bk")
        # Compute dS_i^{(j)}: dS_i^{(j)} = P_i^{(j)} * (dP_i^{(j)} - D_i) / sqrt(d) ∈ R^{Bq × Bk}
        dS_i = P_i * (dP_i - D_i.unsqueeze(-1)) / (math.sqrt(d))
        # Load dQ_i from global memory, then update: dQ_i += dS_i^{(j)} K_j ∈ R^{Bq × d}
        # !!! Must be atomic for correctness!
        dQ_i += einsum(dS_i, K_j, "... Bq Bk, ... Bk d -> ... Bq d")

        dQ[..., start:end, :] = dQ_i[..., : end - start, :]


def flash_attention_backward_two_passes(
    L: Float[Tensor, " ... Nq"],
    Q: Float[Tensor, " ... Nq d"],
    K: Float[Tensor, " ... Nk d"],
    V: Float[Tensor, " ... Nk d"],
    O: Float[Tensor, " ... Nq d"],
    dO: Float[Tensor, " ... Nq d"],
    Bq: Int,
    Bk: Int,
    is_causal: Bool = False,
    mask: Float[Tensor, " Nq Nk"] = None,
):
    Nk = K.shape[-2]
    Tk = math.ceil(Nk / Bk)
    # Important initialization since dQ will accumulate gradients
    dQ: Float[Tensor, " ... Nq d"] = torch.zeros_like(Q)
    # No initialization here, so remember to zero out gradients in the main loop
    dK: Float[Tensor, " ... Nk d"] = torch.empty_like(K)
    dV: Float[Tensor, " ... Nk d"] = torch.empty_like(V)

    # Compute D = rowsum(O * dO), O is never used later
    D: Float[Tensor, " ... Nq"] = torch.sum(O * dO, dim=-1)

    for j in range(Tk):
        start, end = j * Bk, min((j + 1) * Bk, Nk)
        need_padding = Bk - (end - start)
        # Load K_j, V_j
        K_j: Float[Tensor, " ... Bk d"] = K[..., start:end, :]
        V_j: Float[Tensor, " ... Bk d"] = V[..., start:end, :]
        dK_j: Float[Tensor, " ... Bk d"] = dK[..., start:end, :]
        dV_j: Float[Tensor, " ... Bk d"] = dV[..., start:end, :]
        # Initialize dK(j) = dV(j) = 0 ∈ R^{Bk×d}
        dK_j = torch.zeros_like(dK_j)
        dV_j = torch.zeros_like(dV_j)
        mask_j: Float[Tensor, " Nq Bk"] = mask[..., start:end] if is_causal else None
        if need_padding > 0:
            K_j = F.pad(K_j, (0, 0, 0, need_padding), value=0)
            V_j = F.pad(V_j, (0, 0, 0, need_padding), value=0)
            dK_j = F.pad(dK_j, (0, 0, 0, need_padding), value=0)
            dV_j = F.pad(dV_j, (0, 0, 0, need_padding), value=0)
            mask_j = F.pad(mask_j, (0, need_padding), value=0) if mask_j is not None else None

        # Compute inner backward flash attention
        dK_j, dV_j = flash_attention_backward_inner_dKV(D, L, Q, K_j, V_j, dO, dK_j, dV_j, Bq, is_causal, mask_j)
        dK[..., start:end, :] = dK_j[..., : end - start, :]
        dV[..., start:end, :] = dV_j[..., : end - start, :]

        flash_attention_backward_inner_dQ(D, L, Q, K_j, V_j, dO, dQ, Bq, is_causal, mask_j)

    return dQ, dK, dV, None  # None for is_causal since we don't need it in backward

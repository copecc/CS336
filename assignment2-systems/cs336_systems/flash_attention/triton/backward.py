import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torch import Tensor
from jaxtyping import Float, Bool, Int

# fmt: off
# I dont want to mess up the format below...


@triton.jit
def flash_attention_backward_kernel(
    Q_ptr, K_ptr, V_ptr, D_ptr, L_ptr, dO_ptr, dQ_ptr, dK_ptr, dV_ptr, Mask_ptr,  # pointers
    stride_qb, stride_qq, stride_qd,        # strides over Q
    stride_kb, stride_kk, stride_kd,        # strides over K
    stride_vb, stride_vk, stride_vd,        # strides over V
    stride_db, stride_dq,                   # strides over D
    stride_lb, stride_lq,                   # strides over L
    stride_dob, stride_doq, stride_dod,     # strides over dO
    stride_dqb, stride_dqq, stride_dqd,     # strides over dQ
    stride_dkb, stride_dkk, stride_dkd,     # strides over dK
    stride_dvb, stride_dvk, stride_dvd,     # strides over dV
    stride_maskq, stride_maskk,             # strides over mask
    N_QUERIES, N_KEYS, scale,               # shape(Nq, Nk) and scaling(usually 1/sqrt(D))
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr, # d, Bq, Bk
    IS_CAUSAL: tl.constexpr,            # is_causal flag
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,              # Q[batch_index]
        shape=(N_QUERIES, D),                         # Q[batch_index].shape == (N_QUERIES[Nq], D)
        strides=(stride_qq, stride_qd),               # Q[batch_index] layout
        offsets=(0, 0),                               # Q[batch_index][query_tile_index]
        block_shape=(Q_TILE_SIZE, D),                 # Qi.shape == (Q_TILE_SIZE[Bq], D)
        order=(1, 0),                                 # traverse over D first
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES,D),
        strides=(stride_doq, stride_dod),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE,D),
        order=(1,0)
    )

    if IS_CAUSAL:
       Mask_block_ptr = tl.make_block_ptr(
          Mask_ptr,
          shape=(N_QUERIES, N_KEYS),
          strides=(stride_maskq, stride_maskk),
          offsets=(0, key_tile_index * K_TILE_SIZE),
          block_shape=(Q_TILE_SIZE, K_TILE_SIZE),
          order=(1, 0),
      )
    
    # Load K(j), V(j) from global memory
    K_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

    # Initialize dK(j) = dV(j) = 0 ∈ R^{Bk×d}
    dK_j = tl.zeros_like(K_j)
    dV_j = tl.zeros_like(V_j)

    Tq = tl.cdiv(N_QUERIES, Q_TILE_SIZE) # Number of query tiles

    # Loop over the query tiles
    for query_tile_index in range(Tq):
        # Load Qi, Di, dOi, dQi from global memory
        Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")
        
        if IS_CAUSAL:
            mask_i = tl.load(Mask_block_ptr, boundary_check=(0,), padding_option="zero")

        # Compute tile of pre-softmax attention scores: S_i^{(j)} = Q_i (K_j)^T / sqrt(d) ∈ R^{Bq × Bk}
        S_i = tl.dot(Q_i, tl.trans(K_j), allow_tf32=False) * scale
        # Apply mask
        if IS_CAUSAL:
            S_i = S_i + mask_i.to(S_i.dtype)
        # Compute attention probabilities: P_i^{(j)} = exp(S_i^{(j)} - L_i) ∈ R^{Bq × Bk}
        P_i = tl.exp(S_i - L_i[:, None])
        # Compute dV_j: dV_j += (P_i^{(j)})^T dO_i ∈ R^{Bk × d}
        dV_j += tl.dot(tl.trans(P_i.to(dO_i.dtype)), dO_i, allow_tf32=False)
        # Compute dP_i^{(j)}: dP_i^{(j)} = dO_i (V_j)^T ∈ R^{Bq × Bk}
        dP_i = tl.dot(dO_i, tl.trans(V_j.to(dO_i.dtype)), allow_tf32=False)
        # Compute dS_i^{(j)}: dS_i^{(j)} = P_i^{(j)} * (dP_i^{(j)} - D_i) / sqrt(d) ∈ R^{Bq × Bk}
        dS_i = P_i * (dP_i - D_i[:, None]) * scale
        # Load dQ_i from global memory, then update: dQ_i += dS_i^{(j)} K_j ∈ R^{Bq × d}
        # !!! Must be atomic for correctness!
        dQ_i_ptr = dQ_ptr + batch_index * stride_dqb + query_tile_index * Q_TILE_SIZE * stride_dqq
        offset_row, offset_col = tl.arange(0, Q_TILE_SIZE), tl.arange(0, D)
        offset_linear = offset_row[:, None] * D + offset_col[None, :] # generate indexes for dQ_i
        # boundary check
        mask_dQ_i = query_tile_index * Q_TILE_SIZE + offset_row[:, None] < N_QUERIES
        # atomic_add require pointer (Block of dtype=triton.PointerDType) , val (Block of dtype=pointer.dtype.element_ty)
        tl.atomic_add(dQ_i_ptr + offset_linear, tl.dot(dS_i, K_j, allow_tf32=False), mask=mask_dQ_i)
        # Compute dK_j: dK_j += (dS_i^{(j)})^T Q_i ∈ R^{Bk × d}
        dK_j += tl.dot(tl.trans(dS_i), Q_i, allow_tf32=False)

        # Advance pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        if IS_CAUSAL:
            Mask_block_ptr = tl.advance(Mask_block_ptr, (Q_TILE_SIZE, 0))

    # output gradients
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS,D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE,D),
        order=(1,0)
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS,D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE,D),
        order=(1,0)
    )

    tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty), boundary_check=(0,1))

# fmt:on

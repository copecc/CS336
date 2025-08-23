import triton
import triton.language as tl


@triton.jit
def flash_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, Mask_ptr,  # pointers
    stride_qb, stride_qq, stride_qd,    # strides over Q
    stride_kb, stride_kk, stride_kd,    # strides over K
    stride_vb, stride_vk, stride_vd,    # strides over V
    stride_ob, stride_oq, stride_od,    # strides over O
    stride_lb, stride_lq,               # strides over L
    stride_maskq, stride_maskk,         # strides over Mask, if not causal, set to 0
    N_QUERIES, N_KEYS, scale,           # shape(Nq, Nk) and scaling(usually 1/sqrt(D))
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr, # d, Bq, Bk
    IS_CAUSAL: tl.constexpr,            # is_causal flag
): # fmt: skip
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,  # Q[batch_index]
        shape=(N_QUERIES, D),  # Q[batch_index].shape == (N_QUERIES[Nq], D)
        strides=(stride_qq, stride_qd),  # Q[batch_index] layout
        offsets=(query_tile_index * Q_TILE_SIZE, 0),  # Q[batch_index][query_tile_index]
        block_shape=(Q_TILE_SIZE, D),  # Qi.shape == (Q_TILE_SIZE[Bq], D)
        order=(1, 0),  # traverse over D first
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    if IS_CAUSAL:
        Mask_block_ptr = tl.make_block_ptr(
            Mask_ptr,
            shape=(N_QUERIES, N_KEYS),
            strides=(stride_maskq, stride_maskk),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, K_TILE_SIZE),
            order=(1, 0),
        )

    # Load Qi from global memory
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    # Initialize the output and logsumexp
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_ij = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)  # number of key/value tiles

    # Loop over the key/value tiles
    for _ in range(Tk):
        # Load the K and V tiles
        K_j = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        if IS_CAUSAL:
            mask_j = tl.load(Mask_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Compute tile of pre-softmax attention scores: S_i^{(j)} = Q_i (K_j)^T / sqrt(d) ∈ R^{Bq × Bk}
        # setting allow_tf32=False to avoid crashing on old GPUs
        S_i = tl.dot(Q_i, tl.trans(K_j), allow_tf32=False) * scale  # scale = 1/sqrt(D)
        if IS_CAUSAL:
            S_i = S_i + mask_j.to(S_i.dtype)

        # This can be optimized further by using alpha and beta(only one `l`)
        # m_i^{(j)} = max(m_i^{(j-1)}, rowmax(S_i^{(j)})) ∈ R^{Bq}
        m_i = tl.maximum(m_ij, tl.max(S_i, axis=1))
        # P_i^{(j)} = exp(S_i^{(j)} - m_i^{(j)}) ∈ R^{Bq × Bk}
        P_i = tl.exp(S_i - m_i[:, None])
        # l_i^{(j)} = exp(m_i^{(j-1)} - m_i^{(j)}) * l_i^{(j-1)} + rowsum(P_i^{(j)}) ∈ R^{Bq}
        l_i = tl.exp(m_ij - m_i) * l_i + tl.sum(P_i, axis=1)
        # O_i^{(j)} = diag(exp(m_i^{(j-1)} - m_i^{(j)})) O_i^{(j-1)} + P_i^{(j)} V_j
        O_i = tl.exp(m_ij - m_i)[:, None] * O_i + tl.dot(P_i.to(V_j.dtype), V_j, allow_tf32=False)

        m_ij = m_i

        # Advance the K and V block pointers
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        if IS_CAUSAL:
            Mask_block_ptr = tl.advance(Mask_block_ptr, (0, K_TILE_SIZE))

    # O_i = diag(l_i^{(T_k)})^{-1} O_i^{(T_k)}
    O_i = O_i / l_i[:, None]
    # L_i = m_i^{(T_k)} + log(l_i^{(T_k)})
    L_i = m_ij + tl.log(l_i)

    # Write the output and logsumexp
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))

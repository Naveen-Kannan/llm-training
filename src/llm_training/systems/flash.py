# organize
# torch FA gets 3.5s, ALL the improvement should here
import math, torch
from einops import rearrange, einsum
import triton
import triton.language as tl


def _ensure_triton_allocator(device: torch.device) -> None:
    idx = device.index if device.index is not None else torch.cuda.current_device()

    dev = torch.device("cuda", idx)
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=dev))


TILE_SIZE1, TILE_SIZE2 = 128, 64
# math trick from online - use log2 and conv
LOG2E = tl.constexpr(1.4426950408889634)

# time size should be locked
# _fwd_cfgs = [
#     triton.Config({}, num_warps=4, num_stages=2),
#     triton.Config({}, num_warps=4, num_stages=3),
#     triton.Config({}, num_warps=8, num_stages=2),
#     triton.Config({}, num_warps=8, num_stages=3),
# ]
# _bkwd_cfg = [
#     triton.Config({}, num_warps=4, num_stages=1),
#     triton.Config({}, num_warps=4, num_stages=2),
#     triton.Config({}, num_warps=8, num_stages=1),
#     triton.Config({}, num_warps=8, num_stages=2),
# ]


# LEGACY #########################################################
@torch.compile
def flash_bwd(Q, K, V, dO, L, D, is_causal=False):
    d = Q.shape[-1]
    scale = d**-0.5

    # eq 13
    S = einsum(Q, K, "... n_q d_mod, ... n_k d_mod -> ... n_q n_k")
    S = S * scale

    if is_causal:
        q_idxs = torch.arange(Q.shape[1], device=Q.device)
        k_idxs = torch.arange(K.shape[1], device=Q.device)
        # follow fwd pass causal mask
        mask = k_idxs[None, :] > q_idxs[:, None]
        S = torch.where(mask, S + -1e6, S)

    # e1 14 with broadcast L
    P = torch.exp(S - L[:, :, None])

    # e15
    dV = einsum(P.to(dO.dtype), dO, "... n_q n_k, ... n_q d_mod -> ... n_k d_mod")

    # e16
    dP = einsum(dO, V, "... n_q d_mod, ... n_k d_mod -> ... n_q n_k")

    # e17 with broadcast D
    dS = P * (dP - D[:, :, None])

    # e18
    dQ = einsum(dS.to(K.dtype), K, "... n_q n_k, ... n_k d_mod -> ... n_q d_mod") * scale

    # e19
    dK = einsum(dS.to(Q.dtype), Q, "... n_q n_k, ... n_q d_mod -> ... n_k d_mod") * scale

    return dQ, dK, dV


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # assume shapes are right (we have no checks)
        B, n_q, d = Q.shape[0], Q.shape[1], Q.shape[2]
        B, n_k, d = K.shape[0], K.shape[1], K.shape[2]
        B, n_v, d = V.shape[0], V.shape[1], V.shape[2]
        # init output O,L
        O = torch.empty((B, n_q, d), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, n_q), device=Q.device, dtype=torch.float32)

        # lets say the tile is 16 x 16
        b_q, b_k = 16, 16
        T_q = math.ceil(n_q / b_q)
        # split q into tiles with einops
        Q_tiles = rearrange(Q, "B (t_q b_q) d -> B t_q b_q d", t_q=T_q, b_q=b_q)
        O_tiles = rearrange(O, "B (t_q b_q) d -> B t_q b_q d", t_q=T_q, b_q=b_q)

        T_k = math.ceil(n_k / b_k)
        K_tiles = rearrange(K, "B (t_k b_k) d -> B t_k b_k d", t_k=T_k, b_k=b_k)
        V_tiles = rearrange(V, "B (t_k b_k) d -> B t_k b_k d", t_k=T_k, b_k=b_k)

        L_tiles = rearrange(L, "B (t_q b_q) -> B t_q b_q", t_q=T_q, b_q=b_q)

        for i in range(T_q):
            # load Q_i from global memory
            Q_i = Q_tiles[:, i]
            # initialize O_i and l_i and m_i
            O_prev = torch.zeros((B, b_q, d), device=Q.device)
            l_prev = torch.zeros((B, b_q), device=Q.device)
            # -infty
            m_prev = torch.full((B, b_q), -float("inf"), device=Q.device)

            for j in range(T_k):
                # load K_j and V_j from global memory
                K_j = K_tiles[:, j]
                V_j = V_tiles[:, j]
                # tile of atten scores
                S_i_j = einsum(Q_i, K_j, "B b_q d, B b_k d -> B b_q b_k")
                S_i_j = S_i_j / math.sqrt(d)
                # update m_i
                tile_max = torch.max(S_i_j, dim=-1).values
                m_i_j = torch.max(m_prev, tile_max)
                # Compute P_i_j
                P_i_j = torch.exp(S_i_j - m_i_j[:, :, None])
                # Compute l_i_j
                l_i_j = torch.exp(m_prev - m_i_j) * l_prev + torch.sum(P_i_j, dim=-1)
                # Compute O_i_j
                online_decay = torch.exp(m_prev - m_i_j)
                O_i_j = online_decay[:, :, None] * O_prev + P_i_j @ V_j

                O_prev = O_i_j
                l_prev = l_i_j
                m_prev = m_i_j

            # Compute O_i
            O_i = O_prev / l_prev[:, :, None]
            # Compute L_i
            L_i = m_prev + torch.log(l_prev)

            O_tiles[:, i] = O_i
            L_tiles[:, i] = L_i

        O = rearrange(O_tiles, "B t_q b_q d -> B (t_q b_q) d", t_q=T_q, b_q=b_q)
        L = rearrange(L_tiles, "B t_q b_q -> B (t_q b_q)", t_q=T_q, b_q=b_q)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        # triv compiled pass
        D = torch.sum((O * grad_out).to(torch.float32), dim=-1)  # rowsum(O * dO) precompute D
        dQ, dK, dV = flash_bwd(Q, K, V, grad_out, L, D, ctx.is_causal)
        return dQ, dK, dV, None  # no for causal


# TRITON #########################################################


# @triton.autotune(configs=_fwd_cfgs, key=["N_QUERIES", "N_KEYS"]) how do ppl use this...//////
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,  # del unused strides
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    q_tile_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    # Offset each pointer with batch idx, mult with stride
    # Q_block_ptr = tl.make_block_ptr(
    #     Q_ptr + b_idx * stride_qb,
    #     shape=(N_QUERIES, D),
    #     strides=(stride_qq, stride_qd),
    #     offsets=(q_tile_idx * Q_TILE_SIZE, 0),
    #     block_shape=(Q_TILE_SIZE, D),
    #     order=(1, 0),
    # )

    # K_block_ptr = tl.make_block_ptr(
    #     K_ptr + b_idx * stride_kb,
    #     shape=(N_KEYS, D),
    #     strides=(stride_kk, stride_kd),
    #     offsets=(0, 0),
    #     block_shape=(K_TILE_SIZE, D),
    #     order=(1, 0),
    # )

    # # same block shape as K
    # V_block_ptr = tl.make_block_ptr(
    #     V_ptr + b_idx * stride_vb,
    #     shape=(N_KEYS, D),
    #     strides=(stride_vk, stride_vd),
    #     offsets=(0, 0),
    #     block_shape=(K_TILE_SIZE, D),
    #     order=(1, 0),
    # )

    # TMA
    Q_tma = tl.make_tensor_descriptor(
        Q_ptr + b_idx * stride_qb,
        shape=[N_QUERIES, D],
        strides=[stride_qq, 1],
        block_shape=[Q_TILE_SIZE, D],
    )

    K_tma = tl.make_tensor_descriptor(
        K_ptr + b_idx * stride_kb,
        shape=[N_KEYS, D],
        strides=[stride_kk, 1],
        block_shape=[K_TILE_SIZE, D],
    )

    # same block shape as K
    V_tma = tl.make_tensor_descriptor(
        V_ptr + b_idx * stride_vb,
        shape=[N_KEYS, D],
        strides=[stride_vk, 1],
        block_shape=[K_TILE_SIZE, D],
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + b_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # O_block_ptr = tl.make_block_ptr(
    #     O_ptr + b_idx * stride_ob,
    #     shape=(N_QUERIES, D),
    #     strides=(stride_oq, stride_od),
    #     offsets=(q_tile_idx * Q_TILE_SIZE, 0),
    #     block_shape=(Q_TILE_SIZE, D),
    #     order=(1, 0),
    # )

    # O needs tensor descriptor
    O_tma = tl.make_tensor_descriptor(
        O_ptr + b_idx * stride_ob,
        shape=[N_QUERIES, D],
        strides=[stride_oq, 1],
        block_shape=[Q_TILE_SIZE, D],
    )

    # we are starting in the for loop here
    # load Q_i once, block auto loads a tile
    # Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # TMA LOAD;
    Q_i = Q_tma.load([q_tile_idx * Q_TILE_SIZE, 0])

    # initialize O_prev, l_prev, m_prev
    O_prev = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_prev = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)  # was it this or -1e6, check

    # bound the loop, can remove causal loads
    k_tile_bound = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        # it now changes if Q =/= K. 4 hr gone
        k_tile_bound = tl.cdiv((q_tile_idx + 1) * Q_TILE_SIZE, K_TILE_SIZE)
    for j in range(k_tile_bound):
        # load K_j V_j from global memory
        # K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # TMA LOAD; doesnt need to pad; nice
        K_j = K_tma.load([j * K_TILE_SIZE, 0])
        V_j = V_tma.load([j * K_TILE_SIZE, 0])

        # scale is sqrt(1/d)
        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale

        if is_causal:
            # todo: unroll these, wasted each call
            # fade inside causal
            if (j + 1) * K_TILE_SIZE - 1 > q_tile_idx * Q_TILE_SIZE:
                q_idxs = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_idxs = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

                S_i_j = tl.where(k_idxs[None, :] > q_idxs[:, None], S_i_j + -1e6, S_i_j)
                # add -1e6 where triggered

        # update m_i
        tile_max = tl.max(S_i_j, axis=1)
        m_i_j = tl.maximum(m_prev, tile_max)

        # broadcast and exp
        P_i_j = tl.exp2((S_i_j - m_i_j[:, None]) * LOG2E)

        # l_i_j calc; ngl this is identical to torch
        online_decay = tl.exp2((m_prev - m_i_j) * LOG2E)
        l_i_j = online_decay * l_prev + tl.sum(P_i_j, axis=1)

        # O_i_j rescale prev and accum using acc
        O_i_j = online_decay[:, None] * O_prev
        O_i_j = tl.dot(P_i_j.to(V_j.dtype), V_j, acc=O_i_j)

        O_prev = O_i_j
        l_prev = l_i_j
        m_prev = m_i_j

        # move pointers easy
        # K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        # V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # final
    O_i = O_prev / l_prev[:, None]
    L_i = m_prev + tl.log(l_prev)

    # store output casting type following pdf
    # tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    # TMA store
    O_tma.store([q_tile_idx * Q_TILE_SIZE, 0], O_i.to(O_ptr.dtype.element_ty))
    tl.store(L_block_ptr, L_i)


# # @triton.autotune(configs=_bkwd_cfg, key=["N_QUERIES", "N_KEYS"])
# @triton.jit
# def flash_bkwd_kernel(
#     # holy import, we also need the dX buffers
#     Q_ptr, K_ptr, V_ptr,
#     dO_ptr,
#     L_ptr, D_ptr,
#     dK_ptr, dV_ptr,
#     stride_qb, stride_qq, stride_qd,
#     stride_kb, stride_kk, stride_kd,
#     stride_vb, stride_vk, stride_vd,
#     stride_dob, stride_doq, stride_dod,
#     stride_lb, stride_lq,
#     stride_db, stride_dq,
#     stride_dkb, stride_dkk, stride_dkd,
#     stride_dvb, stride_dvk, stride_dvd,
#     N_QUERIES, N_KEYS,
#     scale,
#     D_MODEL: tl.constexpr,
#     Q_TILE_SIZE: tl.constexpr,
#     K_TILE_SIZE: tl.constexpr,
#     is_causal: tl.constexpr, # is constexpr c++ rel?
# ):
#     # Program indices
#     k_tile_idx = tl.program_id(0)
#     b_idx = tl.program_id(1)

#     # Offset each pointer with batch idx, mult with stride

#     #K, V global outs
#     K_block_ptr = tl.make_block_ptr(
#         K_ptr + b_idx * stride_kb,
#         shape=(N_KEYS, D_MODEL),
#         strides=(stride_kk, stride_kd),
#         offsets=(k_tile_idx * K_TILE_SIZE, 0),
#         block_shape=(K_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     dK_block_ptr = tl.make_block_ptr(
#         dK_ptr + b_idx * stride_dkb,
#         shape=(N_KEYS, D_MODEL),
#         strides=(stride_dkk, stride_dkd),
#         offsets=(k_tile_idx * K_TILE_SIZE, 0),
#         block_shape=(K_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     V_block_ptr = tl.make_block_ptr(
#         V_ptr + b_idx * stride_vb,
#         shape=(N_KEYS, D_MODEL),
#         strides=(stride_vk, stride_vd),
#         offsets=(k_tile_idx * K_TILE_SIZE, 0),
#         block_shape=(K_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     dV_block_ptr = tl.make_block_ptr(
#         dV_ptr + b_idx * stride_dvb,
#         shape=(N_KEYS, D_MODEL),
#         strides=(stride_dvk, stride_dvd),
#         offsets=(k_tile_idx * K_TILE_SIZE, 0),
#         block_shape=(K_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     K_j = tl.load(K_block_ptr)
#     V_j = tl.load(V_block_ptr)

#     dK_j = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)
#     dV_j = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)

#     T_q = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
#     start_i = 0
#     if is_causal:
#         start_i = (k_tile_idx* K_TILE_SIZE) // Q_TILE_SIZE

#     # rest must be advanced within loop
#     Q_block_ptr = tl.make_block_ptr(
#         Q_ptr + b_idx * stride_qb,
#         shape=(N_QUERIES, D_MODEL),
#         strides=(stride_qq, stride_qd),
#         offsets=(start_i * Q_TILE_SIZE, 0),
#         block_shape=(Q_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     dO_block_ptr = tl.make_block_ptr(
#         dO_ptr + b_idx * stride_dob,
#         shape=(N_QUERIES, D_MODEL),
#         strides=(stride_doq, stride_dod),
#         offsets=(start_i * Q_TILE_SIZE, 0),
#         block_shape=(Q_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     L_block_ptr = tl.make_block_ptr(
#         L_ptr + b_idx * stride_lb,
#         shape=(N_QUERIES,),
#         strides=(stride_lq,),
#         offsets=(start_i * Q_TILE_SIZE,),
#         block_shape=(Q_TILE_SIZE,),
#         order=(0,),
#     )

#     D_block_ptr = tl.make_block_ptr(
#         D_ptr + b_idx * stride_db,
#         shape=(N_QUERIES,),
#         strides=(stride_dq,),
#         offsets=(start_i * Q_TILE_SIZE,),
#         block_shape=(Q_TILE_SIZE,),
#         order=(0,),
#     )

#     # q tiles stream through this k tile, only iterate active range
#     for i_off in range(T_q - start_i):
#         i = start_i + i_off
#         Q_i = tl.load(Q_block_ptr)
#         dO_i = tl.load(dO_block_ptr)
#         L_i = tl.load(L_block_ptr)
#         D_i = tl.load(D_block_ptr)

#         # todo: math

#         # 11 tile of attn
#         S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
#         if is_causal:
#             # same fade as fwd
#             if (k_tile_idx + 1) * K_TILE_SIZE - 1 > i * Q_TILE_SIZE:
#                 q_idxs = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
#                 k_idxs = k_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
#                 S_i_j = tl.where(k_idxs[None, :] > q_idxs[:, None], S_i_j + -1e6, S_i_j)

#         # 12 attn prob recomp
#         P_i_j = tl.exp2((S_i_j - L_i[:, None]) * LOG2E)
#         # 13 accum dV
#         dV_j += tl.dot(tl.trans(P_i_j.to(V_j.dtype)), dO_i)
#         # 14 accum dP
#         dP_i_j = tl.dot(dO_i, tl.trans(V_j))
#         # 15 accum dS
#         dS_i_j = P_i_j * (dP_i_j - D_i[:, None])
#         # 16 accum dK
#         dK_j += tl.dot(tl.trans(dS_i_j.to(K_j.dtype)), Q_i) * scale

#         Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
#         dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
#         L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
#         D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

#     tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty))
#     tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty))


# # @triton.autotune(configs=_bkwd_cfg, key=["N_QUERIES", "N_KEYS"])
# @triton.jit
# def flash_bkwd_dq_kernel(
#     Q_ptr, K_ptr, V_ptr,
#     dO_ptr, L_ptr, D_ptr,
#     dQ_ptr,
#     stride_qb, stride_qq, stride_qd,
#     stride_kb, stride_kk, stride_kd,
#     stride_vb, stride_vk, stride_vd,
#     stride_dob, stride_doq, stride_dod,
#     stride_lb, stride_lq,
#     stride_db, stride_dq,
#     stride_dqb, stride_dqq, stride_dqd,
#     N_QUERIES, N_KEYS,
#     scale,
#     D_MODEL: tl.constexpr,
#     Q_TILE_SIZE: tl.constexpr,
#     K_TILE_SIZE: tl.constexpr,
#     is_causal: tl.constexpr,
# ):
#     q_tile_idx = tl.program_id(0)
#     b_idx = tl.program_id(1)

#     # q side stuff
#     Q_block_ptr = tl.make_block_ptr(
#         Q_ptr + b_idx * stride_qb,
#         shape=(N_QUERIES, D_MODEL),
#         strides=(stride_qq, stride_qd),
#         offsets=(q_tile_idx * Q_TILE_SIZE, 0),
#         block_shape=(Q_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     dQ_block_ptr = tl.make_block_ptr(
#         dQ_ptr + b_idx * stride_dqb,
#         shape=(N_QUERIES, D_MODEL),
#         strides=(stride_dqq, stride_dqd),
#         offsets=(q_tile_idx * Q_TILE_SIZE, 0),
#         block_shape=(Q_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     dO_block_ptr = tl.make_block_ptr(
#         dO_ptr + b_idx * stride_dob,
#         shape=(N_QUERIES, D_MODEL),
#         strides=(stride_doq, stride_dod),
#         offsets=(q_tile_idx * Q_TILE_SIZE, 0),
#         block_shape=(Q_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     # kv side, advance easy
#     K_block_ptr = tl.make_block_ptr(
#         K_ptr + b_idx * stride_kb,
#         shape=(N_KEYS, D_MODEL),
#         strides=(stride_kk, stride_kd),
#         offsets=(0, 0),
#         block_shape=(K_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     V_block_ptr = tl.make_block_ptr(
#         V_ptr + b_idx * stride_vb,
#         shape=(N_KEYS, D_MODEL),
#         strides=(stride_vk, stride_vd),
#         offsets=(0, 0),
#         block_shape=(K_TILE_SIZE, D_MODEL),
#         order=(1, 0),
#     )

#     L_block_ptr = tl.make_block_ptr(
#         L_ptr + b_idx * stride_lb,
#         shape=(N_QUERIES,),
#         strides=(stride_lq,),
#         offsets=(q_tile_idx * Q_TILE_SIZE,),
#         block_shape=(Q_TILE_SIZE,),
#         order=(0,),
#     )

#     D_block_ptr = tl.make_block_ptr(
#         D_ptr + b_idx * stride_db,
#         shape=(N_QUERIES,),
#         strides=(stride_dq,),
#         offsets=(q_tile_idx * Q_TILE_SIZE,),
#         block_shape=(Q_TILE_SIZE,),
#         order=(0,),
#     )

#     Q_i = tl.load(Q_block_ptr)
#     dQ_i = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32)
#     dO_i = tl.load(dO_block_ptr)
#     L_i = tl.load(L_block_ptr)
#     D_i = tl.load(D_block_ptr)

#     # bound
#     k_tile_bound = tl.cdiv(N_KEYS, K_TILE_SIZE)
#     if is_causal:
#         k_tile_bound = tl.cdiv((q_tile_idx + 1) *Q_TILE_SIZE, K_TILE_SIZE)
#     for j in range(k_tile_bound):
#         K_j = tl.load(K_block_ptr)
#         V_j = tl.load(V_block_ptr)

#         # 25 tile of attn
#         S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
#         if is_causal:
#             # same fade as fwd
#             if (j + 1) * K_TILE_SIZE - 1 > q_tile_idx * Q_TILE_SIZE:
#                 q_idxs = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
#                 k_idxs = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
#                 S_i_j = tl.where(k_idxs[None, :] > q_idxs[:, None], S_i_j + -1e6, S_i_j)

#         # 26-30
#         P_i_j = tl.exp2((S_i_j - L_i[:, None]) * LOG2E)
#         dP_i_j = tl.dot(dO_i, tl.trans(V_j))
#         dS_i_j = P_i_j * (dP_i_j - D_i[:, None])
#         dQ_i += tl.dot(dS_i_j.to(K_j.dtype), K_j) * scale

#         K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
#         V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

#     tl.store(dQ_block_ptr, dQ_i.to(dQ_block_ptr.type.element_ty))


# @triton.autotune(configs=_bkwd_cfg, key=["N_QUERIES", "N_KEYS"])
@triton.jit
def flash_bkwd_kernel(
    # holy import, we also need the dX buffers
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    N_QUERIES,
    N_KEYS,
    scale,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,  # is constexpr c++ rel?
):
    k_tile_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    K_tma = tl.make_tensor_descriptor(
        K_ptr + b_idx * stride_kb,
        shape=[N_KEYS, D_MODEL],
        strides=[stride_kk, 1],
        block_shape=[K_TILE_SIZE, D_MODEL],
    )
    V_tma = tl.make_tensor_descriptor(
        V_ptr + b_idx * stride_vb,
        shape=[N_KEYS, D_MODEL],
        strides=[stride_vk, 1],
        block_shape=[K_TILE_SIZE, D_MODEL],
    )
    dK_tma = tl.make_tensor_descriptor(
        dK_ptr + b_idx * stride_dkb,
        shape=[N_KEYS, D_MODEL],
        strides=[stride_dkk, 1],
        block_shape=[K_TILE_SIZE, D_MODEL],
    )
    dV_tma = tl.make_tensor_descriptor(
        dV_ptr + b_idx * stride_dvb,
        shape=[N_KEYS, D_MODEL],
        strides=[stride_dvk, 1],
        block_shape=[K_TILE_SIZE, D_MODEL],
    )
    Q_tma = tl.make_tensor_descriptor(
        Q_ptr + b_idx * stride_qb,
        shape=[N_QUERIES, D_MODEL],
        strides=[stride_qq, 1],
        block_shape=[Q_TILE_SIZE, D_MODEL],
    )
    dO_tma = tl.make_tensor_descriptor(
        dO_ptr + b_idx * stride_dob,
        shape=[N_QUERIES, D_MODEL],
        strides=[stride_doq, 1],
        block_shape=[Q_TILE_SIZE, D_MODEL],
    )

    K_j = K_tma.load([k_tile_idx * K_TILE_SIZE, 0])
    V_j = V_tma.load([k_tile_idx * K_TILE_SIZE, 0])

    dK_j = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)

    T_q = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    start_i = 0
    if is_causal:
        start_i = (k_tile_idx * K_TILE_SIZE) // Q_TILE_SIZE

    L_block_ptr = tl.make_block_ptr(
        L_ptr + b_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(start_i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + b_idx * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(start_i * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    for i_off in range(T_q - start_i):
        i = start_i + i_off
        Q_i = Q_tma.load([i * Q_TILE_SIZE, 0])
        dO_i = dO_tma.load([i * Q_TILE_SIZE, 0])
        L_i = tl.load(L_block_ptr)
        D_i = tl.load(D_block_ptr)

        # todo: math

        # 11 tile of attn
        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            # same fade as fwd
            if (k_tile_idx + 1) * K_TILE_SIZE - 1 > i * Q_TILE_SIZE:
                q_idxs = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_idxs = k_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                S_i_j = tl.where(k_idxs[None, :] > q_idxs[:, None], S_i_j + -1e6, S_i_j)

        # 12 attn prob recomp
        P_i_j = tl.exp2((S_i_j - L_i[:, None]) * LOG2E)
        # 13 accum dV
        dV_j += tl.dot(tl.trans(P_i_j.to(V_j.dtype)), dO_i)
        # 14 accum dP
        dP_i_j = tl.dot(dO_i, tl.trans(V_j))
        # 15 accum dS
        dS_i_j = P_i_j * (dP_i_j - D_i[:, None])
        # 16 accum dK
        dK_j += tl.dot(tl.trans(dS_i_j.to(K_j.dtype)), Q_i) * scale

        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    dK_tma.store([k_tile_idx * K_TILE_SIZE, 0], dK_j.to(dK_ptr.dtype.element_ty))
    dV_tma.store([k_tile_idx * K_TILE_SIZE, 0], dV_j.to(dV_ptr.dtype.element_ty))


# @triton.autotune(configs=_bkwd_cfg, key=["N_QUERIES", "N_KEYS"])
@triton.jit
def flash_bkwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dQ_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    N_QUERIES,
    N_KEYS,
    scale,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    Q_tma = tl.make_tensor_descriptor(
        Q_ptr + b_idx * stride_qb,
        shape=[N_QUERIES, D_MODEL],
        strides=[stride_qq, 1],
        block_shape=[Q_TILE_SIZE, D_MODEL],
    )
    dQ_tma = tl.make_tensor_descriptor(
        dQ_ptr + b_idx * stride_dqb,
        shape=[N_QUERIES, D_MODEL],
        strides=[stride_dqq, 1],
        block_shape=[Q_TILE_SIZE, D_MODEL],
    )
    dO_tma = tl.make_tensor_descriptor(
        dO_ptr + b_idx * stride_dob,
        shape=[N_QUERIES, D_MODEL],
        strides=[stride_doq, 1],
        block_shape=[Q_TILE_SIZE, D_MODEL],
    )
    K_tma = tl.make_tensor_descriptor(
        K_ptr + b_idx * stride_kb,
        shape=[N_KEYS, D_MODEL],
        strides=[stride_kk, 1],
        block_shape=[K_TILE_SIZE, D_MODEL],
    )
    V_tma = tl.make_tensor_descriptor(
        V_ptr + b_idx * stride_vb,
        shape=[N_KEYS, D_MODEL],
        strides=[stride_vk, 1],
        block_shape=[K_TILE_SIZE, D_MODEL],
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + b_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + b_idx * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i = Q_tma.load([q_tile_idx * Q_TILE_SIZE, 0])
    dQ_i = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32)
    dO_i = dO_tma.load([q_tile_idx * Q_TILE_SIZE, 0])
    L_i = tl.load(L_block_ptr)
    D_i = tl.load(D_block_ptr)

    # bound
    k_tile_bound = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        k_tile_bound = tl.cdiv((q_tile_idx + 1) * Q_TILE_SIZE, K_TILE_SIZE)
    for j in range(k_tile_bound):
        K_j = K_tma.load([j * K_TILE_SIZE, 0])
        V_j = V_tma.load([j * K_TILE_SIZE, 0])

        # 25 tile of attn
        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            # same fade as fwd
            if (j + 1) * K_TILE_SIZE - 1 > q_tile_idx * Q_TILE_SIZE:
                q_idxs = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                k_idxs = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                S_i_j = tl.where(k_idxs[None, :] > q_idxs[:, None], S_i_j + -1e6, S_i_j)

        # 26-30
        P_i_j = tl.exp2((S_i_j - L_i[:, None]) * LOG2E)
        dP_i_j = tl.dot(dO_i, tl.trans(V_j))
        dS_i_j = P_i_j * (dP_i_j - D_i[:, None])
        dQ_i += tl.dot(dS_i_j.to(K_j.dtype), K_j) * scale

    dQ_tma.store([q_tile_idx * Q_TILE_SIZE, 0], dQ_i.to(dQ_ptr.dtype.element_ty))


# ngl woulda learned more without handout guide


class FlashAttentionKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # assume shapes are right (we have no checks)
        B, n_q, d = Q.shape[0], Q.shape[1], Q.shape[2]
        B, n_k, d = K.shape[0], K.shape[1], K.shape[2]
        B, n_v, d = V.shape[0], V.shape[1], V.shape[2]
        _ensure_triton_allocator(Q.device)
        # init output O,L
        O = torch.empty((B, n_q, d), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, n_q), device=Q.device, dtype=torch.float32)  # fix high prec

        b_q, b_k = TILE_SIZE1, TILE_SIZE2
        # triton cdiv auto rounds up to nearest int
        flash_fwd_kernel[(triton.cdiv(n_q, b_q), B)](
            Q,
            K,
            V,  # in
            O,
            L,  # out
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            n_q,
            n_k,
            d**-0.5,  # scale
            D=d,  # d_model
            Q_TILE_SIZE=b_q,
            K_TILE_SIZE=b_k,
            is_causal=is_causal,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        # assume shapes are right (we have no checks)
        B, n_q, d = Q.shape[0], Q.shape[1], Q.shape[2]
        B, n_k, d = K.shape[0], K.shape[1], K.shape[2]
        B, n_v, d = V.shape[0], V.shape[1], V.shape[2]
        _ensure_triton_allocator(Q.device)

        # init output grads
        D = torch.einsum("...nd,...nd->...n", grad_out, O).to(torch.float32)  # rowsum(dO * O)
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        b_q, b_k = TILE_SIZE1, TILE_SIZE2
        # triton cdiv auto rounds up to nearest int
        flash_bkwd_kernel[(triton.cdiv(n_k, b_k), B)](
            Q,
            K,
            V,  # in
            grad_out,
            L,
            D,
            dK,
            dV,  # out
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            L.stride(0),
            L.stride(1),
            D.stride(0),
            D.stride(1),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            n_q,
            n_k,
            d**-0.5,  # scale
            D_MODEL=d,  # d_model
            Q_TILE_SIZE=b_q,
            K_TILE_SIZE=b_k,
            is_causal=ctx.is_causal,
            num_warps=4,
            num_stages=1,
        )

        # goddammit second kernel
        flash_bkwd_dq_kernel[(triton.cdiv(n_q, b_q), B)](
            Q,
            K,
            V,  # in
            grad_out,
            L,
            D,
            dQ,  # out
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            L.stride(0),
            L.stride(1),
            D.stride(0),
            D.stride(1),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            n_q,
            n_k,
            d**-0.5,  # scale
            D_MODEL=d,  # d_model
            Q_TILE_SIZE=b_q,
            K_TILE_SIZE=b_k,
            is_causal=ctx.is_causal,
            num_warps=4,
            num_stages=2,
        )

        return dQ, dK, dV, None  # no for causal


# NOTE: can do 3


# cahnged to log2e, hint from online
# made everything come from empty

import triton
import triton.language as tl
import torch
import math

# Attention forward inner
@triton.jit
def _attn_fwd_inner(
    O_block,
    m_i,
    l_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by the current stage.
    # Need to understand why STAGE values here are different from the
    # STAGE values passed in the _attn_fwd kernel, i.e. why STAGE = 4 - STAGE?
    if STAGE == 1:
        # All the blocks to the left of the diagonal in the causal/non-causal attention.
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Blocks on the diagonal in causal attention.
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # only used in non-causal attention.
        ...

    # This is the block of K and V that we are processing.
    # advance the pointer to the start of the block depending on the call
    # to this function.
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Let the compiler know that start_kv is a multiple of BLOCK_SIZE_KV.
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        K_block = tl.load(K_block_ptr, mask=None)
        # Remember K_block is already transposed.
        QK_block = tl.dot(Q_block, K_block)

        # if STAGE == 2, we know some values will be valid and some will be non-causal
        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]
        
        # Compute the exponent of each dot product
        P_block = tl.math.exp(QK_block)

        # For the current block, compute the sum of the probabilities.
        l_ij = tl.sum(P_block, axis=1)

        # Correction factor; m_i = running max of the softmax block.
        alpha = tl.math.exp(m_i - m_ij)

        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr, mask=None)

        O_block = O_block * alpha[:, None]
        # C = tl.dot(A, B, C) --> C += A @ B
        # Does this invoke the MAC?
        P_block = P_block.to(tl.float16)
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))

    return O_block, l_i, m_i

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64] # , 128]
        for BLOCK_SIZE_KV in [32] #, 64]
        for num_stages in ([3])
        for num_warps in [2]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)

@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    softmax_scale,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # Index into Q to figure out where does the head start
    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    # stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim - These are obtained directly from the tensor.
    # stride_Q_batch = how much to add to get to the next batch.
    # stride_Q_head = how much to add to get to the next head in the same batch.
    # stride_Q_seq = how much to add to get to the next sequence in the same head and batch.
    # stride_Q_dim = how much to add to get to the next dimension in the same sequence, head and batch.
    q_block_ptr = tl.make_block_ptr(  # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        base=Q + qkv_offset,  # a 2D tensor
        shape=(SEQ_LEN, HEAD_DIM),
        # stride_Q_seq: Specifies how many elements you need to move in memory to go from one row to the next
        # stride_Q_dim: Specifies how many elements you need to move in memory to go from one column to the next in the same row.
        strides=(stride_Q_seq, stride_Q_dim),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        # In the tensor view, the start offsets of the queries this block will work on.
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        # Order means the indexing should prioritize the second axis first, followed by the first axis.
        # i.e. data in the block is processed column-wise before row-wise.
        order=(1, 0)
    )
    
    # NOTE: Not skipping KVs into a block of KVs.
    v_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        offsets=(0, 0),
        # TODO: what is this order?
        order=(1, 0)
    )

    # K should be indexed in the transposed manner.
    k_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        # offsets are (0, 0) because we are not skipping anything. We are at the beginning of the cache block
        offsets=(0, 0),
        # If ordering is to prioritize the first axis first followed by the second axis and we are already
        # transposing the K block, then is this order negating the transpose?
        # TODO - really understand what this order means.
        order=(0, 1)
    )

    # How many outputs do we generate?
    O_block_ptr = tl.make_block_ptr(  # O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        base=O + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        order=(1, 0)
    )

    # load Q blocks
    Q_block = tl.load(q_block_ptr, mask=None)

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    # offs_kv: the offsets for the token in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: the running maximum of the softmax block.
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')

    # l_i: the running sum of the softmax block. We have one for each query
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0  # added in the algorithm

    # output block for the current rows of query.
    o_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # What are stages? Just following the tutorial for now.
    if STAGE == 1 or STAGE == 3:
        # This step runs for the blocks to the left of the diagonal in causal attention
        o_block, l_i, m_i = _attn_fwd_inner(
            o_block,
            m_i,
            l_i,
            Q_block,
            k_block_ptr,
            v_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        o_block, l_i, m_i = _attn_fwd_inner(
            o_block,
            m_i,
            l_i,
            Q_block,
            k_block_ptr,
            v_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,  # I have no idea why these weird Stage numberings.
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    # Indeed a smart trick. No longer need to divide and rather just do a lot of subtract.
    m_i += tl.math.log(
        l_i
    ) # This is needed to compute the logsumexp for the backward pass. But I don't care right now.

    o_block = o_block / l_i[:, None]

    # M --> BATCH_SIZE, NUM_HEADS, SEQ_LEN
    # index_batch_head --> heads in a batch are laid out contiguously and each
    # head has a SEQ_LEN long logsumexp.
    # M = points to the beginning of the tensor
    # offs_q = points to the beginning of the block of queries we are processing in this block
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q

    tl.store(m_ptrs, m_i, mask=None)
    tl.store(O_block_ptr, o_block.to(O.type.element_ty), mask=None)


# If we want to define a function that we can backpropagate through, the class needs to be
# derived from torch.autograd.Function.
class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # ctx allows us to save intermediate results for backward pass.

        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        O = torch.empty_like(Q)

        # What is stage?
        stage = 3 if causal else 1
        
        # Parallelize over the batch dimension and the number of heads
        grid = lambda args: (
            # ceil(SEQ_LEN / BLOCK_SIZE_Q) = How many blocks of Q we have.
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']),
            BATCH_SIZE * NUM_HEADS,
            1, # z in the CUDA launch grid
        )

        # M is the logsumexp for the backward pass, one for each query.
        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            O=O,
            M=M,
            softmax_scale=softmax_scale,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage
        )

        # save the intermediate results for backward pass.
        ctx.save_for_backward(Q, K, V, O, M)
        return O
    

import math

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16, DEVICE='cuda'):
    Q = (torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=DEVICE, dtype=dtype
        )
        .normal_(mean=0.0, std=0.5)
    )
    K = (torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=DEVICE, dtype=dtype
        )
        .normal_(mean=0.0, std=0.5)
    )
    V = (torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=DEVICE, dtype=dtype
        )
        .normal_(mean=0.0, std=0.5)
    )

    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    # Backpropagate through the output.
    dO = torch.randn_like(Q)

    mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale

    if causal:
        P[..., mask == 0] = float('-inf')
    P = torch.softmax(P, dim=-1).half()
    ref_O = torch.matmul(P, V)

    if False:
        # Skip backward pass for now.
        ref_O.backward(dO)

        ref_dV, V.grad = V.grad.clone(), None
        ref_dP, P.grad = P.grad.clone(), None
        ref_dK, K.grad = K.grad.clone(), None
        ref_dQ, Q.grad = Q.grad.clone(), None

    # Compare with Triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale)
    
    # Compare
    rtol = 0.0
    atol = 1e-2
    print(torch.allclose(tri_out, ref_O, rtol=rtol, atol=atol))

if __name__ == "__main__":
    test_op(BATCH_SIZE=4, NUM_HEADS=4, SEQ_LEN=512, HEAD_DIM=128, causal=True)
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from einops import pack, rearrange, unpack
import torch.nn.functional as F  # REMOVE REMOVE
from timeit import default_timer as timer
from jaxtyping import Float, Int
from torch import Tensor

import importlib
from llm_training.systems.ddp_naive_bench import setup
from llm_training.reference.model import BasicsTransformerLM, Linear, RotaryEmbedding
from llm_training.reference.optimizer import AdamW
from llm_training.reference.nn_utils import cross_entropy
from llm_training.systems.flash import FlashAttentionKernel
import llm_training.reference.model

from llm_training.systems.ddp import DDP
from llm_training.systems.sharded_optimizer import ShardedOptimizer
################# Leaderboard ################


class LeaderboardCausalAttention(nn.Module):
    """Same specs with some removals/changes. QKV fuse and no mask build"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding | None = None,
    ):
        super().__init__()
        if positional_encoder is None:
            warnings.warn("No positional encoder provided", stacklevel=2)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder: RotaryEmbedding | None = positional_encoder  # RoPE

    def forward(
        self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *batch_dims, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )  # fmt: skip

        if self.positional_encoder is not None:  # RoPE is enabled
            if token_positions is not None:  # We got explicit position ids
                # Duplicate token positions for each head
                token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        # Construct causal mask
        # iota = torch.arange(sequence_length, device=x.device)
        # qi = rearrange(iota, "query -> query 1")
        # kj = rearrange(iota, "key   -> 1   key")
        # causal_mask = qi >= kj  # (query, key)
        # causal_mask = causal_mask.__getitem__((None,) * len(batch_dims) + (...,))  # Add appropriate leading dimensions

        # Shape: (..., num_heads, sequence_length, d_k)
        attn_output = llm_training.reference.model.scaled_dot_product_attention(K=K, Q=Q, V=V, mask=None)

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output


################### Model changes ^^ ############


class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2


def ddp_leaderboard_bench(rank, world_size):

    triton_testing = importlib.import_module("triton.testing")

    setup(rank, world_size)
    torch.cuda.set_device(rank)  # tip

    cfg = Config()
    device = f"cuda:{rank}"

    # quick torch check #REMOVE LATER REMOVE REMOVE
    def flash_attention(Q, K, V, mask=None):
        Q, ps = pack([Q], "* n_q d")
        K, _ = pack([K], "* n_k d")
        V, _ = pack([V], "* n_k d")
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        out = FlashAttentionKernel.apply(Q, K, V, True)
        return unpack(out, ps, "* n_q d")[0]
        # return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    llm_training.reference.model.scaled_dot_product_attention = flash_attention
    llm_training.reference.model.CausalMultiHeadSelfAttention = LeaderboardCausalAttention

    model = BasicsTransformerLM(
        cfg.vocab_size,
        cfg.ctx_len,
        cfg.d_model,
        cfg.num_layers,
        cfg.num_heads,
        cfg.d_ff,
    ).to(device=device, dtype=cfg.torch_dtype)

    model = DDP(model)

    m = model.module

    m.ln_final.forward = torch.compile(m.ln_final.forward)
    for block in m.layers:
        block.ln1.forward = torch.compile(block.ln1.forward)
        block.ln2.forward = torch.compile(block.ln2.forward)
        block.ffn.forward = torch.compile(block.ffn.forward)
        # block.forward = torch.compile(block.forward)

    # per-block activation checkpointing
    for block in model.module.layers[:-16]:
        # for block in model.module.layers:
        orig = block.forward
        block.forward = lambda x, _orig=orig: ckpt.checkpoint(_orig, x, use_reentrant=False)

    optimizer = ShardedOptimizer(model.parameters(), AdamW)

    # standard atp
    if rank == 0:
        labels = torch.randint(high=cfg.vocab_size, size=(cfg.batch_size, cfg.ctx_len), device=device)
        targets = torch.randint(high=cfg.vocab_size, size=(cfg.batch_size, cfg.ctx_len), device=device)
    else:
        labels = torch.empty((cfg.batch_size, cfg.ctx_len), dtype=torch.long, device=device)
        targets = torch.empty((cfg.batch_size, cfg.ctx_len), dtype=torch.long, device=device)

    dist.broadcast(labels, src=0)
    dist.broadcast(targets, src=0)

    local_b = cfg.batch_size // world_size
    local_labels = labels[rank * local_b : (rank + 1) * local_b]
    local_targets = targets[rank * local_b : (rank + 1) * local_b]

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = model(local_labels)
        if rank == 0:
            print("finished forward")
        loss = cross_entropy(res, local_targets).sum()
        loss.backward()
        model.finish_gradient_synchronization()
        if rank == 0:
            print("finished backward")
        optimizer.step()
        dist.barrier()
        torch.cuda.synchronize()

    timing_results = triton_testing.do_bench(train_step, rep=30_000, warmup=10_000)
    # train_step()
    # train_step()
    # train_step()
    # start = timer()
    # train_step()
    # end = timer()
    # timing_results = end - start

    gathered_times = [None] * world_size
    dist.all_gather_object(gathered_times, timing_results)
    if rank == 0:
        print(gathered_times)

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD_SIZE = 2
    mp.spawn(fn=ddp_leaderboard_bench, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.checkpoint as ckpt
from einops import pack, unpack

import importlib
from llm_training.systems.ddp_naive_bench import setup
from llm_training.reference.model import BasicsTransformerLM
from llm_training.reference.optimizer import AdamW
from llm_training.reference.nn_utils import cross_entropy
from llm_training.systems.flash import FlashAttentionKernel
import llm_training.reference.model

from llm_training.systems.fsdp import FSDP
import torch.nn.functional as F  # REMOVE REMOVE
################# Leaderboard ################


# class Config:
#     ctx_len = 32768
#     vocab_size = 151936
#     d_model = 4096
#     d_ff = 11008
#     num_layers = 34
#     num_heads = 32
#     torch_dtype = torch.bfloat16
#     is_causal = True
#     batch_size = 2


class Config:
    ctx_len = 512
    vocab_size = 10_000
    d_model = 2560
    d_ff = 10240
    num_layers = 32
    num_heads = 40
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 4


def fsdp_leaderboard_bench(rank, world_size):
    triton_testing = importlib.import_module("triton.testing")

    setup(rank, world_size)
    torch.cuda.set_device(rank)  # tip

    cfg = Config()
    device = f"cuda:{rank}"

    # quick torch check #REMOVE LATER REMOVE REMOVE
    def flash_attention(Q, K, V, mask=None):
        # cast
        Q = Q.to(V.dtype)
        K = K.to(V.dtype)
        Q, ps = pack([Q], "* n_q d")
        K, _ = pack([K], "* n_k d")
        V, _ = pack([V], "* n_k d")
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        out = FlashAttentionKernel.apply(Q, K, V, True)
        return unpack(out, ps, "* n_q d")[0]
        # return F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    llm_training.reference.model.scaled_dot_product_attention = flash_attention

    model = BasicsTransformerLM(
        cfg.vocab_size,
        cfg.ctx_len,
        cfg.d_model,
        cfg.num_layers,
        cfg.num_heads,
        cfg.d_ff,
    ).to(device)

    model = FSDP(model, compute_dtype=cfg.torch_dtype)

    # per-block activation checkpointing
    for block in model.module.layers:
        orig = block.forward
        block.forward = lambda x, _orig=orig: ckpt.checkpoint(_orig, x, use_reentrant=False)

    optimizer = AdamW(model.parameters())

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
        loss = cross_entropy(res, local_targets).sum()
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        dist.barrier()
        torch.cuda.synchronize()

    timing_results = triton_testing.do_bench(train_step, rep=30_000, warmup=10_000)

    gathered_times = [None] * world_size
    dist.all_gather_object(gathered_times, timing_results)
    if rank == 0:
        print(gathered_times)

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD_SIZE = 2
    mp.spawn(fn=fsdp_leaderboard_bench, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

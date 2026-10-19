import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer


# same as handout, what is 29500?
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"  # one machine, so we localhost
    os.environ["MASTER_PORT"] = "29500"  # any free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def comm_bench(rank, world_size, size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # tip
    WARMUPS = 5
    PASSES = 10

    # 1 mb tensor
    size = size * 2 ** (20 - 2)  # -2 for float32

    data = torch.randn(size, device=f"cuda:{rank}", dtype=torch.float32)  # per rank device, each elem is 4 bytes

    for _ in range(WARMUPS):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()

    # dist.barrier()

    times = list()
    for _ in range(PASSES):
        start = timer()
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        times.append(timer() - start)

    gathered_times = [None] * world_size
    dist.all_gather_object(gathered_times, times)

    if rank == 0:
        print(f"gpu count: {world_size}")
        for r, ts in enumerate(gathered_times):
            print(f"gpu {r}: mean={sum(ts) / len(ts):.6f}s, times={ts}")
        # overall mean
        print(f"overall mean: {sum(sum(ts) for ts in gathered_times) / sum(len(ts) for ts in gathered_times):.6f}s")

    dist.destroy_process_group()  # doesnt hurt


if __name__ == "__main__":
    WORLD_SIZES = [2, 4, 6]
    SIZES = [1, 10, 100, 1000]
    for world_size in WORLD_SIZES:
        for size in SIZES:
            mp.spawn(fn=comm_bench, args=(world_size, size), nprocs=world_size, join=True)

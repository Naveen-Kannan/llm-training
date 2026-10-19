import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer

from llm_training.reference.nn_utils import cross_entropy
from llm_training.systems.ddp import DDP
from llm_training.systems.ddp_naive_bench import _get_model_opt_data, setup

from llm_training.systems.sharded_optimizer import ShardedOptimizer


def sharded_opt_bench(rank, world_size, batch_size, model_size, use_sharded):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    WARMUPS = 2
    PASSES = 5

    device = f"cuda:{rank}"

    # ignore opt now
    model, _, input_data, target_data = _get_model_opt_data(device, batch_size, model_size)

    model = DDP(model)

    if use_sharded:
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-3)
        rank_optimizer = optimizer.optimizer
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # can we scatter this data? tech debt too much to change now
    v, c = 10000, 512
    if rank == 0:
        input_data = torch.randint(0, v, (batch_size, c), device=device)
        target_data = torch.randint(0, v, (batch_size, c), device=device)
    else:
        input_data = torch.empty((batch_size, c), dtype=torch.long, device=device)
        target_data = torch.empty((batch_size, c), dtype=torch.long, device=device)

    dist.broadcast(input_data, src=0)
    dist.broadcast(target_data, src=0)

    local_b = batch_size // world_size
    local_input_data = input_data[rank * local_b : (rank + 1) * local_b]
    local_target_data = target_data[rank * local_b : (rank + 1) * local_b]

    # memory right after model + optimizer init

    # can get rid of a sync?
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(rank)

    torch.cuda.synchronize()
    mem_after_init = torch.cuda.memory_allocated(rank) / 1024**3

    # warmup
    for _ in range(WARMUPS):
        optimizer.zero_grad()

        out = model(local_input_data)
        loss = cross_entropy(out, local_target_data)
        loss.backward()

        model.finish_gradient_synchronization()

        optimizer.step()
        torch.cuda.synchronize()

    step_times = []
    mem_pre = []
    mem_post = []

    for _ in range(PASSES):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(rank)

        out = model(local_input_data)
        loss = cross_entropy(out, local_target_data)
        loss.backward()

        model.finish_gradient_synchronization()

        torch.cuda.synchronize()
        mem_pre_step = torch.cuda.max_memory_allocated(rank) / 1024**3

        start = timer()
        optimizer.step()
        torch.cuda.synchronize()
        step_time = timer() - start  # time goes first
        mem_post_step = torch.cuda.max_memory_allocated(rank) / 1024**3

        step_times.append(step_time)
        mem_pre.append(mem_pre_step)
        mem_post.append(mem_post_step)

    mem_pre_step = max(mem_pre)
    mem_post_step = max(mem_post)

    opt = optimizer.optimizer if use_sharded else optimizer
    tot = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                tot = tot + v.numel() * v.element_size()
    opt_state_gib = tot / (2**30)

    gathered = [None] * world_size
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "times": step_times,
            "mem_init": mem_after_init,
            "mem_pre_step": mem_pre_step,
            "mem_post_step": mem_post_step,
            "opt_state_gib": opt_state_gib,
        },  # dict
    )

    if rank == 0:
        typ = "sharded" if use_sharded else "normal"
        print(f"{typ}")
        print(f"world size: {world_size}")
        print(f"batch size: {batch_size}")
        print(f"model size: {model_size}")

        all_times = [t for g in gathered for t in g["times"]]
        print(f"optimizer.step mean: {sum(all_times) / len(all_times):.6f}s")
        all_mem_init = [g["mem_init"] for g in gathered]
        print(f"mem_init mean: {sum(all_mem_init) / len(all_mem_init):.3f} GiB")
        all_mem_pre = [g["mem_pre_step"] for g in gathered]
        all_mem_post = [g["mem_post_step"] for g in gathered]
        print(f"mem_pre mean: {sum(all_mem_pre) / len(all_mem_pre):.3f} GiB")
        print(f"mem_post mean: {sum(all_mem_post) / len(all_mem_post):.3f} GiB")
        all_opt_state = [g["opt_state_gib"] for g in gathered]
        print(f"opt_state mean: {sum(all_opt_state) / len(all_opt_state):.3f} GiB")

        # for g in gathered:
        #     ts = g["times"]
        #     print(
        #         f"gpu {g['rank']}: "
        #         f"step={sum(ts)/len(ts):.6f}s, "
        #         f"mem_init={g['mem_init']:.3f} GiB, "
        #         f"mem_before_step={g['mem_before_step']:.3f} GiB, "
        #         f"mem_after_step={g['mem_after_step']:.3f} GiB, "
        #         f"optimizer_state={g['opt_state_gib']:.3f} GiB"
        #     )

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD_SIZE = 2
    BATCH_SIZE = 4
    MODEL_SIZE = "xl"

    mp.spawn(
        fn=sharded_opt_bench,
        args=(WORLD_SIZE, BATCH_SIZE, MODEL_SIZE, False),
        nprocs=WORLD_SIZE,
        join=True,
    )

    mp.spawn(
        fn=sharded_opt_bench,
        args=(WORLD_SIZE, BATCH_SIZE, MODEL_SIZE, True),
        nprocs=WORLD_SIZE,
        join=True,
    )

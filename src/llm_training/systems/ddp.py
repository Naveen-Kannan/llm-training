import torch
import torch.distributed as dist
from llm_training.systems.ddp_naive_bench import _get_model_opt_data, setup
import torch.multiprocessing as mp
from timeit import default_timer as timer
from llm_training.reference.nn_utils import cross_entropy
import torch.cuda.nvtx as nvtx

# do something with the register_post_accumulate_grad_hook function
# use async_op = True
# need handle.wait() to see if on gpu queue


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.world_size = dist.get_world_size()  # should be 2, how to set? in dist.init_process_group()
        self.all_handles = list()
        self.module = module
        # broadcast the params
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        # one hook per param
        for param in module.parameters():
            if param.requires_grad:  # tuff
                param.register_post_accumulate_grad_hook(self.grad_hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.all_handles:
            handle.wait()
        self.all_handles.clear()

    def grad_hook(self, param):
        with nvtx.range("ddp grad sync launch"):
            param.grad /= self.world_size
            self.all_handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True))


# why is this so much simpler than naive code
# we dont ahve to split the data and all


# copied script from naive bench
def ddp_bench(rank, world_size, batch_size, model_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # tip

    WARMUPS = 5
    PASSES = 10

    model, optimizer, input_data, target_data = _get_model_opt_data(f"cuda:{rank}", batch_size, model_size)

    model = DDP(model)

    # create data on 0, broadcast to all ranks
    v, c = 10000, 512
    if rank == 0:
        input_data = torch.randint(0, v, (batch_size, c), device=f"cuda:{rank}")
        target_data = torch.randint(0, v, (batch_size, c), device=f"cuda:{rank}")
    else:
        input_data = torch.empty((batch_size, c), dtype=torch.long, device=f"cuda:{rank}")
        target_data = torch.empty((batch_size, c), dtype=torch.long, device=f"cuda:{rank}")

    dist.broadcast(input_data, src=0)
    dist.broadcast(target_data, src=0)

    local_b = batch_size // world_size  # should be divis

    # take the data slice needed
    local_input_data = input_data[rank * local_b : (rank + 1) * local_b]
    local_target_data = target_data[rank * local_b : (rank + 1) * local_b]

    step_times = list()

    # warmup whole process
    for _ in range(WARMUPS):
        optimizer.zero_grad()

        out = model(local_input_data)
        loss = cross_entropy(out, local_target_data)
        loss.backward()

        model.finish_gradient_synchronization()

        optimizer.step()
        torch.cuda.synchronize()

    # dist.barrier()

    for _ in range(PASSES):
        optimizer.zero_grad()

        step_start = timer()

        out = model(local_input_data)
        loss = cross_entropy(out, local_target_data)
        loss.backward()

        model.finish_gradient_synchronization()

        optimizer.step()
        torch.cuda.synchronize()
        step_time = timer() - step_start

        step_times.append(step_time)

    gathered_times = [None] * world_size
    dist.all_gather_object(gathered_times, step_times)

    # torch.cuda.synchronize() for timing stuff
    # use dist.all_gather_object to gather
    if rank == 0:
        tot_step_times = [t for step_times in gathered_times for t in step_times]
        step_mean = sum(tot_step_times) / len(tot_step_times)

        print(f"world size: {world_size}")
        print(f"batch size: {batch_size}")
        print(f"model size: {model_size}")
        print(f"step mean: {step_mean:.6f}s")

        # per gpu times
        for r, step_times in enumerate(gathered_times):
            print(f"gpu {r}: step mean={sum(step_times) / len(step_times):.6f}s")

    dist.destroy_process_group()

    # i think i was meant to return the model at the end... so that i could now do the check again with the single process model... oops.


if __name__ == "__main__":
    WORLD_SIZE = 2
    BATCH_SIZE = 4  # must be a mult of WORLD_SIZE
    mp.spawn(fn=ddp_bench, args=(WORLD_SIZE, BATCH_SIZE, "xl"), nprocs=WORLD_SIZE, join=True)

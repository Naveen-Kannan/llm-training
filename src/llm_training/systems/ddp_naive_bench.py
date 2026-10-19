import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer
from llm_training.reference.model import BasicsTransformerLM
from llm_training.reference.nn_utils import cross_entropy
from llm_training.reference.optimizer import AdamW
from copy import deepcopy
import torch.cuda.nvtx as nvtx


def _get_model_opt_data(device: str, b: int, size: str = "small"):
    if size == "small":
        v, c, d, l, h, f, t = 10000, 512, 768, 12, 12, 3072, 10000.0
    else:  # size == 'xl
        v, c, d, l, h, f, t = 10000, 512, 2560, 32, 32, 10240, 10000.0

    model = BasicsTransformerLM(v, c, d, l, h, f, t).to(device)

    optimizer = AdamW(model.parameters())

    # rand data batch
    input_data = torch.randint(0, v, (b, c))
    input_data = input_data.to(device)
    target_data = torch.randint(0, v, (b, c))
    target_data = target_data.to(device)

    return model, optimizer, input_data, target_data


# same as handout, what is 29500?
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"  # one machine, so we localhost
    os.environ["MASTER_PORT"] = "29500"  # any free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)


def ddp_naive(rank, world_size, batch_size, model_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # tip

    WARMUPS = 5
    PASSES = 10

    model, optimizer, input_data, target_data = _get_model_opt_data(f"cuda:{rank}", batch_size, model_size)

    # broadcast the parameters?
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

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
    comm_times = list()

    # warmup whole process
    for _ in range(WARMUPS):
        optimizer.zero_grad()

        out = model(local_input_data)
        loss = cross_entropy(out, local_target_data)
        loss.backward()

        # NAIVE
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size

        # BATCHED
        # grads_batch = [param.grad for param in model.parameters() if param.grad is not None] # not just if para.mgrad
        # grad_params_batch = torch._utils._flatten_dense_tensors(grads_batch)
        # dist.all_reduce(grad_params_batch, op=dist.ReduceOp.SUM)
        # grad_params_batch /= world_size
        # grads_avg = torch._utils._unflatten_dense_tensors(grad_params_batch, grads_batch)

        # # since the apram.grad are objs we can just copy in place
        # for param_grad, param_grad_avg in zip(grads_batch, grads_avg):
        #     param_grad.copy_(param_grad_avg)

        optimizer.step()
        torch.cuda.synchronize()

    # dist.barrier()

    for _ in range(PASSES):
        optimizer.zero_grad()

        step_start = timer()

        out = model(local_input_data)
        loss = cross_entropy(out, local_target_data)
        loss.backward()

        torch.cuda.synchronize()
        comm_start = timer()

        with nvtx.range("ddp grad sync launch"):
            # NAIVE
            # avg grads per param
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= world_size

        # BATCHED
        # grads_batch = [param.grad for param in model.parameters() if param.grad is not None] # not just if para.mgrad
        # grad_params_batch = torch._utils._flatten_dense_tensors(grads_batch)
        # dist.all_reduce(grad_params_batch, op=dist.ReduceOp.SUM)
        # grad_params_batch /= world_size
        # grads_avg = torch._utils._unflatten_dense_tensors(grad_params_batch, grads_batch)

        # # since the apram.grad are objs we can just copy in place
        # for param_grad, param_grad_avg in zip(grads_batch, grads_avg):
        #     param_grad.copy_(param_grad_avg)

        torch.cuda.synchronize()
        comm_time = timer() - comm_start

        optimizer.step()
        torch.cuda.synchronize()
        step_time = timer() - step_start

        step_times.append(step_time)
        comm_times.append(comm_time)

    gathered_times = [None] * world_size
    dist.all_gather_object(gathered_times, (step_times, comm_times))

    # torch.cuda.synchronize() for timing stuff
    # use dist.all_gather_object to gather
    if rank == 0:
        tot_step_times = [t for step_times, _ in gathered_times for t in step_times]
        tot_comm_times = [t for _, comm_times in gathered_times for t in comm_times]

        step_mean = sum(tot_step_times) / len(tot_step_times)
        comm_mean = sum(tot_comm_times) / len(tot_comm_times)

        print(f"world size: {world_size}")
        print(f"batch size: {batch_size}")
        print(f"model size: {model_size}")
        print(f"step mean: {step_mean:.6f}s")
        print(f"comm mean: {comm_mean:.6f}s")
        print(f"comm fraction: {comm_mean / step_mean:.3f}")

        # per gpu times
        for r, (step_times, comm_times) in enumerate(gathered_times):
            print(
                f"gpu {r}: step mean={sum(step_times) / len(step_times):.6f}s, comm mean={sum(comm_times) / len(comm_times):.6f}s"
            )

    dist.destroy_process_group()

    # i think i was meant to return the model at the end... so that i could now do the check again with the single process model... oops.


if __name__ == "__main__":
    WORLD_SIZE = 2
    BATCH_SIZE = 4  # must be a mult of WORLD_SIZE
    mp.spawn(fn=ddp_naive, args=(WORLD_SIZE, BATCH_SIZE, "xl"), nprocs=WORLD_SIZE, join=True)

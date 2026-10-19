import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer
from llm_training.reference.model import BasicsTransformerLM
from llm_training.reference.nn_utils import cross_entropy
from llm_training.reference.optimizer import AdamW
from copy import deepcopy


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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def ddp_naive(rank, world_size, batch_size, model_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)  # tip

    model, optimizer, input_data, target_data = _get_model_opt_data(f"cuda:{rank}", batch_size, model_size)

    # save a copy of optimizer, model
    if rank == 0:
        model_copy = deepcopy(model)
        optimizer_copy = AdamW(model_copy.parameters())

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

    # do one forward pass
    out = model(local_input_data)
    loss = cross_entropy(out, local_target_data)
    loss.backward()

    # avg grads per param
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

    optimizer.step()  # can just do this on each rank

    # torch.cuda.synchronize() for timing stuff
    # use dist.all_gather_object to gather
    if rank == 0:
        for r in range(world_size):
            local_input_data = input_data[r * local_b : (r + 1) * local_b]
            local_target_data = target_data[r * local_b : (r + 1) * local_b]

            out = model_copy(local_input_data)
            loss = cross_entropy(out, local_target_data)
            loss = loss / world_size  # matches gradient averaging
            loss.backward()

        optimizer_copy.step()

        diff = 0.0
        for p_ddp, p_single in zip(model.parameters(), model_copy.parameters()):
            diff = max((p_ddp - p_single).abs().max().item(), diff)

        print(f"max parameter diff: {diff:.3e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD_SIZE = 2
    BATCH_SIZE = 8  # must be a mult of WORLD_SIZE
    mp.spawn(fn=ddp_naive, args=(WORLD_SIZE, BATCH_SIZE, "small"), nprocs=WORLD_SIZE, join=True)

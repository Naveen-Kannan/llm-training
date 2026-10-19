import numpy as np
import torch


def load_data(path, mmap_mode="r"):
    return np.load(path, mmap_mode=mmap_mode)
    # ngl thought this was harder


def get_batch(x, batch_size, context_length, device):
    seq_idxs = np.random.randint(0, len(x) - context_length, size=batch_size)

    # cntxt length input, 1 for target
    # list comp is fastest
    inputs = [x[idx : idx + context_length] for idx in seq_idxs]
    input_seq = np.stack(inputs)

    targets = [x[idx + 1 : idx + 1 + context_length] for idx in seq_idxs]
    next_token = np.stack(targets)

    # specify device - is dtype long?
    input_tensor = torch.tensor(input_seq, dtype=torch.long, device=device)
    target_tensor = torch.tensor(next_token, dtype=torch.long, device=device)

    return input_tensor, target_tensor


def save_checkpoint(model, optimizer, iteration, out):
    # nest the objs in a dict, pass to torch.save
    save = {
        "opt": optimizer.state_dict(),
        "model": model.state_dict(),
        "itr": iteration,
    }
    torch.save(save, out)


def load_checkpoint(src, model, optimizer):
    # load the saved dict back
    save = torch.load(src, map_location="cpu")  # load on cpu default
    # then pushes to gpu
    model.load_state_dict(save["model"])
    optimizer.load_state_dict(save["opt"])
    return save["itr"]

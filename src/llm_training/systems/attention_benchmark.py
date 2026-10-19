# my cleanest script in a while ngl

import argparse, torch
import pandas as pd
from timeit import default_timer as timer
from llm_training.reference.model import scaled_dot_product_attention as attention
# import torch._functorch.config

# # apparently some problem with retaining the graph
# torch._functorch.config.donated_buffer = False
attention = torch.compile(attention)


def time_forward(Q, K, V, w, passes):
    for _ in range(w):
        y = attention(Q, K, V)

    torch.cuda.synchronize()

    times = []
    for _ in range(passes):
        start = timer()
        y = attention(Q, K, V)
        torch.cuda.synchronize()
        times.append(timer() - start)

    return sum(times) / len(times)


def time_backward(Q, K, V, w, passes):
    y = attention(Q, K, V)
    loss = y.sum() * 0.1  # need something in terms of y, dont want overflow (i tink idrk)
    torch.cuda.synchronize()
    mem = torch.cuda.memory_allocated() / 2**30  # in gibs

    for _ in range(w):
        loss.backward(retain_graph=True)
        Q.grad = None
        K.grad = None
        V.grad = None

    torch.cuda.synchronize()

    times = []
    for _ in range(passes):
        start = timer()
        loss.backward(retain_graph=True)
        torch.cuda.synchronize()
        times.append(timer() - start)
        Q.grad = None
        K.grad = None
        V.grad = None

    return sum(times) / len(times), mem


# combining prev funcs to defeat retain graph issue
def time_forward_backward(Q, K, V, w, passes):
    for _ in range(w):
        Q.grad = None
        K.grad = None
        V.grad = None
        y = attention(Q, K, V)
        loss = y.sum() * 0.1
        loss.backward()

    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []

    y = attention(Q, K, V)
    loss = y.sum() * 0.1
    torch.cuda.synchronize()
    mem = torch.cuda.memory_allocated() / 2**30

    for _ in range(passes):
        Q.grad = None
        K.grad = None
        V.grad = None

        start = timer()
        y = attention(Q, K, V)
        torch.cuda.synchronize()
        fwd = timer()

        loss = y.sum() * 0.1
        loss.backward()
        torch.cuda.synchronize()
        bwd = timer()

        fwd_times.append(fwd - start)
        bwd_times.append(bwd - fwd)

    return sum(fwd_times) / len(fwd_times), sum(bwd_times) / len(bwd_times), mem


def main():
    parser = argparse.ArgumentParser()
    # params from pdf
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--w", type=int, default=5)
    parser.add_argument("--passes", type=int, default=100)
    args = parser.parse_args()

    d_model = [16, 32, 64, 128]
    seq_len = [256, 1024, 4096, 8192, 16384, 32768]
    rows = []  # dataframe build

    for d in d_model:
        for l in seq_len:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # rand without head
                Q = torch.randn(args.batch_size, l, d, device="cuda", requires_grad=True)
                K = torch.randn(args.batch_size, l, d, device="cuda", requires_grad=True)
                V = torch.randn(args.batch_size, l, d, device="cuda", requires_grad=True)

                # fwd = time_forward(Q, K, V, args.w, args.passes)
                # bwd, mem = time_backward(Q, K, V, args.w, args.passes)
                fwd, bwd, mem = time_forward_backward(Q, K, V, args.w, args.passes)

                rows.append(
                    {
                        "seq_len": l,
                        "d_model": d,
                        "fwd": f"{fwd:.6f}",
                        "bwd": f"{bwd:.6f}",
                        "mem": f"{mem:.3f}",
                    }
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append(
                    {
                        "seq_len": l,
                        "d_model": d,
                        "fwd": "OOM",
                        "bwd": "OOM",
                        "mem": "OOM",
                    }
                )

    df = pd.DataFrame(rows)
    print(df.to_latex(index=False))


if __name__ == "__main__":
    main()

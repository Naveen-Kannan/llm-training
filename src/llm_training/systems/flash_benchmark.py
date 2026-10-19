import argparse, torch, triton
import pandas as pd
from llm_training.systems.flash import FlashAttentionKernel
from llm_training.reference.model import scaled_dot_product_attention as attention


# super simle attn bench
def pytorch_attention(q, k, v):
    n = q.shape[1]
    i = torch.arange(n, device=q.device)
    mask = i[:, None] >= i[None, :]
    return attention(q, k, v, mask)


def flash_attention(q, k, v):
    return FlashAttentionKernel.apply(q, k, v, True)


def time_forward(fn, q, k, v):
    return triton.testing.do_bench(lambda: fn(q, k, v))


def time_backward(fn, q, k, v):
    y = fn(q, k, v)
    d_out = torch.randn_like(y)
    return triton.testing.do_bench(
        lambda: torch.autograd.backward(y, d_out, retain_graph=True),
        grad_to_none=[q, k, v],
    )


def time_forward_backward(fn, q, k, v):
    d_out = torch.randn_like(q)
    return triton.testing.do_bench(
        lambda: torch.autograd.backward(fn(q, k, v), d_out),
        grad_to_none=[q, k, v],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=65536)
    args = parser.parse_args()

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    ds = [16, 32, 64, 128]
    # testin
    impls = [("pytorch", pytorch_attention), ("triton", flash_attention)]

    rows = list()

    for dtype in [torch.bfloat16, torch.float32]:
        dtype_name = str(dtype)
        for d in ds:
            for n in seq_lens:
                if n > args.max_seq_len:
                    continue

                for name, fn in impls:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                        q = torch.randn(1, n, d, device="cuda", dtype=dtype, requires_grad=True)
                        k = torch.randn(1, n, d, device="cuda", dtype=dtype, requires_grad=True)
                        v = torch.randn(1, n, d, device="cuda", dtype=dtype, requires_grad=True)

                        # quick warmup so compile doesnt count
                        y = fn(q, k, v)
                        y.sum().backward()
                        q.grad = k.grad = v.grad = None
                        torch.cuda.synchronize()

                        fwd = time_forward(fn, q, k, v)
                        bwd = time_backward(fn, q, k, v)
                        fwd_bwd = time_forward_backward(fn, q, k, v)

                        rows.append(
                            {
                                "impl": name,
                                "dtype": dtype_name,
                                "seq_len": n,
                                "d_model": d,
                                "fwd_ms": f"{fwd:.3f}",
                                "bwd_ms": f"{bwd:.3f}",
                                "fwd_bwd_ms": f"{fwd_bwd:.3f}",
                            }
                        )

                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        rows.append(
                            {
                                "impl": name,
                                "dtype": dtype_name,
                                "seq_len": n,
                                "d_model": d,
                                "fwd_ms": "OOM",
                                "bwd_ms": "OOM",
                                "fwd_bwd_ms": "OOM",
                            }
                        )

    df = pd.DataFrame(rows)
    print(df.to_latex(index=False))


if __name__ == "__main__":
    main()

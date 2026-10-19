from llm_training.reference.model import BasicsTransformerLM
from llm_training.systems.modal_utils import app, build_image, VOLUME_MOUNTS, SUNET_ID, user_volume
from timeit import default_timer as timer
from llm_training.reference.optimizer import AdamW
from llm_training.reference.nn_utils import cross_entropy
import torch, subprocess, os, argparse
import numpy as np
import torch.cuda.nvtx as nvtx
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
import llm_training.reference.model
import math
from llm_training.reference.nn_utils import softmax
from contextlib import nullcontext

VOL = f"basics-{SUNET_ID}"


@nvtx.range("scaled_dot_product_attention")
def scaled_dot_product_attention_nvtx(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("qk"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    with nvtx.range("softmax"):
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("v_mul"):
        out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return out


llm_training.reference.model.scaled_dot_product_attention = scaled_dot_product_attention_nvtx


def forward(model, input_data, target_data=None, optimizer=None, bf16: bool = False):
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    with torch.no_grad():
        with ctx:
            out = model(input_data)
    return out


def forward_backward(model, input_data, target_data, optimizer, bf16: bool = False):
    optimizer.zero_grad()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    with ctx:
        out = model(input_data)
        loss = cross_entropy(out, target_data)
    loss.backward()


def forward_backward_step(model, input_data, target_data, optimizer, bf16: bool = False):
    optimizer.zero_grad()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    with nvtx.range("forward_backward"):
        with ctx:
            out = model(input_data)
            loss = cross_entropy(out, target_data)
        loss.backward()
    optimizer.step()


def _get_model_opt_data(v, c, d, l, h, f, t, b, device, compile_model: bool = False):
    model = BasicsTransformerLM(v, c, d, l, h, f, t).to(device)
    if compile_model:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters())

    # rand data batch
    input_data = torch.randint(0, v, (b, c))
    input_data = input_data.to(device)
    target_data = torch.randint(0, v, (b, c))
    target_data = target_data.to(device)
    return model, optimizer, input_data, target_data


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=1 * 10 * 60,
)
def main(
    v: int = 10000,
    c: int = 512,
    d: int = 512,
    l: int = 4,
    h: int = 16,
    f: int = 1344,
    t: float = 10000.0,
    b: int = 4,
    backward: bool = False,
    step: bool = False,
    w: int = 0,
    device: str = "cuda",
    sample: int = 10,
    bf16: bool = False,
    compile_model: bool = False,
):
    # determine mode
    if step:
        run_func = forward_backward_step
    elif backward:
        run_func = forward_backward
    else:
        run_func = forward

    model, optimizer, input_data, target_data = _get_model_opt_data(v, c, d, l, h, f, t, b, device, compile_model)

    # run warmup steps
    for _ in range(w):
        run_func(model, input_data, target_data, optimizer, bf16)
        torch.cuda.synchronize()

    times = np.zeros(sample)
    for i in range(sample):
        start = timer()
        run_func(model, input_data, target_data, optimizer, bf16)
        torch.cuda.synchronize()
        end = timer()
        times[i] = end - start

    # print with 4 decimal places
    print(f"Mean: {np.mean(times):.4f}")
    print(f"Std: {np.std(times):.4f}")


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=10 * 60,
)
def run_profile(
    v: int = 10000,
    c: int = 512,
    d: int = 768,
    l: int = 12,
    h: int = 12,
    f: int = 3072,
    t: float = 10000.0,
    b: int = 4,
    backward: bool = False,
    step: bool = False,
    w: int = 5,
    device: str = "cuda",
    out: str = "/root/data/profiles/nsys_profile",
):

    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [
        "nsys",
        "profile",
        "--output",
        out,
        "--force-overwrite=true",
        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        "--pytorch=functions-trace,autograd-shapes-nvtx",
        "--cuda-memory-usage=true",  # for f mem torch nsight
        # "--cudabacktrace=all",
        # "--python-backtrace=cuda",
        # "--gpu-metrics-device=0",
        "python",
        "llm_training.systems/benchmark.py",
        "--v",
        str(v),
        "--c",
        str(c),
        "--d",
        str(d),
        "--l",
        str(l),
        "--h",
        str(h),
        "--f",
        str(f),
        "--t",
        str(t),
        "--b",
        str(b),
        "--w",
        str(w),
        "--device",
        device,
    ]

    if backward:
        cmd.append("--backward")
    if step:
        cmd.append("--step")

    # following the slack instr
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"Saved report to {out}.nsys-rep")
    user_volume.commit()
    # pull the report


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=10 * 60,
)
def memory_profile(
    v=10000,
    c=512,
    d=2560,
    l=32,
    h=32,
    f=10240,
    t=10000.0,
    b=4,
    step=False,
    w=3,
    device="cuda",
    bf16=False,
    out="/root/data/profiles/pytorch_profile",
):
    # mostly copied from other func run profile
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # only need these two
    run_func = forward_backward_step if step else forward

    model, optimizer, input_data, target_data = _get_model_opt_data(v, c, d, l, h, f, t, b, device)

    # warmups
    for _ in range(w):
        run_func(model, input_data, target_data, optimizer, bf16)
        torch.cuda.synchronize()

    # follow handout
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    run_func(model, input_data, target_data, optimizer, bf16)
    torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot(f"{out}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"[{out}] peak_alloc={gib:.3f} gb  (c={c}, step={step}, bf16={bf16})")
    # for easy viewing

    user_volume.commit()
    # pull the report for easy viewing


@app.local_entrypoint()
def modal_main(
    v: int = 10000,
    c: int = 512,
    d: int = 768,
    l: int = 12,
    h: int = 12,
    f: int = 3072,
    t: float = 10000.0,
    b: int = 4,
    backward: bool = False,
    step: bool = False,
    w: int = 0,
    sample: int = 10,
    profile: bool = False,
    bf16: bool = False,
    pytorch_profile: bool = False,
    nsys_memory: bool = False,
    compile_model: bool = False,
):
    # nsys mem prifler
    if nsys_memory:
        os.makedirs("profiles/nsys_memory", exist_ok=True)

        # xl size
        d, l, h, f = 2560, 32, 32, 10240

        for c in [128, 1024]:
            tag = f"xl_c{c}_step_mem"
            out = f"/root/data/profiles/nsys_memory/{tag}"
            run_profile.remote(
                v=10000,
                c=c,
                d=d,
                l=l,
                h=h,
                f=f,
                t=10000.0,
                b=4,
                backward=False,
                step=True,
                w=5,
                device="cuda",
                out=out,
            )
            # plz
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    VOL,
                    f"/profiles/nsys_memory/{tag}.nsys-rep",
                    f"profiles/nsys_memory/{tag}.nsys-rep",
                    "--force",
                ],
                check=True,
            )
        return

    if pytorch_profile:
        os.makedirs("profiles/pytorch_profiles", exist_ok=True)

        # xl size
        d, l, h, f = 2560, 32, 32, 10240
        # handout params
        v, t, b = 10000, 10000.0, 4

        runs = [
            # (c, step, bf16)
            # (128,  False, False),
            # (128,  True,  False),
            # (1024, False, False),
            # (1024, True,  False),
            (1024, False, True),
            # (1024, True,  True),
            (128, False, True),
            (128, True, True),
        ]

        for c, step, bf16 in runs:
            tag = f"xl_c{c}_{'step' if step else 'fwd'}_{'bf16' if bf16 else 'fp32'}"
            out = f"/root/data/profiles/pytorch_profiles/{tag}"

            memory_profile.remote(
                v=v,
                c=c,
                d=d,
                l=l,
                h=h,
                f=f,
                t=t,
                b=b,
                step=step,
                w=5,
                bf16=bf16,
                out=out,
            )
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    VOL,
                    f"/profiles/pytorch_profiles/{tag}.pickle",
                    f"profiles/pytorch_profiles/{tag}.pickle",
                    "--force",
                ],
                check=True,
            )

        return

    if profile:
        SIZES = [
            # (768, 12, 12, 3072), # small
            (1024, 24, 16, 4096),  # medium
        ]
        CONTEXTS = [256, 512, 2048]

        for d, l, h, f in SIZES:
            for c in CONTEXTS:
                tag = f"d{d}_l{l}_h{h}_f{f}_c{c}"

                out = f"/root/data/profiles/{tag}_step"

                run_profile.remote(v, c, d, l, h, f, t, b, False, True, 5, "cuda", out)

                subprocess.run(
                    [
                        "modal",
                        "volume",
                        "get",
                        VOL,
                        f"/profiles/{tag}_step.nsys-rep",
                        f"profiles/{tag}_step.nsys-rep",
                        "--force",
                    ],
                    check=True,
                )

        # RUN PROFILE SUBPROCESS
        # run_profile.remote(v, c, d, l, h, f, t, b, backward, step, max(w, 5), "cuda")

        # subprocess.run(
        #     [
        #         "modal", "volume", "get", VOL,
        #         "/profiles/nsys_profile.nsys-rep",
        #         "profiles/nsys_profile.nsys-rep",
        #         "--force",
        #     ],
        #     check=True,
        # )
        return

    # main.remote(v, c, d, l, h, f, t, b, backward, step, w, "cuda", sample)

    MODEL_SIZES = [
        (768, 12, 12, 3072),  # small
        (1024, 24, 16, 4096),  # medium
        (1280, 36, 20, 5120),  # large
        (2560, 32, 32, 10240),  # xl
        (4608, 50, 36, 12288),  # 10B
    ]

    # og runs
    # for w in [0, 1, 2]:
    #     for (d, l, h, f) in MODEL_SIZES:
    #         main.remote(10000, 512, d, l, h, f, 10000.0, 4, False, False, w, "cuda", 10)
    #         main.remote(10000, 512, d, l, h, f, 10000.0, 4, True,  False, w, "cuda", 10)
    #         if d != 4608:
    #             main.remote(10000, 512, d, l, h, f, 10000.0, 4, False, True,  w, "cuda", 10)

    # quantized runs
    # for (d, l, h, f) in MODEL_SIZES:
    #         main.remote(10000, 512, d, l, h, f, 10000.0, 4, False, False, w, "cuda", 10, bf16=False)
    #         main.remote(10000, 512, d, l, h, f, 10000.0, 4, True,  False, w, "cuda", 10, bf16=False)

    #         main.remote(10000, 512, d, l, h, f, 10000.0, 4, False, False, w, "cuda", 10, bf16=True)
    #         main.remote(10000, 512, d, l, h, f, 10000.0, 4, True,  False, w, "cuda", 10, bf16=True)

    # compile runs
    for d, l, h, f in MODEL_SIZES:
        main.remote(10000, 512, d, l, h, f, 10000.0, 4, False, False, 5, "cuda", 10, bf16=False, compile_model=False)
        main.remote(10000, 512, d, l, h, f, 10000.0, 4, False, True, 5, "cuda", 10, bf16=False, compile_model=False)
        main.remote(10000, 512, d, l, h, f, 10000.0, 4, False, False, 5, "cuda", 10, bf16=False, compile_model=True)
        main.remote(10000, 512, d, l, h, f, 10000.0, 4, False, True, 5, "cuda", 10, bf16=False, compile_model=True)


if __name__ == "__main__":
    ## this is what will be profiled: target
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", type=int, default=10000)
    parser.add_argument("--c", type=int, default=512)
    parser.add_argument("--d", type=int, default=768)
    parser.add_argument("--l", type=int, default=12)
    parser.add_argument("--h", type=int, default=12)
    parser.add_argument("--f", type=int, default=3072)
    parser.add_argument("--t", type=float, default=10000.0)
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--w", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--step", action="store_true")
    args = parser.parse_args()

    if args.step:
        run_func = forward_backward_step
    elif args.backward:
        run_func = forward_backward
    else:
        run_func = forward

    model, optimizer, input_data, target_data = _get_model_opt_data(
        args.v, args.c, args.d, args.l, args.h, args.f, args.t, args.b, args.device
    )

    for _ in range(args.w):
        run_func(model, input_data, target_data, optimizer)
        torch.cuda.synchronize()

    with nvtx.range("profile_step"):
        run_func(model, input_data, target_data, optimizer)
        torch.cuda.synchronize()

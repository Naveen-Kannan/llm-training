from math import exp
import torch
import json
import os
from llm_training.pretraining.train import Trainer

ROOT = os.environ.get("LLM_TRAINING_ROOT", ".")  # is this right?

data = {
    "ts_train": f"{ROOT}/runs/tokenized/tinystories/train.npy",
    "ts_val": f"{ROOT}/runs/tokenized/tinystories/valid.npy",
    "owt_train": f"{ROOT}/runs/tokenized/openwebtext/train.npy",
    "owt_val": f"{ROOT}/runs/tokenized/openwebtext/valid.npy",
}

CXT = 512

model_config = {
    "vocab_size": 32000,
    "context_length": CXT,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1344,
    "rope_theta": 10000.0,
}  # given pdf

LR = 4e-3
TOT_TOK = 327_680_000 * 10
B_SIZE = 128
ITR = TOT_TOK // (B_SIZE * CXT)

optimizer_config = {
    "lr": LR,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
}  # tune everything

scheduler_config = {
    "lr_min": LR / 10,
    "lr_max": LR,
    "warmup_itr": 2 * ITR // 100,
    "cos_cyc_itr": ITR,
    "grad_clip": 1.0,
}

run_config = {
    "train_data": data["owt_train"],
    "val_data": data["owt_val"],
    "batch_size": B_SIZE,
    "val_batches": 20,
    "itr": ITR,
    "log_itr": ITR // 200,
    "eval_itr": ITR // 50,
    "ckpt_itr": ITR // 10,
}


def _pick_device() -> str:
    # helper to auto choose device - should be obv
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


exp_config = {
    "model_config": model_config,
    "optimizer_config": optimizer_config,
    "scheduler_config": scheduler_config,
    "run_config": run_config,
    "exp_num": f"owt_l2",
    "device": _pick_device(),
}


def run_experiment(exp_config: dict):
    print("Running experiment:", exp_config["exp_num"])
    # idk pdf said
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    exp_path = f"{ROOT}/runs/transformer/exp_{exp_config['exp_num']}"
    os.makedirs(exp_path, exist_ok=True)

    # log config to exp_path
    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(exp_config, f)

    ckpt_path = os.path.join(exp_path, "checkpoint.pt")

    if os.path.exists(ckpt_path):
        trainer = Trainer.from_checkpoint(exp_config)
    else:
        trainer = Trainer(exp_config)

    trainer.train()


if __name__ == "__main__":
    run_experiment(exp_config)

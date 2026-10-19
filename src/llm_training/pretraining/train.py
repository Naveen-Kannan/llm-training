import torch, os, time
from llm_training.pretraining.model import TransformerLM
from llm_training.pretraining.optim import AdamW_Optimizer, cross_entropy, grad_clip, lr_schedule_cosine
from llm_training.pretraining.training_utils import get_batch, load_checkpoint, load_data, save_checkpoint

ROOT = os.environ.get("LLM_TRAINING_ROOT", ".")


def _init_model(model_config: dict):
    return TransformerLM(
        model_config["d_model"],
        model_config["num_heads"],
        model_config["d_ff"],
        model_config["vocab_size"],
        model_config["context_length"],
        model_config["num_layers"],
        model_config["rope_theta"],
    )


def _init_optimizer(model, opt_config: dict):
    return AdamW_Optimizer(
        model.parameters(),
        opt_config["lr"],
        opt_config["weight_decay"],
        opt_config["betas"],
        opt_config["eps"],
    )


class Trainer:
    def __init__(self, exp_config: dict, itr: int = 0):
        self.exp_num = exp_config["exp_num"]
        self.device = exp_config["device"]
        self.run_config = exp_config["run_config"]
        self.scheduler_config = exp_config["scheduler_config"]
        self.itr = itr

        # Load data
        train_data_path = exp_config["run_config"]["train_data"]
        self.train_data = load_data(train_data_path)

        val_data_path = exp_config["run_config"]["val_data"]
        self.val_data = load_data(val_data_path)

        # create model
        self.model_config = exp_config["model_config"]
        self.model = _init_model(self.model_config).to(self.device)
        self.model = torch.compile(self.model)

        # create optimizer
        opt_config = exp_config["optimizer_config"]
        self.optimizer = _init_optimizer(self.model, opt_config)

        # start wall clock
        self.t0 = time.time()

    @classmethod
    def from_checkpoint(cls, exp_config):
        exp_path = f"{ROOT}/runs/transformer/exp_{exp_config['exp_num']}"
        ckpt_path = os.path.join(exp_path, "checkpoint.pt")

        trainer = cls(exp_config)
        itr = load_checkpoint(ckpt_path, trainer.model, trainer.optimizer)
        trainer.itr = itr

        return trainer

    def train(self):
        while self.itr < self.run_config["itr"]:
            self.train_itr()

    def train_itr(self):
        # get scheduled lr
        lr = lr_schedule_cosine(
            self.itr,
            self.scheduler_config["lr_max"],
            self.scheduler_config["lr_min"],
            self.scheduler_config["warmup_itr"],
            self.scheduler_config["cos_cyc_itr"],
        )

        # feels goofy...
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        # zero grad
        self.optimizer.zero_grad()

        # massive speedup if using bfloat16 on gpu plz
        use_bfloat = self.device == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat):
            # forward
            out, y = self._forward_pass()

            # get loss
            loss = cross_entropy(out, y)

        loss.backward()

        grad_clip(self.model.parameters(), self.scheduler_config["grad_clip"])

        self.optimizer.step()

        self.itr += 1

        # logging
        self._log_itr(loss)

    def _forward_pass(self):
        x, y = get_batch(
            self.train_data, self.run_config["batch_size"], self.model_config["context_length"], self.device
        )
        return self.model(x), y

    def _log_itr(self, loss):
        # elapsed
        t = time.time() - self.t0

        exp_path = f"{ROOT}/runs/transformer/exp_{self.exp_num}"
        # log train loss to console/file
        if self.itr % self.run_config["log_itr"] == 0:
            log = f"t {t:.1f}s | iter {self.itr}: train loss {loss.item():.4f}"
            print(log)
            with open(os.path.join(exp_path, "log.txt"), "a") as f:
                f.write(log + "\n")

        # compute val
        if self.val_data is not None:
            if self.itr % self.run_config["eval_itr"] == 0:
                val_loss = self._val_loss()
                log = f"t {t:.1f}s | iter {self.itr}: val loss {val_loss:.4f}"
                print(log)
                with open(os.path.join(exp_path, "log.txt"), "a") as f:
                    f.write(log + "\n")

        # ckpt save
        if self.itr % self.run_config["ckpt_itr"] == 0:
            save_checkpoint(self.model, self.optimizer, self.itr, os.path.join(exp_path, "checkpoint.pt"))

    @torch.no_grad()  # forgot
    def _val_loss(self):
        loss = list([])

        for _ in range(self.run_config["val_batches"]):
            x, y = get_batch(
                self.val_data, self.run_config["batch_size"], self.model_config["context_length"], self.device
            )
            out = self.model(x)

            loss.append(cross_entropy(out, y).item())

        return sum(loss) / len(loss)

import torch
import math


# @torch.compile
def cross_entropy(predicted_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # last dim is vocab
    # sub the max for stability
    max_logit, _ = predicted_logits.max(dim=-1, keepdim=True)
    stable_logits = predicted_logits - max_logit
    log_tot = torch.log(torch.sum(torch.exp(stable_logits), dim=-1))
    target_logit = stable_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_tot - target_logit).mean()  # is negative


def perplexity(predicted_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(predicted_logits, targets))


class SGD_Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW_Optimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, betas, eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            b1, b2 = betas

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                # init m, v to 0, same shape
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                grad = p.grad.data

                # lr update
                lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)

                # weight decay
                p.data -= lr * wd * p.data

                # moment update
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad**2

                # param update
                p.data -= lr_t * m / (v.sqrt() + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def lr_schedule_cosine(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int):
    if t < t_w:
        return t * lr_max / t_w
    elif t <= t_c:
        sched = 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi))
        return lr_min + sched * (lr_max - lr_min)
    else:
        return lr_min


# @torch.compile
def grad_clip(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]

    # trick: sqrt(sum(sq(sqrt(sum(sq(grad)))))) = sqrt(sum(sq(grad)))
    tot_norm = torch.stack([g.norm(2) for g in grads]).norm(2)

    scale = max_norm / (tot_norm + 1e-6)
    if scale < 1:  # dont have to check again this way
        for g in grads:
            g.mul_(scale)


if __name__ == "__main__":
    # mini test
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD_Optimizer([weights], lr=1e3)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

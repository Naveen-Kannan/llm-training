import torch, torch.distributed as dist

# replace linear and embeds entirely
from llm_training.reference.model import Linear, Embedding
from einops import einsum, rearrange

PREFETCH_DEPTH = 3  # modules to look ahead/behind - 3 fixes swiglu goofiness (pray ts fits)


class fsdp_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, fsdp, mod):
        completed = fsdp._get_orig_shape(weight)
        # rkeep same dtype for rope
        if x.dtype != completed.dtype:
            x = x.to(completed.dtype)

        ctx.fsdp, ctx.mod, ctx.weight = fsdp, mod, weight  # store ref
        ctx.save_for_backward(x)
        if not fsdp.recomputing:
            fsdp.prefetch(mod, +PREFETCH_DEPTH)
        return einsum(x, completed, "... d_in, d_out d_in -> ... d_out")

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        fsdp, mod, weight = ctx.fsdp, ctx.mod, ctx.weight

        completed = fsdp._get_orig_shape(weight)
        fsdp.prefetch(mod, -PREFETCH_DEPTH, only_linear=True)

        if grad_out.dtype != completed.dtype:
            grad_out = grad_out.to(completed.dtype)

        d_x = einsum(grad_out, completed, "... d_out, d_out d_in -> ... d_in")
        d_w = einsum(grad_out, x, "... d_out, ... d_in -> d_out d_in")
        return d_x, fsdp.scatter_grad(weight, d_w), None, None


class fsdp_embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, weight, fsdp, mod):
        completed = fsdp._get_orig_shape(weight)
        ctx.fsdp, ctx.mod, ctx.weight = fsdp, mod, weight
        ctx.save_for_backward(token_ids)
        if not fsdp.recomputing:
            fsdp.prefetch(mod, +PREFETCH_DEPTH)
        return completed[token_ids, :]

    @staticmethod
    def backward(ctx, grad_out):
        (token_ids,) = ctx.saved_tensors
        fsdp, weight = ctx.fsdp, ctx.weight
        fsdp.prefetch(ctx.mod, -PREFETCH_DEPTH, only_linear=True)

        d_w = torch.zeros(fsdp.shapes[weight], dtype=torch.float32, device=grad_out.device)
        d_w.index_add_(
            0, rearrange(token_ids, "... -> (...)"), rearrange(grad_out, "... d -> (...) d").to(torch.float32)
        )
        return None, fsdp.scatter_grad(weight, d_w), None, None


class FSDP(torch.nn.Module):
    MAX_PENDING_GRADS = 2  # bound mem

    def __init__(self, module: torch.nn.Module, compute_dtype=None):
        super().__init__()
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.module = module
        self.compute_dtype = compute_dtype
        self.recomputing = False
        # ISSUE
        # we prefetch recursevly in modules; if checkpointing, we will fetch in 'forward' again, which will go the wrong way.
        # wasted mem. need this flag

        self.shapes = dict()  # store module shapes
        self.fsdp_modules = list()
        self.mod_to_idx = dict()
        self.sharded_param_ids = set()  # need ids in particular for stability. n
        # sharded ids to keep track of when to all_gather

        self.gather_state = dict()  # param -> (handle, full_buf, rank_buf)
        self.pending_shards = list()  # (handle, param, out_shard, src_buf)
        self.replicated_handles = list()  # all_reduces for replicated grads

        for p in module.parameters():
            dist.broadcast(p.data, src=0)

        # Linear and Embedding weights get sharded, norms chill
        for mod in module.modules():
            if isinstance(mod, (Linear, Embedding)):
                self._shard(mod)

        # other params still need to sync grads
        for p in module.parameters():
            if id(p) not in self.sharded_param_ids and p.requires_grad:
                p.register_post_accumulate_grad_hook(self._general_grad_hook)

    def _shard(self, mod):
        p = mod.weight
        # save idx for later -> prefetching :)
        self.mod_to_idx[mod] = len(self.fsdp_modules)
        self.fsdp_modules.append(mod)
        self.sharded_param_ids.add(id(p))  # ts id() dope
        self.shapes[p] = tuple(p.shape)

        shard_n = p.numel() // self.world_size
        start = self.rank * shard_n

        # our shard
        p.data = p.data.view(-1)[start : start + shard_n].clone()

        if isinstance(mod, Linear):
            mod.forward = lambda x, _m=mod: fsdp_linear.apply(x, _m.weight, self, _m)
        else:  # embedding
            mod.forward = lambda ids, _m=mod: fsdp_embedding.apply(ids, _m.weight, self, _m)

    def forward(self, *inputs, **kwargs):
        self.recomputing = False
        # warm the pipeline for forward
        for mod in self.fsdp_modules[:PREFETCH_DEPTH]:
            self._gather_start(mod.weight)

        out = self.module(*inputs, **kwargs)

        # warm the pipeline for backward (last few Linears)
        for mod in self.fsdp_modules[-PREFETCH_DEPTH:]:
            if isinstance(mod, Linear):
                self._gather_start(mod.weight)

        self.recomputing = True  # any later Func.forward is a checkpoint recompute
        return out

    def finish_gradient_synchronization(self):
        for h in self.replicated_handles:
            h.wait()
        self.replicated_handles.clear()
        self._drain_pending(0)
        # only on fwd only
        for p in list(self.gather_state):
            self._get_orig_shape(p)

    # all_gather weights

    def prefetch(self, mod, offset, only_linear=False):
        idx = self.mod_to_idx[mod] + offset
        if not (0 <= idx < len(self.fsdp_modules)):
            return
        target = self.fsdp_modules[idx]
        if only_linear and not isinstance(target, Linear):
            return
        self._gather_start(target.weight)

    def _get_orig_shape(self, param):
        # wait and free gather buffers
        if param not in self.gather_state:  # safety
            self._gather_start(param)

        handle, completed, _ = self.gather_state.pop(param)
        handle.wait()
        return completed.view(self.shapes[param])

    def _gather_start(self, param, cast=True):
        if param in self.gather_state:
            return  # alr gathered

        rank_buf = param.data
        if cast and self.compute_dtype is not None:
            rank_buf = rank_buf.to(self.compute_dtype)

        completed = torch.empty(param.numel() * self.world_size, dtype=rank_buf.dtype, device=rank_buf.device)
        handle = dist.all_gather_into_tensor(completed, rank_buf, async_op=True)
        self.gather_state[param] = (handle, completed, rank_buf)

    # reduce_scatter grads

    def scatter_grad(self, param, grad):
        flat = rearrange(grad.contiguous(), "... -> (...)").to(torch.float32) / self.world_size

        out = torch.empty(param.numel(), dtype=flat.dtype, device=flat.device)
        handle = dist.reduce_scatter_tensor(out, flat, op=dist.ReduceOp.SUM, async_op=True)

        self.pending_shards.append((handle, param, out, flat))
        self._drain_pending(self.MAX_PENDING_GRADS)
        return None  # autograd skips the weight; we set param.grad in _drain_pending

    def _drain_pending(self, keep):
        while len(self.pending_shards) > keep:
            handle, param, out_shard, _ = self.pending_shards.pop(0)
            handle.wait()
            shard = out_shard.to(param.data.dtype)
            param.grad = shard if param.grad is None else param.grad + shard

    # DDP copy
    def _general_grad_hook(self, param):
        param.grad /= self.world_size
        self.replicated_handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True))

    # test helper

    def _gather_full_params(self):
        out = {}
        for name, p in self.module.named_parameters():
            if id(p) in self.sharded_param_ids:
                self._gather_start(p, cast=False)  # gather in master fp32 for verification
                out[name] = self._get_orig_shape(p).clone()
            else:
                out[name] = p.data.clone()
        return out

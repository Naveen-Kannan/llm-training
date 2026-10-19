import torch
from torch.optim import Optimizer
from typing import Any
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs):
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()

        params = list(params)
        if params and isinstance(params[0], dict):  # then ew have param groups
            param_groups = params
        else:
            param_groups = [{"params": params}]

        self.optim_init = False

        self.sharded_param_groups = list()
        self.param_to_rank = dict()

        super().__init__(param_groups, {})

        # itr = 0
        # for group in param_groups:
        #     group_shard = dict(group)
        #     group_shard["params"] = list()

        #     # probs should do list comp
        #     for param in group["params"]:
        #         self.param_to_rank[param] = itr % self.world_size
        #         if itr % self.world_size == self.rank:
        #             group_shard["params"].append(param)
        #         itr += 1

        #     self.sharded_param_groups.append(group_shard)

        self.optimizer = optimizer_cls(self.sharded_param_groups, **kwargs)  # kwargs is just lr, weight decay, etc.
        self.optim_init = True

    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)

        # async maybe?
        handles = list()
        for param_group in self.param_groups:
            for param in param_group["params"]:
                param_rank = self.param_to_rank[param]
                handles.append(dist.broadcast(param.data, src=param_rank, async_op=True))
        for handle in handles:
            handle.wait()

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)  # super class constructuor thing

        rank_shard = dict(param_group)
        rank_shard["params"] = list()

        curr_param_count = sum(len(group["params"]) for group in self.param_groups[:-1])
        itr = curr_param_count
        for param in param_group["params"]:
            self.param_to_rank[param] = itr % self.world_size
            if itr % self.world_size == self.rank:
                rank_shard["params"].append(param)
            itr += 1

        if len(rank_shard["params"]) > 0:
            if self.optim_init:
                self.optimizer.add_param_group(rank_shard)
            else:
                self.sharded_param_groups.append(rank_shard)

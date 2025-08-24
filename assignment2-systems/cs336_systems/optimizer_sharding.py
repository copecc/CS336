import torch
import torch.distributed as dist

from typing import Iterable, Type


class OptimizerSharding(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs
    ):

        self.world_size = dist.get_world_size()
        self.params = list(params)

        rank = dist.get_rank()
        params_shard = [{"params": [param for i, param in enumerate(self.params) if i % self.world_size == rank]}]
        self.optimizer = optimizer_cls(params_shard, **kwargs)
        super().__init__(params_shard, {})

    def step(self, closure=None, **kwargs):
        # Update parameters
        self.optimizer.step(closure, **kwargs)
        # Synchronize parameters across processes
        handles = []
        for i, param in enumerate(self.params):
            rank = i % self.world_size
            handles.append(dist.broadcast(param.data, src=rank, async_op=True))

        for handle in handles:
            handle.wait()

    def add_param_group(self, param_group):
        super().add_param_group(param_group)

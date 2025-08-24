import torch
import torch.distributed as dist


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        # broadcast parameters from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if not param.requires_grad:
                continue
            # Register hooks for gradient synchronization
            if hasattr(param, "register_post_accumulate_grad_hook"):
                param.register_post_accumulate_grad_hook(self.allreduce_param_grad)
            else:  # This way register hook on gradient itself but not on parameter
                param.register_hook(self.allreduce_grad)

    def allreduce_grad(self, grad: torch.Tensor):
        with torch.no_grad():
            grad.data /= dist.get_world_size()

        handle = dist.all_reduce(grad.data, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

    def allreduce_param_grad(self, param: torch.nn.Parameter):
        with torch.no_grad():
            param.grad.data /= dist.get_world_size()

        handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class Bucket:
    def __init__(self, results: list):
        """
        Initialize a bucket for gradient accumulation. Param is bind to the bucket via hook on method `on_param_grad_ready`.

        Args:
            results (list): The list to store the results of the all-reduce operations, shared across buckets.
        """
        self.params_count = 0  # Record the number of parameters in the bucket
        self.ready_params = []  # Record parameters finished computing gradients
        self.results = results  # Store results of all-reduce operations, shared across buckets.

    def register_one_param(self):
        """
        Record the number of parameters should be in the bucket.
        """
        self.params_count += 1

    def on_param_grad_ready(self, param: torch.nn.Parameter):
        """
        Triggered when a parameter's gradient is ready.
        """
        self.ready_params.append(param)
        # If all parameters are ready, initiate all-reduce
        if len(self.ready_params) == self.params_count:
            grads = torch._utils._flatten_dense_tensors([p.grad for p in self.ready_params])
            grads /= dist.get_world_size()
            handle = dist.all_reduce(grads, op=dist.ReduceOp.SUM, async_op=True)
            # Append the bucket to the results for later processing
            self.results.append((handle, self.ready_params, grads))

            # Reset the ready parameters list
            self.ready_params = []


class DDPBucketedParameters(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, bucket_size_mb: int):
        super().__init__()
        self.module = module

        self.results = []

        bucket_size_mb = bucket_size_mb * 1024 * 1024
        bucket = Bucket(self.results)
        bucket_size = 0

        for param in reversed(list(self.module.parameters())):
            dist.broadcast(param.data, src=0)
            if not param.requires_grad:
                continue

            # If the bucket is full, make a new one for the following parameters
            if bucket_size + param.data.nbytes > bucket_size_mb:
                bucket = Bucket(self.results)
                bucket_size = 0

            bucket_size += param.data.nbytes
            # Record that this parameter should be in the current bucket
            bucket.register_one_param()
            # Register the parameter's gradient hook on the bucket
            param.register_post_accumulate_grad_hook(bucket.on_param_grad_ready)

    def finish_gradient_synchronization(self):
        for handle, params, grads in self.results:
            handle.wait()
            grads = torch._utils._unflatten_dense_tensors(grads, params)
            for param, grad in zip(params, grads):
                param.grad = grad

        self.results.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

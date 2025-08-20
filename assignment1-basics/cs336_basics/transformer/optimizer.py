import torch
import math

from collections.abc import Callable, Iterable
from typing import Optional


class SGD(torch.optim.Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Args:
        params (Iterable[torch.nn.Parameter]): The parameters to optimize.
        lr (float, optional): The learning rate. Defaults to 1e-3.
    """

    def __init__(self, params, lr=1e-3):
        """
        Initialize the SGD optimizer.
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    """
    AdamW (Adam with Weight Decay) optimizer.

    AdamW is an improvement over Adam that decouples weight decay from gradient-based update.
    This leads to better generalization and is particularly effective for training transformers.

    Key differences from Adam:
    1. Weight decay is applied directly to parameters (not added to gradients)
    2. Better bias correction handling
    3. More stable training for large models

    Mathematical formulation:
        m_t = β₁·m_{t-1} + (1 - β₁)·g_t        # First moment (momentum)
        v_t = β₂·v_{t-1} + (1 - β₂)·g_t²       # Second moment (adaptive lr)
        lr_t = lr·√(1 - β₂^t) / (1 - β₁^t)     # Learning rate adjustment
        θ_t = θ_{t-1} - lr_t·m̂_t / (√v_t + ε)  # Parameter update
        θ_t = θ_t - lr·λ·θ_t                   # Weight decay (decoupled)

    Args:
        params (Iterable[torch.nn.Parameter]): The parameters to optimize.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        betas (Tuple[float, float], optional): Coefficients for computing
            running averages of gradient and its square. Defaults to (0.9, 0.999).
            - beta1: Momentum coefficient for first moment (typically 0.9)
            - beta2: Momentum coefficient for second moment (typically 0.999)
        eps (float, optional): Term added to denominator for numerical stability. Defaults to 1e-8.
        weight_decay (float, optional): Weight decay coefficient (L2 regularization). Defaults to 1e-2.

    Effects and Benefits:
        - Faster convergence than SGD due to adaptive learning rates
        - Better generalization than Adam due to proper weight decay
        - Stable training for large models (transformers, GPT, BERT, etc.)
        - Automatic learning rate adaptation per parameter
        - Momentum helps escape local minima and smooth optimization
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        Initialize the AdamW optimizer.
        """
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.

                # Get iteration number, start at 1
                t = state.get("t", 1)
                # Get the first moment (mean) and second moment (variance)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                # Get the gradient of loss with respect to p.
                grad = p.grad.data

                # First moment: exponentially weighted average of gradients
                # This provides momentum and helps navigate noisy gradients
                m = beta1 * m + (1 - beta1) * grad
                # Second moment: exponentially weighted average of squared gradients
                # This enables adaptive learning rates per parameter
                v = beta2 * v + (1 - beta2) * grad.pow(2)
                # # Correct for initialization bias (moments start at zero). This automatically handles the "warm-up" phase of training
                # Without correction, early steps would have very small updates
                adjusted_lr = lr * math.sqrt(1 - beta2 ** (t)) / (1 - beta1 ** (t))

                p.data -= adjusted_lr * m / (v.sqrt() + eps)
                # AdamW key innovation: apply weight decay directly to parameters, pull the parameters towards 0
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return loss


def lr_cosine_schedule(
    t: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_annealing_iters: int
) -> float:
    """
    Compute the learning rate for a given step using a cosine schedule with warmup.

    Args:
        t (int): Current step.
        max_learning_rate (float): Maximum learning rate.
        min_learning_rate (float): Minimum learning rate.
        warmup_iters (int): Number of warmup iterations.
        cosine_annealing_iters (int): Number of cosine annealing iterations.

    Returns:
        float: Learning rate for the current step.
    """
    # Phase 1: Linear warmup
    if t < warmup_iters:
        return t / warmup_iters * max_learning_rate

    # Phase 3: Stay at minimum after annealing
    elif t >= cosine_annealing_iters:
        return min_learning_rate

    # Phase 2: Cosine annealing
    # Calculate progress through annealing phase (0 to 1)
    progress = (t - warmup_iters) / (cosine_annealing_iters - warmup_iters)
    # Cosine decay factor: starts at 1, ends at 0
    cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
    # Final learning rate
    return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    """
    Apply gradient clipping to the given parameters.
    """
    l2_norm = math.sqrt(sum(torch.sum(p.grad.data**2).item() for p in parameters if p.grad is not None))
    if l2_norm < max_l2_norm:
        return

    # Clip gradients if L2 norm exceeds threshold
    clip_coef = max_l2_norm / (l2_norm + eps)
    for p in parameters:
        if p.grad is not None:
            p.grad.data *= clip_coef

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of input and target sequences from the dataset.

    Optimized version that avoids slow tensor creation from list of numpy arrays.
    """
    # Randomly sample a batch of input sequences and their corresponding targets
    idx = np.random.choice(len(dataset) - context_length, batch_size)
    # Create input and target indices
    input_indices = idx[:, None] + np.arange(context_length)
    target_indices = idx[:, None] + np.arange(1, context_length + 1)
    # Create numpy arrays first, then convert to tensors
    x_np = dataset[input_indices]
    y_np = dataset[target_indices]
    # Create tensors from numpy arrays, using astype instead of long is for compatibility on torch1.x
    x = torch.from_numpy(x_np.astype(np.int64))
    y = torch.from_numpy(y_np.astype(np.int64))
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

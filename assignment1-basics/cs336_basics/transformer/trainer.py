import argparse
import os
import time
import typing
import torch
import wandb

import numpy as np
import numpy.typing as npt

from loguru import logger
import yaml

from cs336_basics.transformer.data import get_batch
from cs336_basics.transformer.model import TransformerLM
from cs336_basics.transformer.nn_utils import cross_entropy, load_checkpoint, save_checkpoint
from cs336_basics.transformer.optimizer import (
    AdamW,
    gradient_clipping,
    lr_cosine_schedule,
)


def evaluate(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 16,
) -> float:
    """
    Evaluate the model on validation data.

    Args:
        model: Model to evaluate
        dataset: Validation dataset
        batch_size: Batch size for evaluation
        context_length: Context length
        device: Device to run evaluation on
        num_batches: Number of batches to evaluate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()

    model.train()
    return total_loss / max(num_batches, 1)


def evaluate_full_validset(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_size = max(batch_size, 32)
    with torch.no_grad():
        num_batches = len(dataset) // (batch_size * context_length)
        for i in range(num_batches):
            start = i * batch_size * context_length
            end = start + batch_size * context_length
            batch = dataset[start:end]
            batch = batch.reshape(batch_size, context_length)
            x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
            y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()
            total_tokens += 1

    model.train()
    return total_loss / max(total_tokens, 1)


def train(config: dict):
    stop_training = False

    def handle_interrupt(signum, frame):
        nonlocal stop_training
        logger.warning("Received interrupt signal. Will stop after this iteration.")
        stop_training = True

    import signal

    signal.signal(signal.SIGINT, handle_interrupt)

    # Initialize wandb
    wandb.init(project=config["project_name"], name=config["experiment_name"], config=config)
    config = dict(wandb.config)  # Use wandb.config to override original config

    logger.info(f"Training using config: {config}")

    # model config
    vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta = (
        config[k] for k in ["vocab_size", "context_length", "num_layers", "d_model", "num_heads", "d_ff", "theta"]
    )

    device = config["device"]
    dtype = getattr(torch, config["dtype"]) if config["dtype"] is not None else torch.float32

    # optimizer config
    lr, beta1, beta2, eps, weight_decay = (config[k] for k in ["lr", "beta1", "beta2", "eps", "weight_decay"])
    min_lr, warmup_steps, annealing_steps = (config[k] for k in ["min_lr", "warmup_steps", "annealing_steps"])

    # training config
    batch_size, num_epochs, total_steps, max_grad_norm = (
        config[k] for k in ["batch_size", "num_epochs", "total_steps", "max_grad_norm"]
    )
    valid_interval, valid_num_batches = (config[k] for k in ["valid_interval", "valid_num_batches"])
    save_interval, log_interval = (config[k] for k in ["save_interval", "log_interval"])
    desired_tokens, desired_loss = (config[k] for k in ["desired_tokens", "desired_loss"])
    training_minutes = config["training_minutes"]

    # data config
    train_data_path, valid_data_path = (config[k] for k in ["train_data", "valid_data"])
    vocab_path, merge_path, special_tokens = (config[k] for k in ["vocab_path", "merge_path", "special_tokens"])

    # checkpoint config
    ckpt_path, ckpt_dir = (config[k] for k in ["ckpt_path", "ckpt_dir"])

    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device, dtype)

    if hasattr(torch, "compile"):  # Check if torch.compile is available
        if device == "mps":
            model = torch.compile(model, backend="aot_eager")
        else:
            model = torch.compile(model)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    # Load training data and validation data
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(valid_data_path, dtype=np.uint16, mode="r")
    logger.info(
        f"Loaded training data from {train_data_path}({len(train_data):,} tokens) and validation data from {valid_data_path}({len(valid_data):,} tokens)"
    )

    # Precompute token statistics and calculate total iterations
    tokens_per_batch = batch_size * context_length
    steps_per_epoch = len(train_data) // tokens_per_batch
    num_iterations = steps_per_epoch * num_epochs
    if batch_size * num_iterations * context_length < desired_tokens:
        logger.warning(
            f"Total tokens ({batch_size * num_iterations * context_length}) is less than the configured total tokens ({desired_tokens}). Adjusting num_iterations."
        )

    start_iteration = 0
    # Load checkpoint if provided
    if ckpt_path:
        start_iteration = load_checkpoint(ckpt_path, model, optimizer) + 1

    if total_steps:
        num_iterations = min(num_iterations, start_iteration + total_steps)
    # Create checkpoint save directory if it doesn't exist
    ckpt_dir = ckpt_dir or "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.watch(model, log="all", log_freq=log_interval)
    model.train()

    logger.info(f"Starting training...Start at iteration {start_iteration}, end at iteration {num_iterations}")
    start_time = time.time()

    for iteration in range(start_iteration, num_iterations):
        # Get batch
        x, y = get_batch(train_data, batch_size, context_length, device)

        # Forward pass
        optimizer.zero_grad()

        # Update lr
        updated_lr = lr_cosine_schedule(iteration, lr, min_lr, warmup_steps, annealing_steps)
        # update learning rate for each parameter group
        for param_group in optimizer.param_groups:
            param_group["lr"] = updated_lr

        logits = model(x)
        loss = cross_entropy(logits, y)
        if torch.isnan(loss):
            logger.error(f"Loss is NaN at iteration {iteration}, stopping training.")
            stop_training = True

        # Backward pass
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), max_grad_norm)
        optimizer.step()

        # Validate model
        if iteration % valid_interval == 0:
            val_loss = evaluate(model, valid_data, batch_size, context_length, device, valid_num_batches)
            elapsed = time.time() - start_time
            wandb.log({"val_loss": val_loss, "iteration": iteration, "wallclock_time": elapsed})
            logger.info(f"Validation Loss: {val_loss}, Iteration: {iteration}, Wallclock Time: {elapsed:.2f}s")

            # 1.45 for cuda and 2.0 for mps
            if val_loss < desired_loss:
                logger.info(
                    f"Validation loss {val_loss} is below desired loss {desired_loss}. Now validate the full validation set."
                )
                full_val_loss = evaluate_full_validset(model, valid_data, batch_size, context_length, device)
                logger.info(f"Full Validation Loss: {full_val_loss}")
                if full_val_loss < desired_loss:
                    logger.info(
                        f"Full validation loss {full_val_loss} is below desired loss {desired_loss}. Stopping training."
                    )
                    break

        # Log metrics
        if iteration % log_interval == 0:
            elapsed = time.time() - start_time
            wandb.log(
                {
                    "iteration": iteration,
                    "train_loss": loss.item(),
                    "learning_rate": updated_lr,
                    "wallclock_time": elapsed,
                }
            )

            logger.info(f"Iteration {iteration}, Loss: {loss.item()}, Learning Rate: {updated_lr}")

            if training_minutes and (elapsed / 60) >= training_minutes:
                logger.warning(f"Reached training time limit of {training_minutes} minutes.")
                stop_training = True

        # Save checkpoint
        if iteration > start_iteration and iteration % save_interval == 0:
            checkpoint_path = f"{ckpt_dir}/checkpoint_{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            logger.info(f"Saved checkpoint at {checkpoint_path}")

        if stop_training:
            logger.warning("Stopping training...")
            break

    # Full validation loss
    val_loss = evaluate_full_validset(model, valid_data, batch_size, context_length, device)
    elapsed = time.time() - start_time
    wandb.log({"val_loss": val_loss, "iteration": iteration, "wallclock_time": elapsed})
    logger.info(f"Validation Loss: {val_loss}, Iteration: {iteration}, Wallclock Time: {elapsed:.2f}s")

    if not stop_training:
        checkpoint_path = f"{ckpt_dir}/checkpoint_final.pt"
        save_checkpoint(model, optimizer, iteration, checkpoint_path)
        logger.info(f"Training complete. Final checkpoint saved at {checkpoint_path}")

    # Finish wandb run
    wandb.finish()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")

    # Config file
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Model config
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, help="Context length")
    parser.add_argument("--num_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, help="Feed-forward dimension")
    parser.add_argument("--theta", type=float, help="RoPE theta parameter")

    # Device/dtype
    parser.add_argument("--device", type=str, help="Device (cuda/cpu/mps/auto)")
    parser.add_argument("--dtype", type=str, help="Data type (float32/float16/bfloat16)")

    # Optimizer config
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--beta1", type=float, help="Adam beta1")
    parser.add_argument("--beta2", type=float, help="Adam beta2")
    parser.add_argument("--eps", type=float, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")

    # Scheduler config
    parser.add_argument("--min_lr", type=float, help="Final learning rate")
    parser.add_argument("--warmup_steps", type=int, help="Warmup steps for learning rate")
    parser.add_argument("--annealing_steps", type=int, help="Total annealing steps")

    # Training config
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--total_steps", type=int, help="Total number of training iterations")
    parser.add_argument("--max_grad_norm", type=float, help="Gradient clipping threshold")
    parser.add_argument("--valid_interval", type=int, help="Validation interval")
    parser.add_argument("--valid_num_batches", type=int, help="Number of validation batches")
    parser.add_argument("--save_interval", type=int, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, help="Logging interval")
    parser.add_argument("--desired_loss", type=float, help="Desired loss for early stopping")
    parser.add_argument("--desired_tokens", type=int, help="Total number of tokens")
    parser.add_argument("--training_minutes", type=int, help="Total training time in minutes")

    # Data config
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--valid_data", type=str, help="Path to validation data")
    parser.add_argument("--vocab_path", type=str, help="Path to vocabulary file")
    parser.add_argument("--merge_path", type=str, help="Path to merge file")
    parser.add_argument("--special_tokens", type=str, nargs="+", help="Special tokens")

    # Checkpoint path
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--ckpt_dir", type=str, help="Directory to save checkpoints")

    # Wandb config
    parser.add_argument("--project_name", type=str, help="Wandb project name")
    parser.add_argument("--experiment_name", type=str, help="Wandb experiment name")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")

    return parser.parse_args()


def flatten_config(config):
    new_config = {}
    for k, v in config.items():
        if isinstance(v, dict):  # Recursively flatten the dictionary
            nested_config = flatten_config(v)
            for nk, nv in nested_config.items():
                new_config[f"{nk}"] = nv
        else:
            new_config[k] = v
    return new_config


def main():
    # Set up configuration
    args = parse_args()
    config_path = args.config

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config = flatten_config(config)
        # Update config with command line arguments
        config.update({k: v for k, v in vars(args).items() if v is not None})

        train(config)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()

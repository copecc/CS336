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

    model_config = config["model"]
    vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta = (
        model_config[k] for k in ["vocab_size", "context_length", "num_layers", "d_model", "num_heads", "d_ff", "theta"]
    )

    device = config["device"]
    dtype = getattr(torch, config["dtype"]) if config["dtype"] is not None else torch.float32

    optimizer_config = config["optimizer"]
    lr, beta1, beta2, eps, weight_decay = (optimizer_config[k] for k in ["lr", "beta1", "beta2", "eps", "weight_decay"])

    scheduler_config = config["scheduler"]
    min_lr, warmup_steps, annealing_steps = (scheduler_config[k] for k in ["min_lr", "warmup_steps", "annealing_steps"])

    training_config = config["training"]
    batch_size, num_epochs, total_steps, max_grad_norm = (
        training_config[k] for k in ["batch_size", "num_epochs", "total_steps", "max_grad_norm"]
    )
    valid_interval, valid_num_batches = (training_config[k] for k in ["valid_interval", "valid_num_batches"])
    save_interval, log_interval = (training_config[k] for k in ["save_interval", "log_interval"])
    desired_tokens, desired_loss = (training_config[k] for k in ["desired_tokens", "desired_loss"])

    data_config = config["data"]
    train_data_path, valid_data_path = (data_config[k] for k in ["train_data", "valid_data"])
    vocab_path, merge_path, special_tokens = (data_config[k] for k in ["vocab_path", "merge_path", "special_tokens"])

    checkpoint_config = config["checkpoint"]
    ckpt_path, ckpt_dir = (checkpoint_config[k] for k in ["ckpt_path", "ckpt_dir"])

    wandb_config = config["wandb"]

    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta, device, dtype)
    if device == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    # Initialize wandb
    wandb.init(project=wandb_config["project_name"], name=wandb_config["experiment_name"], config=config)
    wandb.watch(model, log="all", log_freq=log_interval)

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
    if total_steps:
        num_iterations = min(num_iterations, total_steps)

    start_iteration = 0
    # Load checkpoint if provided
    if ckpt_path:
        start_iteration = load_checkpoint(ckpt_path, model, optimizer) + 1
    # Create checkpoint save directory if it doesn't exist
    ckpt_dir = ckpt_dir or "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)

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
            wandb.log({"iteration": iteration, "train_loss": loss.item(), "learning_rate": updated_lr})

            logger.info(f"Iteration {iteration}, Loss: {loss.item()}, Learning Rate: {updated_lr}")

        # Save checkpoint
        if iteration > start_iteration and iteration % save_interval == 0:
            checkpoint_path = f"{ckpt_dir}/checkpoint_{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            logger.info(f"Saved checkpoint at {checkpoint_path}")

        if stop_training:
            logger.warning("Stopping training...")
            break

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

    # Checkpoint path
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint to resume training from")

    return parser.parse_args()


def main():
    # Set up configuration
    args = parse_args()
    config_path = args.config

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config: {config}")
        train(config)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()

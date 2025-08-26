import argparse
import contextlib
import numpy as np
import pandas as pd
import torch
import timeit
import cs336_basics

from collections import namedtuple
from loguru import logger

from cs336_basics.model import BasicsTransformerLM as TransformerLM
from cs336_basics.optimizer import AdamW
from annotated_scaled_dot_product_attention import annotated_scaled_dot_product_attention


def benchmarking(config: dict) -> None:

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        logger.error("CUDA and MPS are not available. Please check your installation.")
        return

    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.mps.is_available():
            torch.mps.synchronize()

    batch_size: int = config["batch_size"]
    warmup_steps: int = config["warmup_steps"]
    execution_steps: int = config["execution_steps"]

    # model config
    vocab_size: int = config["vocab_size"]
    context_length: int = config["context_length"]
    d_model: int = config["d_model"]
    num_layers: int = config["num_layers"]
    num_heads: int = config["num_heads"]
    d_ff: int = config["d_ff"]
    rope_theta: float = config["rope_theta"]

    use_annotated: bool = config.get("use_annotated", False)
    use_bf16 = config.get("use_bf16", False)
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    profile_memory = config.get("profile_memory", False)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model = model.to(device)

    # create dummy input and target
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)

    # warmup
    for _ in range(warmup_steps):
        with autocast_ctx:
            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()

    forward_times = []
    backward_times = []

    # benchmark
    if use_annotated:
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    if profile_memory:  # Start recording memory history.
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(execution_steps):
        synchronize()
        start_forward = timeit.default_timer()
        with autocast_ctx:
            outputs = model(input_ids)
        synchronize()
        end_forward = timeit.default_timer()
        forward_times.append(end_forward - start_forward)
        with autocast_ctx:
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
        start_backward = timeit.default_timer()
        loss.backward()
        synchronize()
        end_backward = timeit.default_timer()
        backward_times.append(end_backward - start_backward)

    if profile_memory:  # Stop recording memory history.
        # Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        # Stop recording history.
        torch.cuda.memory._record_memory_history(enabled=None)

    logger.info(f"Forward times: {forward_times}")
    logger.info(f"Backward times: {backward_times}")

    forward_mean = np.mean(forward_times)
    forward_std = np.std(forward_times)
    backward_mean = np.mean(backward_times)
    backward_std = np.std(backward_times)

    logger.info(f"Forward mean: {forward_mean:.6f}s, std: {forward_std:.6f}s")
    logger.info(f"Backward mean: {backward_mean:.6f}s, std: {backward_std:.6f}s")
    return np.array([forward_mean, forward_std, backward_mean, backward_std])


def model_benchmarking(config: dict):
    ModelConfig = namedtuple("ModelConfig", ["name", "d_model", "d_ff", "num_layers", "num_heads"])
    small_model = ModelConfig(name="small", d_model=768, d_ff=3072, num_layers=12, num_heads=12)
    medium_model = ModelConfig(name="medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16)
    large_model = ModelConfig(name="large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20)
    xl_model = ModelConfig(name="xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
    two_seven_model = ModelConfig(name="2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32)

    results = []
    #  [medium_model, large_model, xl_model, two_seven_model]
    for model in [small_model, medium_model]:
        config.update(model._asdict())
        logger.info(f"Benchmarking {model.name}...")
        result = benchmarking(config)
        results.append(
            {
                "model": model.name,
                "forward_mean": result[0],
                "forward_std": result[1],
                "backward_mean": result[2],
                "backward_std": result[3],
            }
        )
        del model
        torch.cuda.empty_cache()
    logger.info("Benchmarking results:")
    print(pd.DataFrame(results))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--execution_steps", type=int, default=10)

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--use_annotated", action="store_true", default=False)
    parser.add_argument("--use_bf16", action="store_true", default=False)
    parser.add_argument("--profile_memory", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmarking(vars(args))

    model_benchmarking(vars(args))


if __name__ == "__main__":
    main()

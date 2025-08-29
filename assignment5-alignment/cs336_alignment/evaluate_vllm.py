import json

from typing import List, Callable
from vllm import LLM, SamplingParams


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    gold_answers: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str = None,
) -> list[dict]:
    """
    Evaluate a language model on a list of prompts, compute evaluation metrics, and serialize results to disk if output_path is provided.
    Returns a list of dictionaries containing the prompt, prediction, gold answer, and evaluation metric.
    """
    generations = []

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for output, prompt, gold in zip(outputs, prompts, gold_answers):
        pred = output.outputs[0].text.strip()
        metric = reward_fn(pred, gold)
        generations.append({"prompt": prompt, "prediction": pred, "gold": gold, "metric": metric})

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in generations:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return generations

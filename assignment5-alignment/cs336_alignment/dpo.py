import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizerBase

from cs336_alignment.common import ALPACA_SFT_PROMPT_PATH


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.
    """
    with open(ALPACA_SFT_PROMPT_PATH, "r", encoding="utf-8") as f:
        alpaca_sft_prompt = f.read().strip()

    eos = tokenizer.eos_token or "<|end_of_text|>"
    text_chosen = alpaca_sft_prompt.format(instruction=prompt, response=response_chosen) + eos
    text_rejected = alpaca_sft_prompt.format(instruction=prompt, response=response_rejected) + eos

    ids_chosen = tokenizer.encode(text_chosen, add_special_tokens=False, return_tensors="pt")
    ids_rejected = tokenizer.encode(text_rejected, add_special_tokens=False, return_tensors="pt")

    lm_ids_chosen = ids_chosen.to(next(lm.parameters()).device)
    lm_ids_rejected = ids_rejected.to(next(lm.parameters()).device)
    lm_ref_ids_chosen = ids_chosen.to(next(lm_ref.parameters()).device)
    lm_ref_ids_rejected = ids_rejected.to(next(lm_ref.parameters()).device)

    with torch.no_grad():
        lm_chosen_logits = lm(lm_ids_chosen).logits
        lm_rejected_logits = lm(lm_ids_rejected).logits

        lm_ref_chosen_logits = lm_ref(lm_ref_ids_chosen).logits
        lm_ref_rejected_logits = lm_ref(lm_ref_ids_rejected).logits

    def get_logp(logits: torch.Tensor, input_ids: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[0, :-1].gather(1, input_ids[:, 1:].T).squeeze(1)
        return token_log_probs.sum()

    logp_lm_chosen = get_logp(lm_chosen_logits, lm_ids_chosen)
    logp_lm_rejected = get_logp(lm_rejected_logits, lm_ids_rejected)
    logp_lm_ref_chosen = get_logp(lm_ref_chosen_logits, lm_ref_ids_chosen)
    logp_lm_ref_rejected = get_logp(lm_ref_rejected_logits, lm_ref_ids_rejected)

    delta_pi = logp_lm_chosen - logp_lm_rejected
    delta_ref = logp_lm_ref_chosen - logp_lm_ref_rejected
    loss = -F.logsigmoid(beta * (delta_pi - delta_ref))

    return loss

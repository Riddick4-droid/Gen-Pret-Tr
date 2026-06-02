import torch
import torch.nn.functional as F
from src.logger import get_logger

logger = get_logger(__name__)

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """
    Apply top‑k and/or nucleus (top‑p) filtering to logits.
    In‑place modification of logits with -inf for filtered tokens.
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Remove all tokens with probability less than the last in the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep at least the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    return logits

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    device: str = "cuda"
) -> str:
    """
    Autoregressive text generation from a prompt.
    Supports greedy (temperature=0) and sampling with temperature, top‑k, top‑p.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        # Limit context window
        if generated.size(1) > model.max_seq_len:
            generated = generated[:, -model.max_seq_len:]

        logits, _ = model(generated, attention_mask=None)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-7)

        # Filter
        next_token_logits = top_k_top_p_filter(next_token_logits, top_k=top_k, top_p=top_p)

        # Sampling or greedy
        if temperature == 0.0:
            next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

@torch.no_grad()
def answer_question(
    model,
    tokenizer,
    question: str,
    context: str,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    device: str = "cuda",
    question_token: str = "<|question|>",
    context_token: str = "<|context|>",
    answer_token: str = "<|answer|>"
) -> str:
    """
    Answer a question given context using the fine‑tuned Q&A model.
    Formats the prompt with special tokens and generates the answer.
    """
    prompt = f"{question_token} {question} {context_token} {context} {answer_token}"
    return generate_text(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device
    )
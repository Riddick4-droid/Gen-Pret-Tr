import torch
import math
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.logger import get_logger
from src.losses.loss import compute_loss
from src.compute_metrics.metrics import evaluate_qa_batch, token_accuracy, perplexity
from functools import partial

logger = get_logger(__name__)

def evaluate_perplexity(model, dataloader, device):
    """
    Compute loss and perplexity over a dataset.
    """
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="Evaluating perplexity"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels)

        non_ignored = (labels != -100).sum().item()
        total_loss_sum += loss.item() * non_ignored
        total_tokens += non_ignored

    avg_loss = total_loss_sum / total_tokens if total_tokens else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, ppl


def generate_answer(model, tokenizer, question, context, max_new_tokens=30, device='cuda',
                    question_token="<|question|>", context_token="<|context|>", answer_token="<|answer|>"):
    """
    Generate an answer for a single question given context.
    """
    prompt = f"{question_token} {question} {context_token} {context} {answer_token}"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    model.eval()
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated, attention_mask=None)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    full_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    # Extract answer part
    ans_start = full_text.find(answer_token) + len(answer_token)
    answer = full_text[ans_start:].strip()
    return answer


def evaluate_qa(model, tokenizer, dataset, device, max_samples=200, max_new_tokens=30,
                question_token="<|question|>", context_token="<|context|>", answer_token="<|answer|>"):
    """
    Evaluate generative QA on a dataset (list of dicts from QADataset).
    Returns a dictionary of metrics.
    """
    predictions = []
    references = []
    n = min(max_samples, len(dataset))
    for idx in tqdm(range(n), desc="QA evaluation"):
        sample = dataset[idx]
        question = sample["question"]
        context = sample["context"]
        ref = sample["answer_raw"]

        pred = generate_answer(model, tokenizer, question, context,
                               max_new_tokens=max_new_tokens, device=device,
                               question_token=question_token,
                               context_token=context_token,
                               answer_token=answer_token)
        predictions.append(pred)
        references.append(ref)

    metrics = evaluate_qa_batch(predictions, references)
    logger.info(f"QA Metrics: {metrics}")
    return metrics


def run_evaluation(model, tokenizer, config, pt_test_loader, qa_dataset, device):
    """
    Run full evaluation: pre‑training perplexity + Q&A metrics.
    """
    results = {}
    # 1. Pre‑training test perplexity
    logger.info("=== Pre‑training Test Perplexity ===")
    test_loss, test_ppl = evaluate_perplexity(model, pt_test_loader, device)
    logger.info(f"Test loss: {test_loss:.4f}, Perplexity: {test_ppl:.1f}")
    results["pretrain_test_loss"] = test_loss
    results["pretrain_test_ppl"] = test_ppl

    # 2. Q&A metrics
    logger.info("=== Q&A Evaluation ===")
    qa_metrics = evaluate_qa(model, tokenizer, qa_dataset, device,
                             max_samples=config.get("eval", {}).get("qa_max_samples", 200))
    results.update(qa_metrics)
    return results
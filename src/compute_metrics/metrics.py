import re
import math
from typing import List
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def token_accuracy(logits, labels, ignore_index=-100):
    """
    Token‑level accuracy: fraction of non‑ignored tokens where
    the predicted token matches the target.
    """
    import torch
    preds = torch.argmax(logits, dim=-1)          # (B, T)
    mask = (labels != ignore_index)
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()

def perplexity(loss: float) -> float:
    """exp(cross‑entropy loss)"""
    return math.exp(loss) if loss < 100 else float('inf')

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    text = str(text).lower().strip()
    return re.sub(r'\s+', ' ', text)

def exact_match(prediction: str, reference: str) -> bool:
    """Case‑insensitive, whitespace‑normalised exact match."""
    return normalize_text(prediction) == normalize_text(reference)

def compute_bleu(reference_tokens: List[str], prediction_tokens: List[str]) -> float:
    """
    Sentence‑level BLEU with smooth method1.
    reference_tokens is a list of tokens for the single reference.
    """
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothie)

def compute_rouge(prediction: str, reference: str) -> dict:
    """
    Computes ROUGE‑1, ROUGE‑2, ROUGE‑L F1 scores.
    Returns a dict with keys 'rouge1', 'rouge2', 'rougeL'.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def evaluate_qa_batch(predictions: List[str], references: List[str]) -> dict:
    """
    Compute aggregate metrics for a list of predictions and references.
    Returns dict with average BLEU, ROUGE‑1/2/L, and exact match rate.
    """
    n = len(predictions)
    total_bleu = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    em_count = 0

    for pred, ref in zip(predictions, references):
        norm_pred = normalize_text(pred)
        norm_ref = normalize_text(ref)

        # Exact Match
        if norm_pred == norm_ref:
            em_count += 1

        # BLEU (tokenize by splitting on whitespace)
        ref_tokens = norm_ref.split()
        pred_tokens = norm_pred.split()
        total_bleu += compute_bleu(ref_tokens, pred_tokens)

        # ROUGE
        rouge_scores = compute_rouge(norm_pred, norm_ref)
        total_rouge1 += rouge_scores['rouge1']
        total_rouge2 += rouge_scores['rouge2']
        total_rougeL += rouge_scores['rougeL']

    return {
        "bleu": total_bleu / n,
        "rouge1": total_rouge1 / n,
        "rouge2": total_rouge2 / n,
        "rougeL": total_rougeL / n,
        "exact_match": em_count / n,
        "sample_count": n
    }
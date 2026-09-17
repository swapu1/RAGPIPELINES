"""
nlp_rag_metrics.py
===================
Standalone, pure mathematical metrics library for Information Retrieval (IR),
Context Quality, and RAG Answer Generation.

This library is self-contained:
  - Does NOT alter any existing files or retrieval pipelines.
  - Implements standard deterministic formulas for IR, Context, and NLP metrics.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence


# ==============================================================================
# 1. RETRIEVAL (IR) & RANKING METRICS
# ==============================================================================

def compute_precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of retrieved top-k items that are in expected_chunk_ids."""
    if k <= 0:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    if not top_k:
        return 0.0
    expected_set = set(expected_chunk_ids)
    hits = sum(1 for cid in top_k if cid in expected_set)
    return hits / k


def compute_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of expected_chunk_ids present in the top-k retrieved items."""
    if not expected_chunk_ids:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    hits = sum(1 for cid in set(expected_chunk_ids) if cid in top_k)
    return hits / len(set(expected_chunk_ids))


def compute_f1_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Harmonic mean of Precision@k and Recall@k."""
    p = compute_precision_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
    r = compute_recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
    if (p + r) == 0.0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


def compute_mrr(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
) -> float:
    """Mean Reciprocal Rank: 1 / (rank of first expected chunk, 1-indexed)."""
    if not expected_chunk_ids or not retrieved_chunk_ids:
        return 0.0
    expected_set = set(expected_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in expected_set:
            return 1.0 / rank
    return 0.0


def compute_average_precision(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
) -> float:
    """Average Precision (AP) for a single query."""
    if not expected_chunk_ids or not retrieved_chunk_ids:
        return 0.0
    expected_set = set(expected_chunk_ids)
    hits = 0
    sum_precisions = 0.0
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in expected_set:
            hits += 1
            sum_precisions += hits / rank
    return sum_precisions / len(expected_set)


def compute_ndcg_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    if not expected_chunk_ids or k <= 0:
        return 0.0
    expected_set = set(expected_chunk_ids)
    top_k = retrieved_chunk_ids[:k]
    
    # DCG calculation: rel_i / log2(i + 1) for i = 1..k
    dcg = 0.0
    for rank, cid in enumerate(top_k, start=1):
        rel = 1.0 if cid in expected_set else 0.0
        dcg += rel / math.log2(rank + 1)
        
    # IDCG calculation (all relevant items ranked at top)
    ideal_hits = min(len(expected_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    
    return dcg / idcg if idcg > 0 else 0.0


# ==============================================================================
# 2. CONTEXT QUALITY & EFFICIENCY METRICS
# ==============================================================================

def compute_context_metrics(
    supplied_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    chunk_token_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    """Computes Context Recall, Context Precision, Sufficiency, and Context Noise Ratio."""
    expected_set = set(expected_chunk_ids)
    supplied_list = list(supplied_chunk_ids)
    supplied_set = set(supplied_list)
    
    if not expected_set:
        return {
            "context_recall": 0.0,
            "context_precision": 0.0,
            "context_sufficiency": 0.0,
            "context_noise_ratio": 0.0,
        }
    
    # Context Recall: |Supplied ∩ Expected| / |Expected|
    relevant_in_context = expected_set & supplied_set
    c_recall = len(relevant_in_context) / len(expected_set)
    
    # Context Precision: |Supplied ∩ Expected| / |Supplied|
    c_precision = len(relevant_in_context) / len(supplied_list) if supplied_list else 0.0
    
    # Context Sufficiency: 1.0 if at least one gold chunk is present, else 0.0
    c_sufficiency = 1.0 if len(relevant_in_context) > 0 else 0.0
    
    # Context Noise Ratio: 1 - (tokens in relevant retrieved context / tokens in supplied context)
    if chunk_token_counts:
        relevant_tokens = sum(chunk_token_counts.get(cid, 0) for cid in relevant_in_context)
        total_tokens = sum(chunk_token_counts.get(cid, 0) for cid in supplied_list)
        if total_tokens > 0:
            noise_ratio = 1.0 - (relevant_tokens / total_tokens)
        else:
            noise_ratio = 0.0
    else:
        # Fallback to chunk count ratio if token counts unavailable
        noise_ratio = 1.0 - c_precision if supplied_list else 0.0
        
    return {
        "context_recall": c_recall,
        "context_precision": c_precision,
        "context_sufficiency": c_sufficiency,
        "context_noise_ratio": max(0.0, min(1.0, noise_ratio)),
    }


# ==============================================================================
# 3. CLASSICAL NLP TEXT OVERLAP METRICS (ROUGE / BLEU)
# ==============================================================================

def _tokenize_text(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for text metrics."""
    return re.findall(r"\b\w+\b", (text or "").lower())


def _get_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Extract n-grams from a token sequence."""
    if len(tokens) < n or n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _longest_common_subsequence_length(s1: Sequence[str], s2: Sequence[str]) -> int:
    """Compute length of longest common subsequence between two token lists."""
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge_scores(
    candidate_text: str,
    reference_text: str,
) -> dict[str, dict[str, float]]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L (precision, recall, f1)."""
    cand_tokens = _tokenize_text(candidate_text)
    ref_tokens = _tokenize_text(reference_text)
    
    results = {}
    
    for name, n in [("rouge1", 1), ("rouge2", 2)]:
        cand_ngrams = _get_ngrams(cand_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        
        overlap_count = 0
        for ng, count in cand_ngrams.items():
            overlap_count += min(count, ref_ngrams.get(ng, 0))
            
        cand_total = sum(cand_ngrams.values())
        ref_total = sum(ref_ngrams.values())
        
        p = (overlap_count / cand_total) if cand_total > 0 else 0.0
        r = (overlap_count / ref_total) if ref_total > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        results[name] = {"precision": p, "recall": r, "f1": f1}
        
    # ROUGE-L
    lcs_len = _longest_common_subsequence_length(cand_tokens, ref_tokens)
    p_l = (lcs_len / len(cand_tokens)) if cand_tokens else 0.0
    r_l = (lcs_len / len(ref_tokens)) if ref_tokens else 0.0
    f1_l = (2 * p_l * r_l / (p_l + r_l)) if (p_l + r_l) > 0 else 0.0
    results["rougeL"] = {"precision": p_l, "recall": r_l, "f1": f1_l}
    
    return results


def compute_bleu_scores(
    candidate_text: str,
    reference_text: str,
    max_n: int = 4,
) -> dict[str, float]:
    """Computes BLEU-1, BLEU-2, and BLEU-4 with brevity penalty."""
    cand_tokens = _tokenize_text(candidate_text)
    ref_tokens = _tokenize_text(reference_text)
    
    c_len = len(cand_tokens)
    r_len = len(ref_tokens)
    
    if c_len == 0 or r_len == 0:
        return {f"bleu{i}": 0.0 for i in [1, 2, 4]}
        
    # Brevity penalty
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (r_len / c_len))
        
    precisions = []
    for n in range(1, max_n + 1):
        cand_ng = _get_ngrams(cand_tokens, n)
        ref_ng = _get_ngrams(ref_tokens, n)
        overlap = sum(min(cnt, ref_ng.get(ng, 0)) for ng, cnt in cand_ng.items())
        total = sum(cand_ng.values())
        p_n = (overlap / total) if total > 0 else 0.0
        precisions.append(p_n)
        
    out = {}
    # BLEU-1
    out["bleu1"] = bp * precisions[0]
    # BLEU-2
    if precisions[0] > 0 and precisions[1] > 0:
        out["bleu2"] = bp * math.exp(0.5 * (math.log(precisions[0]) + math.log(precisions[1])))
    else:
        out["bleu2"] = 0.0
    # BLEU-4
    if all(p > 0 for p in precisions[:4]):
        log_avg = sum(0.25 * math.log(p) for p in precisions[:4])
        out["bleu4"] = bp * math.exp(log_avg)
    else:
        out["bleu4"] = 0.0
        
    return out


# ==============================================================================
# 4. RAG GROUNDING & CITATION QUALITY METRICS
# ==============================================================================

def compute_citation_metrics(
    claimed_chunk_ids: Sequence[str],
    supplied_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
) -> dict[str, float]:
    """Computes Citation Precision, Gold Evidence Citation Coverage, and Citation Validity."""
    claimed_list = list(claimed_chunk_ids)
    supplied_set = set(supplied_chunk_ids)
    expected_set = set(expected_chunk_ids)
    
    if not claimed_list:
        return {
            "citation_precision": 1.0,  # No false citations claimed
            "citation_coverage": 0.0,
            "citation_validity": 1.0,
        }
        
    # Citation Precision: Fraction of claimed citations that were actually in the supplied context
    valid_citations = [cid for cid in claimed_list if cid in supplied_set]
    precision = len(valid_citations) / len(claimed_list)
    
    # Gold Evidence Citation Coverage / Recall:
    # |Claimed Gold Chunks| / |Gold Chunks Present in Supplied Context|
    gold_in_supplied = expected_set & supplied_set
    if gold_in_supplied:
        claimed_gold = set(claimed_list) & gold_in_supplied
        coverage = len(claimed_gold) / len(gold_in_supplied)
    else:
        coverage = 0.0
        
    return {
        "citation_precision": precision,
        "citation_coverage": coverage,
        "citation_validity": 1.0 if precision == 1.0 else precision,
    }


def compute_faithfulness_heuristic(
    answer_text: str,
    context_text: str,
) -> float:
    """Heuristic sentence-level claim groundedness verification."""
    if not answer_text or not context_text:
        return 0.0
        
    # Split into rough sentence claims
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer_text) if len(s.strip().split()) >= 4]
    if not sentences:
        return 1.0
        
    context_tokens = set(_tokenize_text(context_text))
    supported_claims = 0
    
    for sentence in sentences:
        s_tokens = _tokenize_text(sentence)
        non_stop = [t for t in s_tokens if len(t) > 3]
        if not non_stop:
            supported_claims += 1
            continue
        overlap = sum(1 for t in non_stop if t in context_tokens)
        ratio = overlap / len(non_stop)
        if ratio >= 0.50:  # At least 50% of substantial claim tokens grounded in context
            supported_claims += 1
            
    return supported_claims / len(sentences)


def compute_answer_relevance(
    query_text: str,
    answer_text: str,
) -> float:
    """Deterministic token-overlap proxy for answer relevance."""
    q_tokens = set(_tokenize_text(query_text))
    a_tokens = set(_tokenize_text(answer_text))
    if not q_tokens or not a_tokens:
        return 0.0
    intersection = q_tokens & a_tokens
    return len(intersection) / len(q_tokens)

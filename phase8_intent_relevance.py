"""Phase 8 -- Query-conditioned intent relevance and reranking.

Sits AFTER initial retrieval (frozen phase3_hybrid/phase3_rerank stack)
and the intent classifier (phase8_intent_classification), and feeds the
refinement loop (phase8_refinement):

    initial retrieval (phase3, FROZEN, unchanged)
        -> phase8_intent_classification.classify_intent()   (query-time)
        -> THIS MODULE: per-chunk, PER-QUERY intent relevance
        -> phase8_refinement (coverage check -> optional refined retrieval)

HARD REQUIREMENT this module enforces structurally
----------------------------------------------------
Intent relevance is a function of (query, intent, chunk) computed HERE, at
rerank time. It is never written back onto a chunk record and never
persisted -- nothing in this module writes to chunks.json,
chunk_concepts*.json, or any ingestion-time artifact. This is the
counterpart, at the scoring layer, to phase6_chunk_annotation.py's
concept tags: those ARE permanent (ontology identity does not depend on
the question being asked), intent relevance is NOT (the same "Risk for
Bleeding" chunk is highly relevant to a risk_factors query and only
weakly relevant to a definition query).

Scoring design (mirrors the additive-signal discipline already
established for the ontology bridge in phase5_ontology_bridge_v2.py)
----------------------------------------------------------------------------
combined_score = final_score (frozen phase3_rerank output, UNCHANGED)
                  + INTENT_RELEVANCE_WEIGHT * intent_relevance_score

intent_relevance_score in [0, 1] is itself:
    CUE_WEIGHT   * structural_cue_score(chunk_text, primary_intent)
  + OVERLAP_WEIGHT * query_term_overlap(query, chunk_text)

structural_cue_score reuses the SAME closed, curated cue-phrase table
phase8_intent_classification.py already defines (no second vocabulary) --
here it asks "does this chunk's own wording carry the phrasing of this
intent" (e.g. a "Risk factor:" chunk scores high on RISK_FACTORS), the
mirror-image of asking whether the QUERY carries that phrasing.

query_term_overlap is a plain, deterministic Jaccard-style token overlap
between the query and the chunk (via phase5_ontology.tokenize/STOPWORDS,
reused rather than re-implemented) -- this is what makes the signal
QUERY-CONDITIONED rather than a static per-chunk property: the same chunk
gets a different overlap score for a different query, even under the
same primary_intent.

Neither term replaces or is derived from BM25/FAISS/hybrid/rerank/
ontology-bridge scores; final_score is passed through untouched and
intent_relevance_score is purely additive, exactly like the ontology
bridge's own combination rule.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from phase5_ontology import STOPWORDS, tokenize
from phase8_intent_classification import INTENT_CUES

PHASE8_INTENT_RELEVANCE_VERSION = "phase8_intent_relevance_v1"

# Fixed, declared weights (not tuned against the 24-query eval set),
# same discipline as phase5_ontology_bridge_v2's MESH/DOID/SYMP weights.
CUE_WEIGHT = 0.6
OVERLAP_WEIGHT = 0.4
CUE_SATURATION = 3.0  # raw weighted cue-hit total that saturates to 1.0

DEFAULT_INTENT_RELEVANCE_WEIGHT = 0.05  # Base-gated scaling factor (multiplicative boost)

DEFAULT_COVERAGE_THRESHOLD = 0.35
DEFAULT_MIN_STRONG_CHUNKS = 1

COVERAGE_SUFFICIENT = "sufficient"
COVERAGE_WEAK = "weak"

_INTENT_STOPWORDS = frozenset({
    "what", "how", "why", "when", "who", "where", "which",
    "difference", "between", "versus", "compare", "compared",
    "risk", "factors", "factor", "causes", "cause", "symptoms", "signs",
    "defining", "characteristics", "nurse", "nursing", "client", "patient",
    "use", "type", "instructions", "increase", "increases", "recommended",
    "threshold", "holding", "due", "care", "caregiver", "caregivers",
})


def _extract_entity_anchor(query: str) -> str | None:
    """Identify the primary topical entity noun in the clinical query.
    Extracts substantive terms after removing clinical framing and intent stopwords.
    """
    raw_tokens = [t for t in tokenize(query) if t not in STOPWORDS]
    candidates = [t for t in raw_tokens if t not in _INTENT_STOPWORDS]
    return candidates[-1] if candidates else None


def _structural_cue_score(chunk_text: str, intent: str) -> float:
    cues = INTENT_CUES.get(intent)
    if not cues or not chunk_text:
        return 0.0
    raw = sum(weight for pattern, weight in cues if pattern.search(chunk_text))
    return min(1.0, raw / CUE_SATURATION)


def _query_term_overlap(query: str, chunk_text: str) -> float:
    query_tokens = {t for t in tokenize(query) if t not in STOPWORDS}
    if not query_tokens:
        return 0.0
    chunk_tokens = {t for t in tokenize(chunk_text or "") if t not in STOPWORDS}
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def score_chunk_intent_relevance(
    query: str, primary_intent: str, chunk_text: str
) -> float:
    """Per-(query, intent, chunk) relevance in [0, 1]. Deterministic,
    no LLM, no persisted state.
    """
    chunk_tokens = {t for t in tokenize(chunk_text or "") if t not in STOPWORDS}
    entity_anchor = _extract_entity_anchor(query)
    if entity_anchor and entity_anchor not in chunk_tokens:
        # Grounding gate: off-topic chunk missing the query entity
        # cannot receive intent boost.
        return 0.0

    if primary_intent not in INTENT_CUES:
        # UNKNOWN or unrecognized intent -> fall back to plain query-term overlap
        return round(_query_term_overlap(query, chunk_text), 6)

    cue_score = _structural_cue_score(chunk_text, primary_intent)
    overlap_score = _query_term_overlap(query, chunk_text)
    combined = CUE_WEIGHT * cue_score + OVERLAP_WEIGHT * overlap_score
    return round(max(0.0, min(1.0, combined)), 6)


def _validate_results(results: Any) -> list[dict[str, Any]]:
    if not isinstance(results, list):
        raise ValueError("results must be a list of chunk result dicts")
    for r in results:
        if not isinstance(r, dict):
            raise ValueError("each result must be a dict")
        if not isinstance(r.get("chunk_id"), str) or not r["chunk_id"].strip():
            raise ValueError("each result must have a non-empty chunk_id")
        if "final_score" not in r:
            raise ValueError(
                f"result for chunk_id {r.get('chunk_id')!r} is missing 'final_score' "
                "(expected output of phase3_rerank.rerank())"
            )
    return results


def rerank_with_intent(
    query: str,
    intent_result: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    intent_relevance_weight: float = DEFAULT_INTENT_RELEVANCE_WEIGHT,
) -> list[dict[str, Any]]:
    """Attach query-conditioned intent_relevance_score to every result and
    return a new list resorted by combined_score.

    Uses base-score gating: combined_score = final_score * (1 + weight * relevance).
    This ensures that:
    1. An off-topic or low-scoring chunk cannot leapfrog high-relevance chunks.
    2. When intent_relevance_weight == 0 or relevance == 0, base final_score order is preserved bit-for-bit.
    """
    validated = _validate_results(list(results))
    primary_intent = intent_result.get("primary_intent", "unknown")

    scored: list[dict[str, Any]] = []
    for r in validated:
        relevance = score_chunk_intent_relevance(query, primary_intent, r.get("text", ""))
        enriched = dict(r)
        enriched["intent_relevance_score"] = relevance
        base_final = float(r["final_score"])
        enriched["combined_score"] = base_final * (1.0 + intent_relevance_weight * relevance)
        scored.append(enriched)

    scored.sort(key=lambda item: (-item["combined_score"], -item["final_score"], item["chunk_id"]))
    return scored


def assess_coverage(
    top_results: Sequence[Mapping[str, Any]],
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    min_strong_chunks: int = DEFAULT_MIN_STRONG_CHUNKS,
) -> dict[str, Any]:
    """Summarize how well the current top results cover the query's intent.

    This NEVER recommends discarding chunks -- it only reports a status
    ("sufficient" | "weak") plus supporting stats for the refinement loop
    and, ultimately, for the LLM-facing coverage warning. An empty
    `top_results` list is reported as weak/zero-evidence, never as an
    error -- the caller (phase8_refinement / phase8_intent_pipeline)
    decides what to do about it.
    """
    if not top_results:
        return {
            "status": COVERAGE_WEAK,
            "strong_chunk_count": 0,
            "top_relevance": 0.0,
            "mean_relevance": 0.0,
            "num_chunks": 0,
        }

    relevances = [float(r.get("intent_relevance_score", 0.0)) for r in top_results]
    strong_count = sum(1 for v in relevances if v >= coverage_threshold)
    top_relevance = max(relevances)
    mean_relevance = sum(relevances) / len(relevances)

    status = (
        COVERAGE_SUFFICIENT
        if strong_count >= min_strong_chunks and top_relevance >= coverage_threshold
        else COVERAGE_WEAK
    )

    return {
        "status": status,
        "strong_chunk_count": strong_count,
        "top_relevance": round(top_relevance, 6),
        "mean_relevance": round(mean_relevance, 6),
        "num_chunks": len(top_results),
    }

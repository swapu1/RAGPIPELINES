"""Phase 8 -- Intent-aware retrieval + answer pipeline.

REPLACES the hard evidence-validation gate (phase6_evidence_validator.py /
phase6_retry_controller.py) as the layer between retrieval and answer
generation. This module does NOT import either of those modules and does
NOT call phase5_integration.run_integrated_search()'s
enable_evidence_validation/enable_retry path -- it is a new, parallel
orchestrator built directly on the still-frozen Phase 3/5 stack.

TARGET FLOW implemented here
----------------------------------------------------------------------------
    query
      -> phase8_intent_classification.classify_intent()
      -> phase5_ontology.resolve_query() + phase5_specificity + phase5_retrieval
         (existing, UNCHANGED query-enrichment stack -- concept tagging
         stays ingestion-time; nothing here re-tags chunks)
      -> phase3_rerank.run_rerank_search()   (existing, UNCHANGED: BM25 +
         semantic + hybrid fusion + structural quality rerank)
      -> phase8_intent_relevance.rerank_with_intent()   (query-conditioned
         auxiliary signal, additive, on top of final_score)
      -> phase8_refinement.run_refinement_loop()   (<= 3 rounds, gap-
         targeted refined retrieval, chunk-preserving)
      -> phase8-shaped output compatible with
         phase7_answer_generation.generate_answer()'s `integrated_output`
         contract (original_query + results at minimum)
      -> generate_intent_aware_answer(): calls the EXISTING, UNMODIFIED
         phase7_answer_generation building blocks (select_evidence isn't
         reused directly since there's no evidence_validation key here by
         design; build_prompt / extract_cited_chunk_ids / verify_citations
         / strip_citations_line ARE reused verbatim) with
         require_validation effectively off (no hard gate), and prepends
         a short coverage warning when intent coverage stayed weak.

No hard evidence-validation gate
----------------------------------------------------------------------------
There is intentionally no boolean "sufficient" flag that can block the
LLM call, and (correction pass) no zero-evidence short circuit either:
`generate_intent_aware_answer` calls the LLM whenever an `llm_client` is
supplied, regardless of whether retrieval found strong, weak, or zero
evidence. Evidence quality only changes which warning is prepended to the
answer (`WEAK_COVERAGE_WARNING` / `ZERO_EVIDENCE_WARNING`) and the
reported `grounding_status` -- it never withholds the call. Citations are
still verified deterministically against exactly the evidence chunks
supplied to the LLM (via the existing, unmodified
`phase7_answer_generation.verify_citations`), so a citation can never be
fabricated: with zero evidence the supplied set is empty, so every
claimed citation is necessarily invalid and dropped.
`phase8_intent_relevance.assess_coverage` and `phase8_refinement`
guarantee the accumulated result set is never emptied by intent
reasoning; only genuine "phase3 retrieval found nothing at all" can
produce zero results, and even that case still calls the LLM (with an
empty evidence block) and returns an explicit `ZERO_EVIDENCE_WARNING`-
prefixed answer rather than silently refusing.

Concept tagging remains ingestion-time
----------------------------------------------------------------------------
This module never calls phase6_chunk_annotation / phase6_concept_filtering
and never computes or attaches ontology concept tags to a chunk. It only
*reads* whatever phase3_rerank/phase5_ontology already resolved for the
QUERY (ontology concept resolution has always been query-time and is
unchanged) -- corpus-side concept tagging continues to happen exactly
where it already does, upstream of this module, at ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import phase3_hybrid
import phase3_rerank
import phase5_ontology
import phase5_retrieval
import phase8_intent_classification as intent_classifier
import phase8_intent_relevance as intent_relevance
import phase8_refinement as refinement
import phase7_answer_generation as p7

try:
    import phase5_specificity as _specificity_module
except ImportError:  # pragma: no cover - real deployments ship this module;
    # this fallback only keeps phase8 importable/testable in environments
    # where phase5_specificity.py happens to be absent (e.g. a partial
    # checkout). It reproduces ONLY the minimal, valid output shape
    # phase5_retrieval.build_retrieval_query() requires -- it does not
    # reimplement phase5_specificity's real routing/ambiguity logic, and
    # is never used when the real module is importable.
    _specificity_module = None

PHASE8_PIPELINE_VERSION = "phase8_intent_pipeline_v1"

DEFAULT_TOP_K = 10
DEFAULT_INITIAL_CANDIDATE_K = 20
DEFAULT_REFINE_CANDIDATE_K = 30
DEFAULT_MAX_REFINE_ROUNDS = 2

# WEAK_COVERAGE_WARNING / ZERO_EVIDENCE_WARNING are defined further below,
# next to generate_intent_aware_answer() where they are used.


def _fallback_specificity(query: str, resolved: dict[str, Any]) -> dict[str, Any]:
    """Minimal, valid-shaped stand-in for phase5_specificity.evaluate_specificity,
    used ONLY when the real module cannot be imported. See module docstring.
    """
    return {
        "intent": {"label": "unknown"},
        "specificity": {},
        "ambiguity": {"has_ambiguous_match": resolved.get("has_ambiguous_match", False)},
        "routing_state": "unknown",
    }


def _evaluate_specificity(query: str, resolved: dict[str, Any]) -> dict[str, Any]:
    if _specificity_module is not None:
        return _specificity_module.evaluate_specificity(query, resolved)
    return _fallback_specificity(query, resolved)


def run_intent_aware_search(
    query: str,
    store: "phase5_ontology.OntologyStore",
    *,
    bm25_index_path: str | Path,
    index_path: str | Path,
    index_metadata_path: str | Path,
    chunks_path: str | Path,
    model_name: str,
    top_k: int = DEFAULT_TOP_K,
    initial_candidate_k: int = DEFAULT_INITIAL_CANDIDATE_K,
    refine_candidate_k: int = DEFAULT_REFINE_CANDIDATE_K,
    max_refine_rounds: int = DEFAULT_MAX_REFINE_ROUNDS,
    bm25_weight: float = phase3_hybrid.DEFAULT_BM25_WEIGHT,
    semantic_weight: float = phase3_hybrid.DEFAULT_SEMANTIC_WEIGHT,
    relevance_weight: float = phase3_rerank.DEFAULT_RELEVANCE_WEIGHT,
    quality_weight: float = phase3_rerank.DEFAULT_QUALITY_WEIGHT,
    intent_relevance_weight: float = intent_relevance.DEFAULT_INTENT_RELEVANCE_WEIGHT,
    coverage_threshold: float = intent_relevance.DEFAULT_COVERAGE_THRESHOLD,
    min_strong_chunks: int = intent_relevance.DEFAULT_MIN_STRONG_CHUNKS,
    bm25_payload: Any = None,
    model: Any = None,
    index: Any = None,
    intent_model: Any = None,
    # --- Concept signal (Phase 3/5 bridge, phase3_concept.py) -----------
    # OFF by default. When enabled, this reuses the SAME `store` already
    # required by this function for Step 2's query-side ontology
    # resolution -- no second ontology store is needed, and the query is
    # resolved against it only once per retrieval call inside
    # phase3_concept.build_concept_retrieval_results (Step 3's `_search`),
    # which is a distinct, later resolve_query() call than Step 2's
    # (Step 2 resolves against the ORIGINAL query for retrieval-query
    # enrichment; Step 3's concept scoring resolves against whatever
    # search_query it is actually retrieving with -- the enriched
    # retrieval_query on the initial round, and the enriched refined
    # query on later rounds -- so concept matching stays consistent with
    # what was actually searched).
    enable_concept_signal: bool = False,
    concept_weight: float = phase3_hybrid.DEFAULT_CONCEPT_WEIGHT,
    chunk_concepts_path: str | Path | None = None,
    concept_index: Any = None,
    allow_concept_only: bool = phase3_hybrid.DEFAULT_ALLOW_CONCEPT_ONLY,
) -> dict[str, Any]:
    """Run the full Phase 8 intent-aware pipeline for one query.

    Returns a dict shaped as a strict superset of what
    ``phase5_integration.run_integrated_search`` returns (same
    ``original_query``/``retrieval_query``/``resolved_concepts``/
    ``has_ambiguous_match``/``specificity``/``prepared_query``/``results``
    keys, so it is a drop-in ``integrated_output`` for
    ``phase7_answer_generation.generate_answer``), PLUS Phase 8 fields:
    ``intent``, ``intent_coverage``, ``refinement``, and (new)
    ``concept_signal_enabled``/``concept_weight``.

    BACKWARD COMPATIBILITY: ``enable_concept_signal`` defaults to False,
    and ``concept_weight`` defaults to ``phase3_hybrid.DEFAULT_CONCEPT_WEIGHT``
    (0.0). Any existing caller of this function that does not pass the new
    kwargs gets EXACTLY the prior retrieval behavior -- every internal
    call this function makes to ``phase3_rerank.run_rerank_search`` below
    passes ``ontology_store=None`` in that case, which
    ``phase3_hybrid.run_hybrid_search`` treats as "concept signal fully
    disabled", reproducing the original BM25+semantic+quality-rerank
    pipeline bit-for-bit.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    original_query = query.strip()

    concept_ontology_store = store if enable_concept_signal else None

    # -- Step 1: intent classification (query-time only, never persisted).
    intent_result = intent_classifier.classify_intent(original_query, model=intent_model)

    # -- Step 2: existing, UNCHANGED ontology/specificity/query-enrichment
    # stack. Concept tagging itself is ingestion-time and untouched;
    # resolve_query() here only resolves THIS query's concepts, exactly
    # as it already did before Phase 8 existed.
    resolved = phase5_ontology.resolve_query(original_query, store)
    specificity = _evaluate_specificity(original_query, resolved)
    prepared = phase5_retrieval.build_retrieval_query(original_query, resolved, specificity)
    retrieval_query = prepared["retrieval_query"]

    # -- Step 3: initial retrieval via the existing, UNCHANGED Phase 3
    # stack (BM25 + semantic + hybrid fusion + structural quality rerank),
    # PLUS the optional concept overlap signal (Phase 3/5 bridge). The
    # concept signal is computed inside phase3_hybrid.run_hybrid_search
    # (via phase3_rerank.run_rerank_search's passthrough kwargs below) as
    # a THIRD fusion input alongside BM25/semantic -- never as query
    # expansion, and never by mutating search_query.
    def _search(search_query: str, candidate_k: int) -> list[dict[str, Any]]:
        return phase3_rerank.run_rerank_search(
            query=search_query,
            bm25_index_path=bm25_index_path,
            index_path=index_path,
            index_metadata_path=index_metadata_path,
            chunks_path=chunks_path,
            model_name=model_name,
            top_k=candidate_k,
            candidate_k=candidate_k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            relevance_weight=relevance_weight,
            quality_weight=quality_weight,
            bm25_payload=bm25_payload,
            model=model,
            index=index,
            ontology_store=concept_ontology_store,
            concept_weight=concept_weight,
            chunk_concepts_path=chunk_concepts_path,
            concept_index=concept_index,
            allow_concept_only=allow_concept_only,
        )

    initial_raw = _search(retrieval_query, initial_candidate_k)

    # -- Step 4: query-conditioned intent relevance (auxiliary, additive).
    initial_reranked = intent_relevance.rerank_with_intent(
        original_query, intent_result, initial_raw,
        intent_relevance_weight=intent_relevance_weight,
    )

    # -- Step 5: refinement loop (<= 3 rounds, chunk-preserving). Refined
    # searches reuse the SAME retrieval_query as their base (so ontology/
    # intent-enrichment terms from Step 2 are never lost), with the
    # Phase 8 gap terms appended on top.
    def _refine_search(gap_query_suffix_applied_to_original: str, candidate_k: int) -> list[dict[str, Any]]:
        # gap_query_suffix_applied_to_original is built by
        # phase8_refinement against `original_query`; re-derive the same
        # suffix against the ENRICHED retrieval_query so refined rounds
        # keep ontology/intent enrichment too.
        added_suffix = gap_query_suffix_applied_to_original[len(original_query):].strip()
        enriched_refined_query = f"{retrieval_query} {added_suffix}".strip()
        return _search(enriched_refined_query, candidate_k)

    refinement_output = refinement.run_refinement_loop(
        original_query,
        intent_result,
        initial_reranked,
        _refine_search,
        max_rounds=max_refine_rounds,
        top_k_for_coverage=top_k,
        coverage_threshold=coverage_threshold,
        min_strong_chunks=min_strong_chunks,
        refine_candidate_k=refine_candidate_k,
    )

    final_results = refinement_output["results"][:top_k]

    return {
        "phase": PHASE8_PIPELINE_VERSION,
        "original_query": original_query,
        "retrieval_query": retrieval_query,
        "resolved_concepts": resolved["concepts"],
        "has_ambiguous_match": resolved["has_ambiguous_match"],
        "specificity": specificity,
        "prepared_query": prepared,
        "results": final_results,
        "intent": intent_result,
        "intent_coverage": refinement_output["coverage"],
        "refinement": {
            "rounds_used": refinement_output["rounds_used"],
            "queries_tried": refinement_output["queries_tried"],
            "stop_reason": refinement_output["stop_reason"],
        },
        # Provenance for the concept signal (additive; harmless when
        # disabled). Individual result items already carry their own
        # concept_score/matched_concepts/concept_rank fields end-to-end
        # from phase3_hybrid.fuse_rankings through phase3_rerank.rerank
        # (both just do `dict(result)`, so extra keys pass through
        # untouched) -- these two top-level fields just make it obvious,
        # without inspecting individual chunks, whether this particular
        # call had the signal turned on at all.
        "concept_signal_enabled": concept_ontology_store is not None,
        "concept_weight": concept_weight if concept_ontology_store is not None else 0.0,
    }


# --------------------------------------------------------------------------
# Answer generation: reuses phase7_answer_generation building blocks
# verbatim (build_prompt / extract_cited_chunk_ids / verify_citations /
# strip_citations_line) without a hard evidence gate. The LLM is ALWAYS
# called when an llm_client is supplied -- weak or zero corpus evidence
# changes only which warning is prepended, never whether the LLM runs.
# Citations are still verified against exactly what was supplied (never
# trusted from the LLM's own claim), so a fabricated citation is dropped
# exactly as it would be with real evidence; with zero evidence the
# supplied set is empty, so EVERY citation the LLM might claim is
# necessarily invalid and dropped -- fabrication is structurally
# impossible here, not just discouraged.
# --------------------------------------------------------------------------

GROUNDING_INTENT_COVERED = "intent_covered"
GROUNDING_INTENT_UNCERTAIN = "intent_uncertain"
GROUNDING_ZERO_EVIDENCE = "zero_evidence_fallback"

WEAK_COVERAGE_WARNING = (
    "Note: intent matching could not fully verify that the retrieved "
    "evidence covers every part of this question. The answer below is "
    "grounded only in the excerpts shown; please confirm against a "
    "primary source if this is used for clinical decision-making."
)

ZERO_EVIDENCE_WARNING = (
    "Note: no supporting corpus evidence was retrieved for this question. "
    "The response below is NOT grounded in any excerpt from this corpus "
    "and must not be treated as a sourced clinical answer; please verify "
    "against a primary source."
)


def generate_intent_aware_answer(
    intent_output: dict[str, Any],
    *,
    llm_client: Optional[Callable[[str, str], str]],
    max_evidence_chunks: int = p7.DEFAULT_MAX_EVIDENCE_CHUNKS,
) -> dict[str, Any]:
    """Generate an answer from ``run_intent_aware_search`` output.

    There is no evidence-based short circuit: the LLM is called whenever
    ``llm_client`` is provided, regardless of how much (or how little)
    evidence retrieval found. Evidence quality (weak vs. sufficient vs.
    zero) only changes which warning is prepended to the answer text and
    the reported ``grounding_status`` -- it never withholds the call.
    """
    original_query = intent_output.get("original_query")
    if not isinstance(original_query, str) or not original_query.strip():
        raise ValueError("intent_output['original_query'] must be a non-empty string")
    if llm_client is None:
        raise ValueError("llm_client is required to generate an answer")

    results = intent_output.get("results") or []
    evidence_chunks = results[:max_evidence_chunks]
    coverage_status = (intent_output.get("intent_coverage") or {}).get("status", "weak")

    system_prompt = p7.SYSTEM_PROMPT
    user_prompt = p7.build_prompt(original_query, evidence_chunks)

    raw_output = llm_client(system_prompt, user_prompt)
    if not isinstance(raw_output, str):
        raise ValueError("llm_client must return a string")

    # Citations are cross-checked against exactly `evidence_chunks` --
    # when that list is empty, every claimed citation is necessarily
    # invalid and dropped here, so a hallucinated citation can never
    # reach the caller regardless of what the LLM outputs.
    cited_ids = p7.extract_cited_chunk_ids(raw_output)
    valid_ids, invalid_ids = p7.verify_citations(cited_ids, evidence_chunks)
    answer_text = p7.strip_citations_line(raw_output)

    if not evidence_chunks:
        answer_text = f"{ZERO_EVIDENCE_WARNING}\n\n{answer_text}"
        grounding_status = GROUNDING_ZERO_EVIDENCE
    elif coverage_status != "sufficient":
        answer_text = f"{WEAK_COVERAGE_WARNING}\n\n{answer_text}"
        grounding_status = GROUNDING_INTENT_UNCERTAIN
    else:
        grounding_status = GROUNDING_INTENT_COVERED

    meta_by_id = {c.get("chunk_id"): p7._chunk_metadata(c) for c in evidence_chunks}
    citations = [meta_by_id[cid] for cid in valid_ids]

    return {
        "phase": PHASE8_PIPELINE_VERSION,
        "original_query": original_query,
        "answer": answer_text,
        "citations": citations,
        "evidence_used": [c.get("chunk_id") for c in evidence_chunks],
        "grounding_status": grounding_status,
        "intent": intent_output.get("intent"),
        "intent_coverage": intent_output.get("intent_coverage"),
        "invalid_citations_dropped": invalid_ids,
    }

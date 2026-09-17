"""
Phase 3 hybrid retrieval.

Combines BM25 lexical retrieval, semantic/FAISS retrieval, and (optionally)
a corpus ontology-concept overlap signal using score-based weighted fusion
(min-max normalized scores). Existing retrieval modules are reused without
modification.

    hybrid_score = bm25_weight * normalized_bm25
                   + semantic_weight * normalized_semantic
                   + concept_weight * normalized_concept   (optional)

The concept signal is OFF by default (concept_weight=0.0, ontology_store
unset). With those defaults, every function in this module behaves
IDENTICALLY to the pre-concept implementation -- this is what makes
w_concept=0 a true backward-compatibility guarantee rather than an
approximation.

CANDIDATE-POOL POLICY -- recall tradeoff (requirement: reconsider whether
concept candidates should enter the union independently of BM25/FAISS)
----------------------------------------------------------------------------
Two options were considered for how concept-matched chunks enter the
candidate pool that gets fused/reranked:

  (A) RERANK-ONLY (default here, allow_concept_only=False): concept
      scoring is computed only over chunk_ids already present in the
      BM25 top-candidate_k or the FAISS top-candidate_k. A chunk with a
      strong ontology-concept match but absent from both retrievers'
      candidate lists is never scored and never surfaces.

  (B) CONCEPT-ONLY ADMISSION (allow_concept_only=True): every chunk
      tagged with ANY concept the query resolved to is added to the
      candidate pool, even if neither BM25 nor FAISS retrieved it.

Tradeoff, evaluated against THIS corpus (353 chunks, verified):
  - candidate_k in this codebase already defaults to 20 (initial) / 30
    (Phase 8 refinement) -- i.e. 6-8% of the ENTIRE corpus is already
    retrieved per query by BM25+FAISS alone before concept scoring even
    runs. The marginal recall a concept-only chunk could add is therefore
    small: it only matters for the case where a chunk is genuinely
    relevant AND both BM25 and FAISS independently missed it in their
    top ~25, which is the harder failure mode for concept tags to reliably
    catch (concept tags are precision-oriented lexical/stemmed matches,
    not a stronger retriever).
  - The precision risk of (B) is real and already documented for this
    exact artifact: phase5_ontology_bridge_v2.py's own docstring records
    that MeSH "Risk Factors" (D012307) tags 141/353 = 40% of chunks and
    "Evidence-Based Nursing" tags 119/353 = 34%. If a query resolves to
    either of those generic concepts, concept-only admission with
    allow_concept_only=True would inject on the order of a hundred
    additional candidates purely on a generic tag, most with no real
    lexical/semantic relevance to the specific question asked. The IDF
    damping in phase3_concept.py reduces their SCORE contribution but
    does not stop them from ENTERING the candidate pool in the first
    place under option (B).
  - phase5_ontology_bridge_v2.py itself already established the correct
    precedent for this exact tradeoff: its own `allow_bridge_only` flag
    defaults to False for the identical reason ("conservative bridge
    reranking... makes the ontology bridge a reranking signal rather than
    an independent corpus retriever").

DECISION: default to (A), rerank-only, matching the established
phase5_ontology_bridge_v2.py precedent, for a corpus this small where
BM25+FAISS candidate_k already covers a meaningful fraction of the corpus
and generic-concept precision risk is empirically demonstrated. Option
(B) remains available as an explicit opt-in (`allow_concept_only=True`)
for later experimentation, exactly mirroring the bridge module's own
`allow_bridge_only` escape hatch, rather than being silently unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import phase2_bm25
import phase3_search
import phase3_concept


DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_SEMANTIC_WEIGHT = 0.6
DEFAULT_CONCEPT_WEIGHT = 0.0
DEFAULT_CANDIDATE_K = 20
DEFAULT_TOP_K = 10
DEFAULT_ALLOW_CONCEPT_ONLY = False


def validate_weights(bm25_weight: Any, semantic_weight: Any) -> None:
    """Validate fusion weights (2-way; kept for backward compatibility)."""
    if isinstance(bm25_weight, bool) or isinstance(semantic_weight, bool):
        raise ValueError("weights must be numeric, not bool")

    if not isinstance(bm25_weight, (int, float)):
        raise ValueError("bm25_weight must be numeric")
    if not isinstance(semantic_weight, (int, float)):
        raise ValueError("semantic_weight must be numeric")

    if bm25_weight < 0:
        raise ValueError("bm25_weight cannot be negative")
    if semantic_weight < 0:
        raise ValueError("semantic_weight cannot be negative")
    if bm25_weight == 0 and semantic_weight == 0:
        raise ValueError("bm25_weight and semantic_weight cannot both be zero")


def validate_weights_3(bm25_weight: Any, semantic_weight: Any, concept_weight: Any) -> None:
    """Validate fusion weights including the optional concept weight.

    concept_weight == 0 is always individually valid (that's the disabled
    state); it only participates in the "not all zero" check together
    with the other two.
    """
    if isinstance(concept_weight, bool) or not isinstance(concept_weight, (int, float)):
        raise ValueError("concept_weight must be numeric")
    if concept_weight < 0:
        raise ValueError("concept_weight cannot be negative")

    if isinstance(bm25_weight, bool) or isinstance(semantic_weight, bool):
        raise ValueError("weights must be numeric, not bool")
    if not isinstance(bm25_weight, (int, float)):
        raise ValueError("bm25_weight must be numeric")
    if not isinstance(semantic_weight, (int, float)):
        raise ValueError("semantic_weight must be numeric")
    if bm25_weight < 0:
        raise ValueError("bm25_weight cannot be negative")
    if semantic_weight < 0:
        raise ValueError("semantic_weight cannot be negative")

    if bm25_weight == 0 and semantic_weight == 0 and concept_weight == 0:
        raise ValueError(
            "bm25_weight, semantic_weight and concept_weight cannot all be zero"
        )


def _rank_score(rank: int, list_length: int) -> float:
    """
    Convert 1-based rank into a normalized rank score.

    rank 1 in a list of length N -> 1.0
    rank N in a list of length N -> 1/N

    NOTE: fuse_rankings no longer uses this for scoring (it now does
    score-based min-max fusion). Kept for backward compatibility since
    it is a small, independently useful/testable utility.
    """
    if list_length <= 0:
        return 0.0
    if rank <= 0:
        raise ValueError("rank must be >= 1")
    if rank > list_length:
        raise ValueError("rank cannot exceed list_length")
    return (list_length - rank + 1) / list_length


def _minmax_normalize(scores: Sequence[float]) -> List[float]:
    """
    Min-max normalize a list of scores to the [0, 1] range.

    If the list is empty, returns an empty list.
    If all scores are equal (max == min), every score maps to 1.0 as a
    deterministic, safe fallback (avoids division by zero and avoids
    arbitrarily zeroing out a list of otherwise-tied candidates).
    """
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _required_numeric_score(result: Mapping[str, Any], retriever_name: str) -> float:
    """
    Extract a numeric score required for score-based fusion.

    Unlike `_numeric_or_none`, this raises if the score is missing,
    since min-max normalization needs a real numeric value for every
    candidate in the list being normalized.
    """
    score = _numeric_or_none(result.get("score"))
    if score is None:
        raise ValueError(
            f"Malformed {retriever_name} retrieval result: missing score "
            "(required for score-based fusion)"
        )
    return score


def _validate_result(result: Any, retriever_name: str) -> str:
    if not isinstance(result, dict):
        raise ValueError(f"Malformed {retriever_name} retrieval result: expected dict")
    chunk_id = result.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(f"Malformed {retriever_name} retrieval result: missing chunk_id")
    return chunk_id


def _numeric_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("retrieval score cannot be bool")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"retrieval score must be numeric, got {value!r}") from exc


def _dedupe_retrieval_list(
    results: Sequence[Mapping[str, Any]],
    retriever_name: str,
) -> List[Mapping[str, Any]]:
    """
    Deduplicate a single retriever's list.

    Earliest occurrence is retained because it represents the best rank.
    """
    seen: set[str] = set()
    output: List[Mapping[str, Any]] = []
    for result in results:
        chunk_id = _validate_result(result, retriever_name)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        output.append(result)
    return output


def fuse_rankings(
    bm25_results: Sequence[Mapping[str, Any]],
    semantic_results: Sequence[Mapping[str, Any]],
    *,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    concept_results: Optional[Sequence[Mapping[str, Any]]] = None,
    concept_weight: float = DEFAULT_CONCEPT_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    Fuse BM25, semantic, and (optionally) concept-overlap result lists using
    weighted min-max normalized scores.

        hybrid_score = bm25_weight * normalized_bm25
                        + semantic_weight * normalized_semantic
                        + concept_weight * normalized_concept   (if provided)

    BACKWARD COMPATIBILITY: when concept_results is None or
    concept_weight == 0, behavior and output shape are IDENTICAL to the
    original 2-way fusion -- no concept_rank/concept_score/
    matched_concepts keys are even attached to the output, and the
    concept branch of this function does not execute at all. This is
    verified by test_phase3_hybrid_concept.py
    (test_fuse_rankings_w_concept_zero_matches_original_behavior).

    A chunk returned by only some retrievers gets 0 contribution from the
    others. If a retriever's candidate list has all-equal scores, every
    candidate in that list normalizes to 1.0 (deterministic safe
    fallback, avoids division by zero).

    A chunk present ONLY in concept_results (i.e. concept-only admission,
    see phase3_concept.expand_candidates_with_concept_postings) is added
    to the fused pool with bm25_rank=semantic_rank=None and
    bm25_score=semantic_score=None -- exactly the same "absent from this
    retriever" representation already used for a chunk found by only one
    of bm25/semantic.

    Raw retriever scores are preserved on the output for provenance.
    """
    use_concept = concept_results is not None and concept_weight != 0.0
    if use_concept:
        validate_weights_3(bm25_weight, semantic_weight, concept_weight)
    else:
        validate_weights(bm25_weight, semantic_weight)

    bm25 = _dedupe_retrieval_list(bm25_results, "BM25")
    semantic = _dedupe_retrieval_list(semantic_results, "semantic")

    bm25_raw_scores = [_required_numeric_score(r, "BM25") for r in bm25]
    semantic_raw_scores = [_required_numeric_score(r, "semantic") for r in semantic]

    bm25_normalized = _minmax_normalize(bm25_raw_scores)
    semantic_normalized = _minmax_normalize(semantic_raw_scores)

    fused: Dict[str, Dict[str, Any]] = {}

    def _base_item(chunk_id: str) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "hybrid_score": 0.0,
            "source": "bm25",
            "bm25_rank": None,
            "semantic_rank": None,
            "bm25_score": None,
            "semantic_score": None,
        }
        if use_concept:
            item["concept_rank"] = None
            item["concept_score"] = None
            item["matched_concepts"] = []
        return item

    for position, (result, norm_score) in enumerate(
        zip(bm25, bm25_normalized), start=1
    ):
        chunk_id = _validate_result(result, "BM25")
        item = fused.setdefault(chunk_id, _base_item(chunk_id))
        item["bm25_rank"] = position
        item["bm25_score"] = _numeric_or_none(result.get("score"))
        item["hybrid_score"] += bm25_weight * norm_score

    for position, (result, norm_score) in enumerate(
        zip(semantic, semantic_normalized), start=1
    ):
        chunk_id = _validate_result(result, "semantic")
        item = fused.setdefault(chunk_id, _base_item(chunk_id))
        item["semantic_rank"] = position
        item["semantic_score"] = _numeric_or_none(result.get("score"))
        item["hybrid_score"] += semantic_weight * norm_score

    if use_concept:
        concept = _dedupe_retrieval_list(concept_results, "concept")
        concept_raw_scores = [_required_numeric_score(r, "concept") for r in concept]
        concept_normalized = _minmax_normalize(concept_raw_scores)

        for position, (result, norm_score) in enumerate(
            zip(concept, concept_normalized), start=1
        ):
            chunk_id = _validate_result(result, "concept")
            item = fused.setdefault(chunk_id, _base_item(chunk_id))
            item["concept_rank"] = position
            item["concept_score"] = _numeric_or_none(result.get("score"))
            item["matched_concepts"] = list(result.get("matched_concepts", []))
            item["hybrid_score"] += concept_weight * norm_score

    for item in fused.values():
        if use_concept:
            present = []
            if item["bm25_rank"] is not None:
                present.append("bm25")
            if item["semantic_rank"] is not None:
                present.append("semantic")
            if item.get("concept_rank") is not None:
                present.append("concept")
            item["source"] = "+".join(present) if present else "unknown"
        else:
            # EXACT original 2-way behavior preserved bit-for-bit (including
            # its pre-existing quirk: _base_item's dict-literal template
            # always initializes "source" to "bm25" regardless of which
            # retriever's setdefault() call actually created the entry, so
            # a chunk found ONLY by semantic search keeps "bm25" here, same
            # as before this module ever had a concept signal). This branch
            # only runs when concept scoring is disabled, so w_concept=0
            # reproduces the original output field-for-field, not just
            # score-for-score.
            if item["bm25_rank"] is not None and item["semantic_rank"] is not None:
                item["source"] = "both"

    def sort_key(item: Dict[str, Any]) -> tuple:
        rank_fields = [item["bm25_rank"], item["semantic_rank"]]
        if use_concept:
            rank_fields.append(item.get("concept_rank"))
        ranks = [rank for rank in rank_fields if rank is not None]
        best_rank = min(ranks) if ranks else 10**9
        return (-float(item["hybrid_score"]), best_rank, item["chunk_id"])

    return sorted(fused.values(), key=sort_key)


def attach_chunk_text(
    fused_results: Sequence[Mapping[str, Any]],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach authoritative chunk metadata/text from chunks.json."""
    output: List[Dict[str, Any]] = []

    for item in fused_results:
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Malformed fused result: missing chunk_id")

        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"chunk_id {chunk_id!r} not found in chunks.json")

        enriched = dict(item)
        enriched["pdf_page_start"] = chunk.get("pdf_page_start")
        enriched["pdf_page_end"] = chunk.get("pdf_page_end")
        enriched["section_title"] = chunk.get("section_title")
        enriched["chapter"] = chunk.get("chapter")
        enriched["text"] = chunk.get("text", "")
        output.append(enriched)

    return output


def _load_chunks_by_id(chunks_path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(chunks_path)
    with path.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a JSON list")

    by_id: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            if chunk_id in by_id:
                raise ValueError(f"Duplicate chunk_id in chunks.json: {chunk_id}")
            by_id[chunk_id] = chunk

    return by_id


def _validate_search_params(top_k: int, candidate_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
        raise ValueError("candidate_k must be a positive integer")


# Process-lifetime cache so repeated calls (e.g. one per eval query, or
# many queries within one long-running server process) don't reparse
# chunk_concepts_filtered.json / rebuild the inverted index every time.
# Keyed by resolved absolute path. Corpus concepts are static within a
# process run (never retagged per query -- see phase3_concept.py), so
# this cache is safe.
_CONCEPT_INDEX_CACHE: Dict[str, "phase3_concept.ConceptIndex"] = {}


def _get_concept_index(
    chunk_concepts_path: Optional[str | Path],
    concept_index: Optional["phase3_concept.ConceptIndex"],
) -> Optional["phase3_concept.ConceptIndex"]:
    if concept_index is not None:
        return concept_index
    if chunk_concepts_path is None:
        return None
    key = str(Path(chunk_concepts_path).resolve())
    if key not in _CONCEPT_INDEX_CACHE:
        _CONCEPT_INDEX_CACHE[key] = phase3_concept.ConceptIndex.from_path(chunk_concepts_path)
    return _CONCEPT_INDEX_CACHE[key]


def run_hybrid_search(
    *,
    query: str,
    bm25_index_path: str | Path,
    index_path: str | Path,
    index_metadata_path: str | Path,
    chunks_path: str | Path,
    model_name: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    model: Any = None,
    index: Any = None,
    bm25_payload: Any = None,
    # --- Concept signal (all optional, all default to disabled) ---
    ontology_store: Any = None,
    concept_weight: float = DEFAULT_CONCEPT_WEIGHT,
    chunk_concepts_path: Optional[str | Path] = None,
    concept_index: Optional["phase3_concept.ConceptIndex"] = None,
    allow_concept_only: bool = DEFAULT_ALLOW_CONCEPT_ONLY,
) -> List[Dict[str, Any]]:
    """Run BM25 + semantic (+ optional concept) retrieval and return fused
    top-k chunks.

    BACKWARD COMPATIBILITY: concept scoring only activates when BOTH an
    ontology_store is given AND concept_weight != 0 AND a concept index is
    resolvable (via concept_index or chunk_concepts_path). Any other
    combination -- in particular, every existing caller that does not
    pass these new kwargs at all -- falls back to EXACTLY the original
    2-way BM25+semantic fusion: same function calls, same arguments, same
    validate_weights() (2-arg) path, same output shape.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    _validate_search_params(top_k, candidate_k)

    resolved_concept_index = _get_concept_index(chunk_concepts_path, concept_index)
    concept_enabled = (
        ontology_store is not None
        and concept_weight != 0.0
        and resolved_concept_index is not None
    )

    if concept_enabled:
        validate_weights_3(bm25_weight, semantic_weight, concept_weight)
    else:
        validate_weights(bm25_weight, semantic_weight)

    if bm25_payload is None:
        bm25_payload = phase2_bm25.load_index(bm25_index_path)

    bm25_results = phase2_bm25.search(
        bm25_payload,
        query,
        top_k=candidate_k,
        method="bm25",
    )

    semantic_results = phase3_search.run_search(
        query,
        index_path,
        index_metadata_path,
        chunks_path,
        model_name,
        candidate_k,
        model,
        index,
    )

    concept_results = None
    resolved_query_concepts = None
    if concept_enabled:
        candidate_ids = list(
            dict.fromkeys(
                [r["chunk_id"] for r in bm25_results] + [r["chunk_id"] for r in semantic_results]
            )
        )
        concept_results, resolved_query_concepts = phase3_concept.build_concept_retrieval_results(
            query,
            ontology_store,
            candidate_ids,
            resolved_concept_index,
            allow_concept_only=allow_concept_only,
        )

    fused = fuse_rankings(
        bm25_results,
        semantic_results,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        concept_results=concept_results,
        concept_weight=concept_weight if concept_enabled else 0.0,
    )

    # Attach authoritative text/metadata from chunks.json after fusion.
    chunks_by_id = _load_chunks_by_id(chunks_path)
    enriched = attach_chunk_text(fused, chunks_by_id)

    if concept_enabled and resolved_query_concepts is not None:
        for item in enriched:
            item["query_resolved_concepts"] = resolved_query_concepts.get("concepts", [])

    return enriched[:top_k]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + semantic (+ optional concept) retrieval"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="run hybrid search")
    search.add_argument("--query", required=True)
    search.add_argument("--bm25-index", required=True)
    search.add_argument("--index", required=True)
    search.add_argument(
        "--index-metadata",
        default="phase3_output/index_metadata.json",
    )
    search.add_argument("--chunks", required=True)
    search.add_argument(
        "--model",
        default="pritamdeka/S-PubMedBert-MS-MARCO",
    )
    search.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    search.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
    )
    search.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
    )
    search.add_argument(
        "--semantic-weight",
        type=float,
        default=DEFAULT_SEMANTIC_WEIGHT,
    )
    search.add_argument(
        "--concept-weight",
        type=float,
        default=DEFAULT_CONCEPT_WEIGHT,
        help="Weight for the ontology-concept overlap signal (0 = disabled).",
    )
    search.add_argument(
        "--chunk-concepts",
        default=None,
        help="Path to chunk_concepts_filtered.json (required if --concept-weight != 0).",
    )
    search.add_argument(
        "--ontology-json",
        default=None,
        help="Path to a saved OntologyStore JSON (phase5_ontology.OntologyStore.save_json output).",
    )
    search.add_argument(
        "--allow-concept-only",
        action="store_true",
        help=(
            "Allow chunks matched ONLY by ontology concept overlap (absent "
            "from both BM25 and FAISS top candidate_k) into the candidate "
            "pool. Default off -- see phase3_hybrid.py module docstring "
            "for the recall/precision tradeoff."
        ),
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ontology_store = None
    if args.concept_weight != 0.0:
        if not args.chunk_concepts or not args.ontology_json:
            parser.exit(
                1,
                "Error: --concept-weight != 0 requires both --chunk-concepts "
                "and --ontology-json\n",
            )
        import phase5_ontology

        ontology_store = phase5_ontology.OntologyStore.load_json(args.ontology_json)

    try:
        results = run_hybrid_search(
            query=args.query,
            bm25_index_path=args.bm25_index,
            index_path=args.index,
            index_metadata_path=args.index_metadata,
            chunks_path=args.chunks,
            model_name=args.model,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            bm25_weight=args.bm25_weight,
            semantic_weight=args.semantic_weight,
            ontology_store=ontology_store,
            concept_weight=args.concept_weight,
            chunk_concepts_path=args.chunk_concepts,
            allow_concept_only=args.allow_concept_only,
        )
    except ValueError as exc:
        parser.exit(1, f"Error: {exc}\n")
    except FileNotFoundError as exc:
        parser.exit(1, f"Error: {exc}\n")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

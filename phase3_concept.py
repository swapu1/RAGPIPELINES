"""
Phase 3/5 bridge -- Corpus ontology-concept tagging as a DIRECT retrieval
signal, alongside BM25 and FAISS.

Verified against the real chunk_concepts_filtered.json shipped with this
project (353 records):

    [
      {
        "chunk_id": "nursing_handbook_v1_p00003_c001",
        "concepts": [
          {
            "ontology": "mesh" | "doid" | "symp" | "snomed",
            "concept_id": "...",
            "preferred_name": "...",
            "matched_text": "...",
            "match_type": "exact" | "stemmed",
            "ambiguous": false,
            "ambiguity_concept_ids": []
          },
          ...
        ],
        "concept_count": <int>,
        "exact_count": <int>,
        "stemmed_count": <int>
      },
      ...
    ]

Only "chunk_id" and each concept's "ontology"/"concept_id" are required by
this module; the other fields (preferred_name, matched_text, match_type,
ambiguous, ambiguity_concept_ids, concept_count, exact_count,
stemmed_count) are provenance and are not needed for scoring, so this
loader tolerates their presence/absence without treating format drift as
fatal.

This module does NOT retag the corpus per query. Corpus-side concepts are
precomputed once (chunk_concepts_filtered.json, produced upstream) and
loaded/indexed here at process start (cached -- see phase3_hybrid.py's
_get_concept_index). Query-side concepts are resolved at query time using
the existing, UNMODIFIED Phase 5 machinery (phase5_ontology.resolve_query).

Scoring is a direct set-overlap comparison -- never query expansion:

    (query_ontology, query_concept_id) == (chunk_ontology, chunk_concept_id)

weighted by an IDF-style corpus-rarity term, independently derived here
(not copied from phase5_ontology_bridge_v2.py, though the *idea* --
demote concepts that tag a large fraction of the corpus, e.g. MeSH
"Risk Factors" -- is the same one that module's docstring documents
finding empirically in this exact corpus).

CANDIDATE-POOL POLICY (see phase3_hybrid.py docstring for the full
recall-tradeoff writeup): by default, concept scoring is a RERANKING
signal restricted to chunks already retrieved by BM25 or FAISS
(`allow_concept_only=False`). A concept-only chunk (matched by ontology
overlap but absent from both BM25 and FAISS top candidate_k) can
optionally be admitted into the pool when `allow_concept_only=True`,
mirroring the same opt-in escape hatch phase5_ontology_bridge_v2.py
already established for its own bridge signal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

CONCEPT_PHASE_VERSION = "phase3_concept_v1"

# Down-weight for query concepts whose ontology resolution was itself
# ambiguous (phase5_ontology.ConceptMatch.ambiguous=True -- a stemmed
# match collided across multiple distinct concepts). An ambiguous query
# match is real evidence but weaker than an exact, unambiguous one.
AMBIGUOUS_QUERY_MATCH_WEIGHT = 0.5

# Concepts tagging more than this fraction of the (concept-tagged) corpus
# are corpus-generic and get their weight strongly damped rather than
# dropped. Empirically, in this corpus's real chunk_concepts_filtered.json,
# MeSH concepts like "Risk Factors" / "Evidence-Based Nursing" recur across
# a large minority of chunks (see phase5_ontology_bridge_v2.py's own
# documented df counts for this exact artifact) -- treating every match on
# such a concept as equally informative as a rare, specific match would let
# boilerplate vocabulary dominate the signal.
MAX_INFORMATIVE_DOCUMENT_FREQUENCY_RATIO = 0.5
GENERIC_CONCEPT_DAMPING_FACTOR = 0.25

ConceptKey = Tuple[str, str]  # (ontology, concept_id)


def _normalize_ontology(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def _normalize_concept_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v or None


def _extract_concept_keys_from_record_list(records: Any) -> List[ConceptKey]:
    """Extract (ontology, concept_id) pairs from a list of concept dicts.

    Tolerant of missing/malformed individual entries (skips them) so one
    bad record in chunk_concepts_filtered.json doesn't kill the whole load.
    Extra fields (preferred_name, matched_text, match_type, ambiguous,
    ambiguity_concept_ids) are ignored here -- they are provenance, not
    needed for direct concept-id matching.
    """
    keys: List[ConceptKey] = []
    if not isinstance(records, list):
        return keys
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        ontology = _normalize_ontology(rec.get("ontology"))
        concept_id = _normalize_concept_id(rec.get("concept_id"))
        if ontology is None or concept_id is None:
            continue
        keys.append((ontology, concept_id))
    return keys


def load_chunk_concepts(path: str | Path) -> Dict[str, List[ConceptKey]]:
    """Load precomputed corpus chunk->concept tags.

    Handles the REAL, verified on-disk shape used by this project:
        [ {"chunk_id": "...", "concepts": [ {...}, ... ], ...}, ... ]

    Also tolerantly supports a dict-keyed-by-chunk_id shape
    ({"<chunk_id>": [ {...}, ... ], ...}) in case a future export changes
    format, since detecting that shape costs nothing and breaks nothing.

    Returns: chunk_id -> list of (ontology, concept_id) tuples (deduped,
    order-preserving). Chunks with zero valid concepts are simply absent
    from the returned mapping (this project's real file has 3 such
    chunks) -- callers must treat "chunk_id not in mapping" the same as
    "chunk has no concepts", never as an error.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, List[ConceptKey]] = {}

    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, Mapping):
                continue
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            keys = _extract_concept_keys_from_record_list(entry.get("concepts"))
            deduped = list(dict.fromkeys(keys))
            if deduped:
                result[chunk_id] = deduped
        return result

    if isinstance(data, dict):
        for chunk_id, records in data.items():
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            keys = _extract_concept_keys_from_record_list(records)
            deduped = list(dict.fromkeys(keys))
            if deduped:
                result[chunk_id] = deduped
        return result

    raise ValueError(
        "chunk_concepts_filtered.json must be a JSON list of "
        "{chunk_id, concepts} records (the format actually shipped with "
        f"this project), or a dict keyed by chunk_id; got {type(data).__name__}"
    )


class ConceptIndex:
    """Lightweight in-memory inverted index: concept -> set(chunk_id).

    Deliberately not a real inverted-index library / database: the corpus
    is 353 chunks (verified), so a plain dict of sets plus a precomputed
    per-concept weight is the appropriate amount of engineering. Building
    this is O(total concept tags) ~= 14.5k rows for this corpus -- trivial,
    done once per process and cached (see phase3_hybrid._get_concept_index).
    """

    def __init__(self, chunk_concepts: Mapping[str, Sequence[ConceptKey]]):
        self.chunk_concepts: Dict[str, Tuple[ConceptKey, ...]] = {
            cid: tuple(dict.fromkeys(keys)) for cid, keys in chunk_concepts.items()
        }

        postings: Dict[ConceptKey, set] = {}
        for chunk_id, keys in self.chunk_concepts.items():
            for key in keys:
                postings.setdefault(key, set()).add(chunk_id)
        self.postings: Dict[ConceptKey, frozenset] = {
            key: frozenset(ids) for key, ids in postings.items()
        }

        self.num_tagged_chunks = len(self.chunk_concepts)
        self.concept_weight: Dict[ConceptKey, float] = self._compute_weights()

    def _compute_weights(self) -> Dict[ConceptKey, float]:
        """IDF-style weight per concept, damped for corpus-generic concepts.

        weight = log(1 + N / df)   [standard IDF shape, always > 0]
        then damped toward (not to) zero if the concept covers more than
        MAX_INFORMATIVE_DOCUMENT_FREQUENCY_RATIO of tagged chunks.
        """
        weights: Dict[ConceptKey, float] = {}
        n = max(1, self.num_tagged_chunks)
        for key, chunk_ids in self.postings.items():
            df = len(chunk_ids)
            idf = math.log(1.0 + (n / float(df)))
            ratio = df / float(n)
            if ratio > MAX_INFORMATIVE_DOCUMENT_FREQUENCY_RATIO:
                idf *= GENERIC_CONCEPT_DAMPING_FACTOR
            weights[key] = idf
        return weights

    @classmethod
    def from_path(cls, path: str | Path) -> "ConceptIndex":
        return cls(load_chunk_concepts(path))

    def concepts_for_chunk(self, chunk_id: str) -> Tuple[ConceptKey, ...]:
        return self.chunk_concepts.get(chunk_id, tuple())

    def chunks_for_concept(self, key: ConceptKey) -> frozenset:
        return self.postings.get(key, frozenset())


def extract_query_concept_keys(
    resolved_query: Mapping[str, Any],
) -> Dict[ConceptKey, float]:
    """Turn phase5_ontology.resolve_query()'s output into weighted concept
    keys for DIRECT matching (never text expansion -- the query string
    itself is never touched or rewritten by this function).

    Each distinct (ontology, concept_id) gets weight 1.0 normally, or
    AMBIGUOUS_QUERY_MATCH_WEIGHT if the match found was flagged ambiguous
    by phase5_ontology (a stem collision across multiple concepts). If the
    same concept is ALSO reached via an unambiguous exact match elsewhere
    in the query, the unambiguous weight wins (max, not overwrite).
    """
    weights: Dict[ConceptKey, float] = {}
    concepts = resolved_query.get("concepts", [])
    if not isinstance(concepts, list):
        return weights

    for match in concepts:
        if not isinstance(match, Mapping):
            continue
        ontology = _normalize_ontology(match.get("ontology"))
        concept_id = _normalize_concept_id(match.get("concept_id"))
        if ontology is None or concept_id is None:
            continue
        key = (ontology, concept_id)
        w = AMBIGUOUS_QUERY_MATCH_WEIGHT if match.get("ambiguous") else 1.0
        weights[key] = max(weights.get(key, 0.0), w)

    return weights


def score_candidates(
    query_concept_weights: Mapping[ConceptKey, float],
    candidate_chunk_ids: Iterable[str],
    concept_index: ConceptIndex,
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """Direct (ontology, concept_id) overlap scoring for candidate chunks.

    Returns:
      scores: chunk_id -> raw concept score (sum of matched concept
              weights; 0.0 for candidates with no overlap -- they are
              still included so downstream min-max normalization sees the
              full candidate set, exactly like a retriever that "found"
              them with the lowest score).
      matched_concepts: chunk_id -> list of "ontology:concept_id" strings
              that matched, for provenance/evaluation reporting.
    """
    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}

    if not query_concept_weights:
        for chunk_id in candidate_chunk_ids:
            scores[chunk_id] = 0.0
            matched[chunk_id] = []
        return scores, matched

    for chunk_id in candidate_chunk_ids:
        chunk_keys = concept_index.concepts_for_chunk(chunk_id)
        if not chunk_keys:
            scores[chunk_id] = 0.0
            matched[chunk_id] = []
            continue

        chunk_key_set = set(chunk_keys)
        total = 0.0
        hits: List[str] = []
        for key, qweight in query_concept_weights.items():
            if key in chunk_key_set:
                concept_weight = concept_index.concept_weight.get(key, 0.0)
                total += qweight * concept_weight
                hits.append("{}:{}".format(key[0], key[1]))

        scores[chunk_id] = total
        matched[chunk_id] = hits

    return scores, matched


def expand_candidates_with_concept_postings(
    query_concept_weights: Mapping[ConceptKey, float],
    existing_candidate_ids: Sequence[str],
    concept_index: ConceptIndex,
) -> List[str]:
    """Union `existing_candidate_ids` with every chunk tagged by ANY
    concept the query resolved to (concept-only admission).

    Bounded by construction: the result can never exceed
    concept_index.num_tagged_chunks (353 for the real corpus), so this is
    safe to compute unconditionally when allow_concept_only=True -- no
    extra capping/overengineering needed for a corpus this size.
    """
    expanded: Set[str] = set(existing_candidate_ids)
    for key in query_concept_weights:
        expanded |= concept_index.chunks_for_concept(key)
    # Deterministic ordering: original candidates first (preserves their
    # relative order), then newly-admitted concept-only chunks sorted by
    # chunk_id for reproducibility.
    existing_set = set(existing_candidate_ids)
    new_ids = sorted(expanded - existing_set)
    return list(existing_candidate_ids) + new_ids


def build_concept_retrieval_results(
    query: str,
    ontology_store: Any,
    candidate_chunk_ids: Sequence[str],
    concept_index: ConceptIndex,
    *,
    allow_concept_only: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """End-to-end: resolve query concepts (Phase 5, UNMODIFIED) and score
    them via DIRECT (ontology, concept_id) overlap against the given
    candidate chunk_ids' precomputed corpus concepts.

    This performs no query rewriting: `phase5_ontology.resolve_query`
    receives the raw query string exactly as any other caller would, and
    its output concepts are compared to corpus concept tags by identity,
    never turned into additional query terms.

    Returns (concept_results, resolved_query) where concept_results has
    the same shape as a bm25/semantic retrieval result list
    ({"chunk_id":..., "score":...}) so phase3_hybrid's existing
    _dedupe_retrieval_list / _minmax_normalize can be reused unmodified.
    """
    import phase5_ontology  # local import: keeps this module importable
    # even in contexts where phase5_ontology / nltk isn't installed and
    # concept scoring isn't being used (w_concept effectively 0).

    resolved = phase5_ontology.resolve_query(query, ontology_store)
    query_weights = extract_query_concept_keys(resolved)

    effective_candidates: Sequence[str] = candidate_chunk_ids
    if allow_concept_only:
        effective_candidates = expand_candidates_with_concept_postings(
            query_weights, candidate_chunk_ids, concept_index
        )

    scores, matched = score_candidates(query_weights, effective_candidates, concept_index)

    concept_results = [
        {"chunk_id": cid, "score": scores[cid], "matched_concepts": matched[cid]}
        for cid in effective_candidates
    ]
    return concept_results, resolved

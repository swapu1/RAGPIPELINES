"""Phase 5 — Medical ontology index + deterministic query concept resolution.

This module implements only the ontology/query-resolution block from the
Phase 5 architecture. It does not perform retrieval, reranking, validation,
or answer generation.

Resolution baseline:
    1. normalize query text
    2. exact-match ontology names/synonyms first
    3. stemmed fallback only for query spans not covered by exact matches
    4. if a stem maps to multiple distinct concepts, mark the match AMBIGUOUS
    5. concept tags are soft retrieval metadata; nothing is hard-filtered

The ontology vocabularies remain separate. MeSH is the primary organizing
vocabulary; SNOMED, DOID, and SYMP are supporting layers.

Ontology input schema (JSON):
[
  {
    "ontology": "mesh",
    "concept_id": "D000000",
    "preferred_name": "Example Concept",
    "synonyms": ["Example term", "Alternative term"],
    "semantic_type": "...",
    "parent_ids": ["..."]
  }
]

A file may contain records for one or more vocabularies. Records are never
merged into a single concept identity: concept_id is unique within its
ontology, and cross-ontology mappings are represented explicitly if later
added.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    from nltk.stem import PorterStemmer
except ImportError:  # pragma: no cover - exercised only in minimal installs
    PorterStemmer = None


PHASE5_ONTOLOGY_VERSION = "phase5_ontology_v1"
ONTOLOGY_PRIORITY = ("mesh", "snomed", "doid", "symp")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# --------------------------------------------------------------------------
# Single-token function-word guard.
#
# normalize_text() lowercases everything, which destroys the one signal
# ("A", "CAN", "OF", "DO", "WITH", ...) that would otherwise distinguish a
# real short clinical abbreviation/unit symbol from an ordinary English
# function word. Because exact/stemmed matching has no minimum-
# informativeness check, a bare single-token query like "a", "can", "of",
# "do", or "with" was able to resolve to whatever ontology record happens
# to carry that string as a synonym/abbreviation (e.g. a MeSH entry for the
# letter "A", a syndrome abbreviated "CAN", a unit abbreviated "OF").
#
# This is a fixed, deterministic linguistic list (closed-class English
# function words: articles, auxiliary/modal verbs, prepositions,
# conjunctions, pronouns, wh-words, and a small set of common adverbs/
# determiners) -- not a similarity/confidence threshold and not a per-
# concept filter. It only ever suppresses a match when the ENTIRE matched
# span is a single token AND that token is one of these closed-class
# words. Multi-word aliases that happen to contain a stopword (e.g. a
# genuine phrase like "shortness of breath" or "signs of infection") are
# completely unaffected, because the span there is longer than one token.
# Legitimate single-word medical concepts ("anxiety", "confusion",
# "airway") are unaffected because they are not in this list.
STOPWORDS = frozenset(
    {
        # articles / determiners
        "a", "an", "the", "this", "that", "these", "those", "any", "all",
        "each", "every", "some", "no", "such", "own", "same", "other",
        # auxiliary / modal verbs
        "am", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing",
        "have", "has", "had", "having",
        "can", "could", "will", "would", "shall", "should", "may",
        "might", "must",
        # prepositions
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "up",
        "down", "out", "off", "over", "under", "into", "onto", "about",
        "against", "between", "through", "during", "before", "after",
        "above", "below",
        # conjunctions
        "and", "or", "but", "nor", "so", "if", "then", "than", "as",
        "because", "while",
        # pronouns
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "mine", "your", "yours", "his", "its", "our",
        "ours", "their", "theirs",
        # wh-words
        "who", "whom", "whose", "which", "what", "when", "where", "why",
        "how",
        # misc common function words
        "not", "too", "very", "just", "also", "only", "again", "further",
        "once", "here", "there", "more", "most", "few",
    }
)


def _is_single_token_stopword(phrase_tokens: Sequence[str]) -> bool:
    """True only when the whole matched span is one closed-class word."""
    return len(phrase_tokens) == 1 and phrase_tokens[0] in STOPWORDS

# SNOMED CT RF2 description type identifiers (fixed, standard SCT concept IDs).
SNOMED_FSN_TYPE_ID = "900000000000003001"
SNOMED_SYNONYM_TYPE_ID = "900000000000013009"


def normalize_text(text: str) -> str:
    """Normalize text for deterministic concept matching."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    value = text.lower().replace("\u00ad", "-")
    value = re.sub(r"[-_/]+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def stem_tokens(tokens: Sequence[str]) -> list[str]:
    if PorterStemmer is None:
        raise RuntimeError(
            "NLTK is required for stemmed fallback. Install it with: pip install nltk"
        )
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


@dataclass(frozen=True)
class NursingContext:
    """Nursing-relevance overlay metadata for an existing ontology concept.

    This is populated from the SNOMED CT Nursing Problem List Subset (a
    2017 subset release) and attached to an already-loaded SNOMED concept.
    It is a soft retrieval/relevance signal only -- it never creates a new
    concept and it never overrides the concept's CURRENT SNOMED
    active/inactive status, which is determined solely by whichever
    (typically newer) SNOMED RF2 snapshot was loaded into the store.

    Fields:
        in_nursing_problem_subset: always True when this object exists on a
            record (kept explicit/serializable rather than implied by
            presence, so downstream code can check it without a None guard).
        umls_cui: UMLS CUI reported by the subset row, if any. Preserved as
            opaque metadata only -- no UMLS mapping/expansion is performed.
        subset_status_2017: the subset file's own SNOMED_CONCEPT_STATUS
            value at the time the subset was published (e.g. "Current" /
            "Inactive"). This is HISTORICAL subset-file metadata, not the
            concept's current SNOMED status, and must never be read as such.
        is_retired_from_subset: whether the subset itself later marked this
            entry retired from the nursing subset. This describes subset
            membership history, not current SNOMED concept status.
    """

    in_nursing_problem_subset: bool = True
    umls_cui: str | None = None
    subset_status_2017: str | None = None
    is_retired_from_subset: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_nursing_problem_subset": self.in_nursing_problem_subset,
            "umls_cui": self.umls_cui,
            "subset_status_2017": self.subset_status_2017,
            "is_retired_from_subset": self.is_retired_from_subset,
        }


@dataclass(frozen=True)
class ConceptRecord:
    ontology: str
    concept_id: str
    preferred_name: str
    synonyms: tuple[str, ...] = field(default_factory=tuple)
    semantic_type: str | None = None
    parent_ids: tuple[str, ...] = field(default_factory=tuple)
    # MeSH-only: descriptor tree numbers (a descriptor may have several).
    tree_numbers: tuple[str, ...] = field(default_factory=tuple)
    # MeSH-only: (qualifier_ui, qualifier_name) pairs. Qualifiers are kept as
    # metadata on the descriptor record; they are never treated as their own
    # independent concepts / vocabulary entries.
    qualifiers: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    # Optional free-text definition (e.g. SNOMED TextDefinition).
    definition: str | None = None
    # Optional nursing-relevance overlay (e.g. SNOMED CT Nursing Problem
    # List Subset membership). This is NOT a separate ontology -- it is
    # soft metadata attached to an existing concept (currently SNOMED
    # only). See NursingContext for field semantics.
    nursing_context: "NursingContext | None" = None

    @property
    def qualifiers_as_dicts(self) -> list[dict[str, str]]:
        return [
            {"qualifier_id": qid, "qualifier_name": qname}
            for qid, qname in self.qualifiers
        ]

    @property
    def aliases(self) -> tuple[str, ...]:
        seen: set[str] = set()
        values: list[str] = []
        for value in (self.preferred_name,) + self.synonyms:
            normalized = normalize_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(normalized)
        return tuple(values)


@dataclass(frozen=True)
class ConceptMatch:
    ontology: str
    concept_id: str
    preferred_name: str
    matched_text: str
    normalized_match: str
    match_type: str  # exact | stemmed
    ambiguous: bool = False
    ambiguity_concept_ids: tuple[str, ...] = field(default_factory=tuple)
    ambiguity_reason: str | None = None
    # Carried through from the matched ConceptRecord so callers get qualifier /
    # tree-number provenance without a second lookup. Qualifiers remain
    # represented separately from the descriptor concept itself (see spec).
    tree_numbers: tuple[str, ...] = field(default_factory=tuple)
    qualifiers: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    # Carried through from the matched ConceptRecord, mirroring
    # tree_numbers/qualifiers above. None when the concept has no nursing
    # subset overlay attached (the common case for non-SNOMED and
    # non-subset SNOMED concepts).
    nursing_context: "NursingContext | None" = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ontology": self.ontology,
            "concept_id": self.concept_id,
            "preferred_name": self.preferred_name,
            "matched_text": self.matched_text,
            "normalized_match": self.normalized_match,
            "match_type": self.match_type,
            "ambiguous": self.ambiguous,
            "ambiguity_concept_ids": list(self.ambiguity_concept_ids),
            "ambiguity_reason": self.ambiguity_reason,
            "tree_numbers": list(self.tree_numbers),
            "qualifiers": [
                {"qualifier_id": qid, "qualifier_name": qname}
                for qid, qname in self.qualifiers
            ],
            "nursing_context": (
                self.nursing_context.to_dict() if self.nursing_context else None
            ),
        }


class OntologyStore:
    """Separate, deterministic indexes for MeSH/SNOMED/DOID/SYMP."""

    def __init__(self, records: Iterable[Mapping[str, Any]] = ()) -> None:
        self._records: dict[tuple[str, str], ConceptRecord] = {}
        self._exact: dict[str, dict[str, set[str]]] = {
            ontology: defaultdict(set) for ontology in ONTOLOGY_PRIORITY
        }
        self._stem: dict[str, dict[str, set[str]]] = {
            ontology: defaultdict(set) for ontology in ONTOLOGY_PRIORITY
        }
        self.add_records(records)

    @staticmethod
    def _validate_ontology(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ontology must be a non-empty string")
        normalized = value.strip().lower()
        if normalized not in ONTOLOGY_PRIORITY:
            raise ValueError(
                "Unsupported ontology '{}'; expected one of {}".format(
                    value, ONTOLOGY_PRIORITY
                )
            )
        return normalized

    @staticmethod
    def _as_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
        if value is None:
            return tuple()
        if not isinstance(value, (list, tuple)):
            raise ValueError("{} must be a list/tuple of strings".format(field_name))
        output: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("{} must contain only non-empty strings".format(field_name))
            output.append(item.strip())
        return tuple(output)

    @staticmethod
    def _as_qualifier_tuple(value: Any) -> tuple[tuple[str, str], ...]:
        if value is None:
            return tuple()
        if not isinstance(value, (list, tuple)):
            raise ValueError("qualifiers must be a list of qualifier objects")
        output: list[tuple[str, str]] = []
        for item in value:
            if isinstance(item, Mapping):
                qid = item.get("qualifier_id")
                qname = item.get("qualifier_name")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                qid, qname = item
            else:
                raise ValueError(
                    "qualifiers entries must be {qualifier_id, qualifier_name} objects"
                )
            if (
                not isinstance(qid, str)
                or not qid.strip()
                or not isinstance(qname, str)
                or not qname.strip()
            ):
                raise ValueError(
                    "qualifier_id and qualifier_name must be non-empty strings"
                )
            output.append((qid.strip(), qname.strip()))
        return tuple(output)

    @staticmethod
    def _as_nursing_context(value: Any) -> "NursingContext | None":
        if value is None:
            return None
        if isinstance(value, NursingContext):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                "nursing_context must be an object with nursing overlay fields"
            )
        umls_cui = value.get("umls_cui")
        if umls_cui is not None and not isinstance(umls_cui, str):
            raise ValueError("nursing_context.umls_cui must be a string or null")
        subset_status_2017 = value.get("subset_status_2017")
        if subset_status_2017 is not None and not isinstance(subset_status_2017, str):
            raise ValueError(
                "nursing_context.subset_status_2017 must be a string or null"
            )
        return NursingContext(
            in_nursing_problem_subset=bool(
                value.get("in_nursing_problem_subset", True)
            ),
            umls_cui=umls_cui.strip() if isinstance(umls_cui, str) else None,
            subset_status_2017=(
                subset_status_2017.strip()
                if isinstance(subset_status_2017, str)
                else None
            ),
            is_retired_from_subset=bool(value.get("is_retired_from_subset", False)),
        )

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "OntologyStore":
        return cls(records)

    def add_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        for raw in records:
            if not isinstance(raw, Mapping):
                raise ValueError("Ontology records must be objects")

            ontology = self._validate_ontology(raw.get("ontology"))
            concept_id = raw.get("concept_id")
            preferred_name = raw.get("preferred_name")

            if not isinstance(concept_id, str) or not concept_id.strip():
                raise ValueError("concept_id must be a non-empty string")
            if not isinstance(preferred_name, str) or not preferred_name.strip():
                raise ValueError("preferred_name must be a non-empty string")

            key = (ontology, concept_id.strip())
            if key in self._records:
                raise ValueError(
                    "Duplicate ontology concept: {}/{}".format(ontology, concept_id)
                )

            synonyms = self._as_string_tuple(raw.get("synonyms", []), "synonyms")
            parent_ids = self._as_string_tuple(raw.get("parent_ids", []), "parent_ids")
            tree_numbers = self._as_string_tuple(raw.get("tree_numbers", []), "tree_numbers")
            qualifiers = self._as_qualifier_tuple(raw.get("qualifiers", []))
            semantic_type = raw.get("semantic_type")
            if semantic_type is not None and not isinstance(semantic_type, str):
                raise ValueError("semantic_type must be a string or null")
            definition = raw.get("definition")
            if definition is not None and not isinstance(definition, str):
                raise ValueError("definition must be a string or null")
            nursing_context = self._as_nursing_context(raw.get("nursing_context"))

            record = ConceptRecord(
                ontology=ontology,
                concept_id=concept_id.strip(),
                preferred_name=preferred_name.strip(),
                synonyms=synonyms,
                semantic_type=semantic_type.strip() if isinstance(semantic_type, str) else None,
                parent_ids=parent_ids,
                tree_numbers=tree_numbers,
                qualifiers=qualifiers,
                definition=definition.strip() if isinstance(definition, str) else None,
                nursing_context=nursing_context,
            )
            self._records[key] = record

            for alias in record.aliases:
                self._exact[ontology][alias].add(record.concept_id)
                alias_stems = " ".join(stem_tokens(tokenize(alias)))
                if alias_stems:
                    self._stem[ontology][alias_stems].add(record.concept_id)

    def get(self, ontology: str, concept_id: str) -> ConceptRecord | None:
        ontology = self._validate_ontology(ontology)
        return self._records.get((ontology, concept_id))

    def records(self, ontology: str | None = None) -> list[ConceptRecord]:
        if ontology is None:
            values = list(self._records.values())
        else:
            normalized = self._validate_ontology(ontology)
            values = [r for r in self._records.values() if r.ontology == normalized]
        return sorted(values, key=lambda record: (ONTOLOGY_PRIORITY.index(record.ontology), record.concept_id))

    def exact_lookup(self, normalized_phrase: str) -> list[ConceptRecord]:
        phrase = normalize_text(normalized_phrase)
        output: list[ConceptRecord] = []
        for ontology in ONTOLOGY_PRIORITY:
            ids = self._exact[ontology].get(phrase, set())
            for concept_id in sorted(ids):
                record = self._records[(ontology, concept_id)]
                output.append(record)
        return output

    def stem_lookup(self, normalized_phrase: str) -> list[ConceptRecord]:
        phrase = normalize_text(normalized_phrase)
        key = " ".join(stem_tokens(tokenize(phrase)))
        output: list[ConceptRecord] = []
        if not key:
            return output
        for ontology in ONTOLOGY_PRIORITY:
            ids = self._stem[ontology].get(key, set())
            for concept_id in sorted(ids):
                output.append(self._records[(ontology, concept_id)])
        return output

    def apply_nursing_subset(
        self,
        subset_records: Iterable[Mapping[str, Any]],
        *,
        ontology: str = "snomed",
    ) -> dict[str, Any]:
        """Attach nursing-relevance overlay metadata onto EXISTING concepts.

        ``subset_records`` are lightweight subset-membership rows (e.g. as
        produced by ``load_snomed_nursing_subset_csv``) -- each must contain
        at least a ``concept_id``. This method never creates a new
        ``ConceptRecord``: a subset row whose concept_id is not already
        present in this store (under ``ontology``) is reported as
        unmatched and skipped, per the architectural rule that the
        2026 SNOMED snapshot -- not the 2017 subset file -- is
        authoritative for which SNOMED concepts currently exist/are active.

        Attaching nursing_context never touches preferred_name, synonyms,
        definition, tree_numbers, qualifiers, or the exact/stemmed match
        indexes (those are alias-derived and aliases are unchanged), so
        existing query-resolution behavior is unaffected.

        Returns a deterministic report:
            {
                "matched_concept_ids": [...] (sorted),
                "unmatched_concept_ids": [...] (sorted),
                "duplicate_subset_concept_ids": [...] (sorted),
            }
        Duplicate subset rows for the same concept_id are handled
        deterministically: the FIRST row encountered (in iteration order)
        wins, and later duplicates are recorded but do not overwrite it.
        """
        normalized_ontology = self._validate_ontology(ontology)

        matched: set[str] = set()
        unmatched: set[str] = set()
        duplicates: set[str] = set()
        seen_ids: set[str] = set()

        for raw in subset_records:
            if not isinstance(raw, Mapping):
                raise ValueError("Nursing subset records must be objects")
            concept_id = raw.get("concept_id")
            if not isinstance(concept_id, str) or not concept_id.strip():
                raise ValueError("Nursing subset concept_id must be a non-empty string")
            concept_id = concept_id.strip()

            if concept_id in seen_ids:
                duplicates.add(concept_id)
                continue
            seen_ids.add(concept_id)

            key = (normalized_ontology, concept_id)
            existing = self._records.get(key)
            if existing is None:
                unmatched.add(concept_id)
                continue

            nursing_context = self._as_nursing_context(
                {
                    "in_nursing_problem_subset": True,
                    "umls_cui": raw.get("umls_cui"),
                    "subset_status_2017": raw.get("subset_status_2017"),
                    "is_retired_from_subset": raw.get("is_retired_from_subset", False),
                }
            )
            # Replace in place. Aliases/exact/stem indexes are untouched
            # because they are derived only from ontology/preferred_name/
            # synonyms, none of which change here.
            self._records[key] = ConceptRecord(
                ontology=existing.ontology,
                concept_id=existing.concept_id,
                preferred_name=existing.preferred_name,
                synonyms=existing.synonyms,
                semantic_type=existing.semantic_type,
                parent_ids=existing.parent_ids,
                tree_numbers=existing.tree_numbers,
                qualifiers=existing.qualifiers,
                definition=existing.definition,
                nursing_context=nursing_context,
            )
            matched.add(concept_id)

        return {
            "matched_concept_ids": sorted(matched),
            "unmatched_concept_ids": sorted(unmatched),
            "duplicate_subset_concept_ids": sorted(duplicates),
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        payload = [
            {
                "ontology": record.ontology,
                "concept_id": record.concept_id,
                "preferred_name": record.preferred_name,
                "synonyms": list(record.synonyms),
                "semantic_type": record.semantic_type,
                "parent_ids": list(record.parent_ids),
                "tree_numbers": list(record.tree_numbers),
                "qualifiers": record.qualifiers_as_dicts,
                "definition": record.definition,
                "nursing_context": (
                    record.nursing_context.to_dict() if record.nursing_context else None
                ),
            }
            for record in self.records()
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "OntologyStore":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("Ontology JSON must contain a list of concept records")
        return cls(payload)


def _find_exact_matches(
    query_tokens: Sequence[str], store: OntologyStore
) -> tuple[list[ConceptMatch], list[tuple[int, int]]]:
    """Find longest exact alias matches first and mark covered token spans."""
    matches: list[ConceptMatch] = []
    covered: list[tuple[int, int]] = []

    alias_phrases: set[str] = set()
    for record in store.records():
        alias_phrases.update(record.aliases)

    aliases_by_len: dict[int, list[str]] = defaultdict(list)
    for alias in alias_phrases:
        length = len(tokenize(alias))
        if length > 0:
            aliases_by_len[length].append(alias)
    lengths = sorted(aliases_by_len.keys(), reverse=True)

    qtext = " ".join(query_tokens)
    for length in lengths:
        if length > len(query_tokens):
            continue
        for start in range(0, len(query_tokens) - length + 1):
            end = start + length
            span = tuple(range(start, end))
            if any(_span_overlaps(span, existing) for existing in covered):
                continue
            phrase_tokens = query_tokens[start:end]
            if _is_single_token_stopword(phrase_tokens):
                # A bare function word never becomes a concept tag on its
                # own, regardless of what it happens to alias to. See
                # STOPWORDS docstring above.
                continue
            phrase = " ".join(phrase_tokens)
            records = store.exact_lookup(phrase)
            if not records:
                continue

            for record in records:
                matches.append(
                    ConceptMatch(
                        ontology=record.ontology,
                        concept_id=record.concept_id,
                        preferred_name=record.preferred_name,
                        matched_text=phrase,
                        normalized_match=phrase,
                        match_type="exact",
                        tree_numbers=record.tree_numbers,
                        qualifiers=record.qualifiers,
                        nursing_context=record.nursing_context,
                    )
                )
            covered.append((start, end))

    return matches, covered


def _span_overlaps(span: Sequence[int], other: Sequence[int]) -> bool:
    left = max(span[0], other[0])
    right = min(span[-1], other[-1])
    return left <= right


def _token_index_covered(index: int, covered_spans: Sequence[tuple[int, int]]) -> bool:
    for start, end in covered_spans:
        if start <= index < end:
            return True
    return False


def _find_stemmed_matches(
    query_tokens: Sequence[str],
    covered_spans: Sequence[tuple[int, int]],
    store: OntologyStore,
) -> list[ConceptMatch]:
    """Stem only query spans not already covered by exact matching."""
    results: list[ConceptMatch] = []
    qstems = stem_tokens(query_tokens)

    start = 0
    while start < len(query_tokens):
        if _token_index_covered(start, covered_spans):
            start += 1
            continue

        found_for_position = False
        end = start + 1
        while end <= len(query_tokens):
            span = tuple(range(start, end))
            if any(_span_overlaps(span, existing) for existing in covered_spans):
                break

            stem_phrase = " ".join(qstems[start:end])
            phrase_tokens = query_tokens[start:end]
            if stem_phrase and not _is_single_token_stopword(phrase_tokens):
                candidates = store.stem_lookup(" ".join(phrase_tokens))

                if candidates:
                    query_phrase = " ".join(query_tokens[start:end])
                    concept_ids = tuple(sorted({r.concept_id for r in candidates}))
                    ambiguous = len(concept_ids) > 1
                    reason = None
                    if ambiguous:
                        reason = "stem_collision_multiple_concepts"
                    for record in candidates:
                        results.append(
                            ConceptMatch(
                                ontology=record.ontology,
                                concept_id=record.concept_id,
                                preferred_name=record.preferred_name,
                                matched_text=query_phrase,
                                normalized_match=query_phrase,
                                match_type="stemmed",
                                ambiguous=ambiguous,
                                ambiguity_concept_ids=concept_ids if ambiguous else tuple(),
                                ambiguity_reason=reason,
                                tree_numbers=record.tree_numbers,
                                qualifiers=record.qualifiers,
                                nursing_context=record.nursing_context,
                            )
                        )
                    found_for_position = True
                    break
            end += 1

        # A stemmed match is a soft tag but still consumes the matched span so
        # we don't emit duplicate shorter stem matches from the same position.
        start += 1 if not found_for_position else max(1, end - start)

    return results


def resolve_query(query: str, store: OntologyStore) -> dict[str, Any]:
    """Resolve a raw user query into deterministic ontology concept tags."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    normalized = normalize_text(query)
    tokens = tokenize(query)
    exact_matches, covered = _find_exact_matches(tokens, store)
    stemmed_matches = _find_stemmed_matches(tokens, covered, store)

    # De-duplicate concept/span pairs while preserving deterministic ordering.
    combined: list[ConceptMatch] = []
    seen: set[tuple[Any, ...]] = set()
    for match in exact_matches + stemmed_matches:
        key = (
            match.ontology,
            match.concept_id,
            match.matched_text,
            match.match_type,
            match.ambiguous,
        )
        if key not in seen:
            seen.add(key)
            combined.append(match)

    combined.sort(
        key=lambda item: (
            0 if item.match_type == "exact" else 1,
            1 if item.ambiguous else 0,
            ONTOLOGY_PRIORITY.index(item.ontology),
            item.matched_text,
            item.concept_id,
        )
    )

    return {
        "phase": PHASE5_ONTOLOGY_VERSION,
        "query": query.strip(),
        "normalized_query": normalized,
        "concepts": [match.to_dict() for match in combined],
        "has_ambiguous_match": any(match.ambiguous for match in combined),
        "exact_match_count": sum(1 for match in combined if match.match_type == "exact"),
        "stemmed_match_count": sum(1 for match in combined if match.match_type == "stemmed"),
    }


def _chunked(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------
# MeSH loader (mesh.db SQLite: descriptors / entry_terms / tree_numbers /
# qualifiers tables).
# --------------------------------------------------------------------------

def load_mesh_sqlite(
    db_path: str | Path, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load MeSH descriptor records from the real mesh.db SQLite database.

    Each returned record represents one descriptor. Entry terms become
    synonyms, tree numbers are preserved (a descriptor may have several),
    and qualifiers are attached as separate qualifier metadata rather than
    being treated as independent concepts.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"MeSH SQLite database not found: {path}")

    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        query = "SELECT descriptor_ui, name FROM descriptors ORDER BY descriptor_ui"
        if limit is not None:
            cur.execute(query + " LIMIT ?", (limit,))
        else:
            cur.execute(query)
        descriptor_rows = cur.fetchall()
        uis = [row["descriptor_ui"] for row in descriptor_rows]
        if not uis:
            return []

        entry_terms: dict[str, list[str]] = defaultdict(list)
        tree_numbers: dict[str, list[str]] = defaultdict(list)
        qualifiers: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for batch in _chunked(uis, 500):
            placeholders = ",".join("?" for _ in batch)

            cur.execute(
                f"SELECT descriptor_ui, term FROM entry_terms "
                f"WHERE descriptor_ui IN ({placeholders})",
                tuple(batch),
            )
            for ui, term in cur.fetchall():
                entry_terms[ui].append(term)

            cur.execute(
                f"SELECT descriptor_ui, tree_number FROM tree_numbers "
                f"WHERE descriptor_ui IN ({placeholders})",
                tuple(batch),
            )
            for ui, tn in cur.fetchall():
                tree_numbers[ui].append(tn)

            cur.execute(
                f"SELECT descriptor_ui, qualifier_ui, qualifier_name FROM qualifiers "
                f"WHERE descriptor_ui IN ({placeholders})",
                tuple(batch),
            )
            for ui, qid, qname in cur.fetchall():
                qualifiers[ui].append((qid, qname))
    finally:
        conn.close()

    records: list[dict[str, Any]] = []
    for row in descriptor_rows:
        ui = row["descriptor_ui"]
        name = row["name"]
        synonyms = sorted(
            {
                term.strip()
                for term in entry_terms.get(ui, [])
                if term and term.strip() and term.strip() != name
            }
        )
        records.append(
            {
                "ontology": "mesh",
                "concept_id": ui,
                "preferred_name": name,
                "synonyms": synonyms,
                "tree_numbers": sorted(set(tree_numbers.get(ui, []))),
                "qualifiers": [
                    {"qualifier_id": qid, "qualifier_name": qname}
                    for qid, qname in sorted(set(qualifiers.get(ui, [])))
                ],
            }
        )
    return records


# --------------------------------------------------------------------------
# Lightweight, deterministic OBO parser shared by DOID and SYMP.
# --------------------------------------------------------------------------

_OBO_SYNONYM_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def parse_obo_terms(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream [Term] stanzas out of an OBO file.

    Only the fields Phase 5 needs are extracted: id, name, synonym text,
    and is_obsolete. Non-Term stanzas (Typedef, header, etc.) are skipped.
    """
    obo_path = Path(path)
    if not obo_path.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_path}")

    term: dict[str, Any] | None = None
    with obo_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n").rstrip("\r")
            stripped = line.strip()

            if stripped == "[Term]":
                if term is not None:
                    yield term
                term = {"id": None, "name": None, "synonyms": [], "is_obsolete": False}
                continue

            if stripped.startswith("[") and stripped.endswith("]"):
                # Entering a different stanza type (e.g. [Typedef]).
                if term is not None:
                    yield term
                term = None
                continue

            if term is None or not stripped or ":" not in stripped:
                continue

            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            if key == "id":
                term["id"] = value
            elif key == "name":
                term["name"] = value
            elif key == "synonym":
                match = _OBO_SYNONYM_RE.search(value)
                if match:
                    term["synonyms"].append(match.group(1))
            elif key == "is_obsolete":
                term["is_obsolete"] = value.strip().lower() == "true"

    if term is not None:
        yield term


def _load_obo_ontology(
    path: str | Path, ontology: str, limit: int | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for term in parse_obo_terms(path):
        if term.get("is_obsolete"):
            continue
        term_id = term.get("id")
        name = term.get("name")
        if not term_id or not name:
            continue
        records.append(
            {
                "ontology": ontology,
                "concept_id": term_id,
                "preferred_name": name,
                "synonyms": sorted(set(term.get("synonyms", []))),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def load_doid_obo(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load DOID (Disease Ontology) concept records. Obsolete terms are dropped."""
    return _load_obo_ontology(path, "doid", limit=limit)


def load_symp_obo(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load SYMP (Symptom Ontology) concept records. Obsolete terms are dropped."""
    return _load_obo_ontology(path, "symp", limit=limit)


# --------------------------------------------------------------------------
# SNOMED CT RF2 loader (streaming/line-oriented; never loads the huge
# Relationship file, and Relationships are not used for matching yet).
# --------------------------------------------------------------------------

_SEMANTIC_TAG_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _find_rf2_file(directory: Path, filename_prefix: str) -> Path | None:
    matches = sorted(directory.glob(f"{filename_prefix}*.txt"))
    return matches[0] if matches else None


def _iter_rf2_rows(path: Path) -> Iterator[dict[str, str]]:
    # RF2 files are plain tab-separated values with no CSV-style quoting or
    # escaping (per the RF2 spec). Real description terms can legitimately
    # contain literal double-quote characters (e.g. `Blood group antibody
    # rh"`), so csv's default QUOTE_MINIMAL misreads `"` as opening a quoted
    # field and reads forward across many lines looking for a closing quote,
    # eventually raising "field larger than field limit". QUOTE_NONE treats
    # `"` as an ordinary character, matching the real RF2 format.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            yield row


def _strip_semantic_tag(fsn: str) -> str:
    stripped = _SEMANTIC_TAG_RE.sub("", fsn).strip()
    return stripped or fsn


def load_snomed_rf2(
    directory: str | Path,
    limit: int | None = None,
    language_code: str = "en",
) -> list[dict[str, Any]]:
    """Load SNOMED CT concepts from a directory containing RF2 snapshot files.

    At minimum requires an ``sct2_Concept_Snapshot*.txt`` and an
    ``sct2_Description_Snapshot*.txt`` file in ``directory``.
    ``sct2_TextDefinition_Snapshot*.txt`` is optional and, when present, is
    used to attach a definition to each concept.

    Only active (active == "1") Concept and Description rows are used.
    Preferred names come from the active Fully Specified Name (FSN)
    description, with its trailing semantic tag stripped; if no FSN is
    present, the first active synonym is promoted to preferred_name and the
    remaining active synonyms are kept as acceptable synonyms.

    SNOMED relationships are intentionally not read here; qualifier/
    hierarchy expansion is a later Phase 5 step. The Description file is
    the largest of the required inputs and is streamed row-by-row rather
    than being loaded into memory as a whole file.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"SNOMED RF2 directory not found: {dir_path}")

    concept_file = _find_rf2_file(dir_path, "sct2_Concept_Snapshot")
    description_file = _find_rf2_file(dir_path, "sct2_Description_Snapshot")
    if concept_file is None or description_file is None:
        raise FileNotFoundError(
            "SNOMED RF2 Concept and Description snapshot files are required "
            f"in {dir_path} (looked for sct2_Concept_Snapshot*.txt and "
            "sct2_Description_Snapshot*.txt)"
        )
    textdefinition_file = _find_rf2_file(dir_path, "sct2_TextDefinition_Snapshot")

    active_concepts: set[str] = set()
    for row in _iter_rf2_rows(concept_file):
        if row.get("active") == "1" and row.get("id"):
            active_concepts.add(row["id"])

    descriptions: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"fsn": None, "synonyms": set()}
    )
    for row in _iter_rf2_rows(description_file):
        if row.get("active") != "1":
            continue
        if row.get("languageCode") != language_code:
            continue
        concept_id = row.get("conceptId")
        if concept_id not in active_concepts:
            continue
        term = (row.get("term") or "").strip()
        if not term:
            continue
        entry = descriptions[concept_id]
        if row.get("typeId") == SNOMED_FSN_TYPE_ID:
            entry["fsn"] = term
        elif row.get("typeId") == SNOMED_SYNONYM_TYPE_ID:
            entry["synonyms"].add(term)

    definitions: dict[str, str] = {}
    if textdefinition_file is not None:
        for row in _iter_rf2_rows(textdefinition_file):
            if row.get("active") != "1":
                continue
            if row.get("languageCode") != language_code:
                continue
            concept_id = row.get("conceptId")
            if not concept_id or concept_id in definitions:
                continue
            term = (row.get("term") or "").strip()
            if term:
                definitions[concept_id] = term

    records: list[dict[str, Any]] = []
    for concept_id in sorted(active_concepts):
        entry = descriptions.get(concept_id)
        if not entry:
            continue

        fsn = entry["fsn"]
        synonyms = sorted(entry["synonyms"])
        if fsn:
            preferred_name = _strip_semantic_tag(fsn)
            if fsn not in synonyms and _strip_semantic_tag(fsn) != fsn:
                # Keep the raw FSN text itself matchable too.
                synonyms = sorted(set(synonyms) | {fsn})
        elif synonyms:
            preferred_name = synonyms[0]
            synonyms = synonyms[1:]
        else:
            continue

        record: dict[str, Any] = {
            "ontology": "snomed",
            "concept_id": concept_id,
            "preferred_name": preferred_name,
            "synonyms": synonyms,
        }
        definition = definitions.get(concept_id)
        if definition:
            record["definition"] = definition
        records.append(record)

        if limit is not None and len(records) >= limit:
            break

    return records


# --------------------------------------------------------------------------
# SNOMED CT Nursing Problem List Subset loader (a nursing-relevance OVERLAY
# on existing SNOMED concepts -- not an independent ontology). Produces
# lightweight subset-membership rows meant to be passed to
# OntologyStore.apply_nursing_subset(), never raw ConceptRecord input.
# --------------------------------------------------------------------------

_NURSING_SUBSET_REQUIRED_COLUMNS = (
    "SNOMED_CID",
    "SNOMED_FSN",
    "SNOMED_CONCEPT_STATUS",
    "UMLS_CUI",
    "IS_RETIRED_FROM_SUBSET",
)


def load_snomed_nursing_subset_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load the SNOMED CT Nursing Problem List Subset CSV.

    Returns lightweight subset-membership rows -- NOT ConceptRecords -- of
    the shape expected by ``OntologyStore.apply_nursing_subset()``:

        {
            "concept_id": "<SNOMED_CID>",
            "umls_cui": "<UMLS_CUI>" | None,
            "subset_status_2017": "<SNOMED_CONCEPT_STATUS>" | None,
            "is_retired_from_subset": bool,
        }

    Notes:
      - SNOMED_FSN from the subset file is intentionally NOT used to build
        or update a concept's preferred_name: the subset is an overlay on
        top of whatever SNOMED release is loaded separately (see
        apply_nursing_subset), and the subset's own FSN text can be stale
        relative to the current SNOMED snapshot.
      - SNOMED_CONCEPT_STATUS from the subset file is HISTORICAL (as of the
        subset's 2017 publication) and must never be treated as the
        concept's current SNOMED active/inactive status -- it is preserved
        as-is under ``subset_status_2017`` precisely so callers can't
        confuse it with current status.
      - Rows with a missing/blank SNOMED_CID are skipped as malformed.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Nursing subset CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [
            col for col in _NURSING_SUBSET_REQUIRED_COLUMNS if col not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise ValueError(
                "Nursing subset CSV is missing required column(s): {}".format(
                    missing_columns
                )
            )
        for raw_row in reader:
            concept_id = (raw_row.get("SNOMED_CID") or "").strip()
            if not concept_id:
                continue

            umls_cui = (raw_row.get("UMLS_CUI") or "").strip() or None
            subset_status = (raw_row.get("SNOMED_CONCEPT_STATUS") or "").strip() or None
            is_retired = (raw_row.get("IS_RETIRED_FROM_SUBSET") or "").strip().upper() == "TRUE"

            rows.append(
                {
                    "concept_id": concept_id,
                    "umls_cui": umls_cui,
                    "subset_status_2017": subset_status,
                    "is_retired_from_subset": is_retired,
                }
            )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 5: build/load ontology concepts and resolve a query."
    )
    parser.add_argument(
        "--ontology", help="JSON ontology record file (as produced by save_json)"
    )
    parser.add_argument("--mesh-db", help="Path to a MeSH mesh.db SQLite file")
    parser.add_argument("--doid-obo", help="Path to a DOID doid.obo file")
    parser.add_argument("--symp-obo", help="Path to a SYMP symp.obo file")
    parser.add_argument(
        "--snomed-dir", help="Directory containing SNOMED CT RF2 snapshot files"
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-source record limit")
    parser.add_argument("--query", required=True, help="User query to resolve")
    return parser


def _build_store_from_args(args: argparse.Namespace) -> OntologyStore:
    store = OntologyStore()
    if args.ontology:
        store = OntologyStore.load_json(args.ontology)
    if args.mesh_db:
        store.add_records(load_mesh_sqlite(args.mesh_db, limit=args.limit))
    if args.doid_obo:
        store.add_records(load_doid_obo(args.doid_obo, limit=args.limit))
    if args.symp_obo:
        store.add_records(load_symp_obo(args.symp_obo, limit=args.limit))
    if args.snomed_dir:
        store.add_records(load_snomed_rf2(args.snomed_dir, limit=args.limit))
    if not store.records():
        raise ValueError(
            "No ontology source provided. Use --ontology and/or "
            "--mesh-db/--doid-obo/--symp-obo/--snomed-dir."
        )
    return store


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        store = _build_store_from_args(args)
        result = resolve_query(args.query, store)
    except Exception as exc:
        print("Phase 5 ontology/query resolution failed: {}".format(exc))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
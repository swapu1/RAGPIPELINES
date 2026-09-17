"""Phase 8 -- Query intent classification.

NEW module (not a modification of any existing file). Sits at the very
front of the intent-aware pipeline:

    USER QUERY
        -> INTENT CLASSIFICATION   (this module)
        -> initial retrieval (phase3, unchanged)
        -> query-conditioned intent relevance / rerank (phase8_intent_relevance)
        -> refinement loop (phase8_refinement)
        -> LLM (phase8_intent_pipeline)

Design choice: hybrid rule + optional embedding tiebreak, NOT bare
keyword/synonym matching.
--------------------------------------------------------------------------
1. PRIMARY signal is a deterministic, weighted, multi-word CUE-PATTERN
   match (regex phrases, not single keywords) per intent, in the same
   spirit as the deterministic detectors already used elsewhere in this
   project (phase3_rerank's structural detectors, phase5_ontology's
   stopword/connector guards). This keeps classification reproducible,
   dependency-light, unit-testable without a model download, and fast
   enough to run on every query at retrieval time.

   Bare single-keyword matching was explicitly rejected: a single word
   like "cause" or "risk" is highly ambiguous across intents ("what
   CAUSES anxiety" vs "risk of complications FROM the CAUSE"), whereas a
   short PHRASE ("what causes", "risk factor", "how do i prevent") is a
   much stronger, low-false-positive signal of intent -- this mirrors why
   phase5_concept_selection's Rule 1 rejects bare connector words but
   keeps them as part of longer phrases.

2. SECONDARY, OPTIONAL signal: when an embedding model is injected (the
   same SentenceTransformer-compatible object already loaded elsewhere in
   this pipeline -- never loaded here), a cosine-similarity tiebreak
   against a short exemplar phrase per intent is used ONLY to break a
   near-tie between the top two rule-based candidates, or to pick an
   intent when literally zero rule cues fired. This reuses the exact
   embed/cosine utilities already implemented in phase5_contextual.py
   (embed_unique_texts / cosine_similarity) rather than duplicating them,
   and never overrides a clear rule-based winner.

This module never calls an LLM and never mutates chunks/corpus data --
intent here is purely a property of the QUERY, computed at query time.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

try:
    # Reused, not duplicated: same embed/cosine utilities phase5_contextual
    # already implements for query<->concept cosine scoring.
    from phase5_contextual import cosine_similarity, embed_unique_texts
except ImportError:  # pragma: no cover - embedding tiebreak simply disabled
    cosine_similarity = None
    embed_unique_texts = None

PHASE8_INTENT_VERSION = "phase8_intent_classification_v1"

# --------------------------------------------------------------------------
# Taxonomy (18 clinical/nursing intents + UNKNOWN = 19).
# --------------------------------------------------------------------------
DEFINITION = "definition"
SIGNS_SYMPTOMS = "signs_symptoms"
CAUSES_ETIOLOGY = "causes_etiology"
RISK_FACTORS = "risk_factors"
ASSESSMENT = "assessment"
DIAGNOSIS = "diagnosis"
NURSING_INTERVENTION = "nursing_intervention"
TREATMENT = "treatment"
MEDICATION = "medication"
PROCEDURE = "procedure"
PREVENTION = "prevention"
COMPLICATIONS = "complications"
CONTRAINDICATIONS = "contraindications"
EDUCATION = "education"
MONITORING = "monitoring"
PROGNOSIS = "prognosis"
COMPARISON = "comparison"
INDICATIONS = "indications"
UNKNOWN = "unknown"

INTENT_TAXONOMY: tuple[str, ...] = (
    DEFINITION, SIGNS_SYMPTOMS, CAUSES_ETIOLOGY, RISK_FACTORS, ASSESSMENT,
    DIAGNOSIS, NURSING_INTERVENTION, TREATMENT, MEDICATION, PROCEDURE,
    PREVENTION, COMPLICATIONS, CONTRAINDICATIONS, EDUCATION, MONITORING,
    PROGNOSIS, COMPARISON, INDICATIONS,
)

# --------------------------------------------------------------------------
# Cue patterns: (compiled_regex, weight). Weight 2 = distinctive multi-word
# phrase, weight 1 = supporting single/short cue. Fixed, closed, hand-
# curated -- extend by adding real false-negative queries, not by
# guessing exhaustively up front (same philosophy as phase5's curated
# closed vocabularies).
# --------------------------------------------------------------------------
def _cues(*pairs: tuple[str, int]) -> tuple[tuple[re.Pattern, int], ...]:
    return tuple((re.compile(p, re.IGNORECASE), w) for p, w in pairs)


INTENT_CUES: dict[str, tuple[tuple[re.Pattern, int], ...]] = {
    DEFINITION: _cues(
        (r"\bwhat is\b", 2), (r"\bwhat does .* mean\b", 2),
        (r"\bdefine\b", 2), (r"\bdefinition of\b", 2), (r"\bmeaning of\b", 1),
        (r"\bstand for\b", 2), (r"\bacronym\b", 1),
    ),
    SIGNS_SYMPTOMS: _cues(
        (r"\bsigns and symptoms\b", 2), (r"\bwhat does .* look like\b", 2),
        (r"\bclinical (presentation|manifestations?)\b", 2),
        (r"\bfindings\b", 1), (r"\bsymptoms?\b", 1), (r"\bsigns?\b", 1),
    ),
    CAUSES_ETIOLOGY: _cues(
        (r"\bwhat causes?\b", 2), (r"\bcaused by\b", 2), (r"\betiology\b", 2),
        (r"\bwhy does\b", 1), (r"\breason for\b", 1), (r"\bcauses?\b", 1),
        (r"\bfactors are associated with\b", 2), (r"\bassociated with\b", 1),
        (r"\brelated to\b", 1),
    ),
    RISK_FACTORS: _cues(
        (r"\brisk factors?\b", 2), (r"\bwho is at risk\b", 2),
        (r"\bpredisposing factors?\b", 2), (r"\bat risk for\b", 1),
        (r"\bincrease(s)? the risk\b", 2),
    ),
    ASSESSMENT: _cues(
        (r"\bhow (should|do|is) .* assess\b", 2), (r"\bassessment\b", 2),
        (r"\bwhat should .* (check|evaluate)\b", 1), (r"\bevaluate\b", 1),
    ),
    DIAGNOSIS: _cues(
        (r"\bnursing diagnos(is|es)\b", 2), (r"\bdefining characteristics\b", 2),
        (r"\bdiagnostic criteria\b", 2), (r"\bhow is .* diagnosed\b", 2),
        (r"\bdiagnos(is|es)\b", 1),
    ),
    NURSING_INTERVENTION: _cues(
        (r"\bnursing interventions?\b", 2), (r"\bnursing (care|measures)\b", 2),
        (r"\bwhat should (the )?nurse do\b", 2), (r"\bnursing actions?\b", 1),
    ),
    TREATMENT: _cues(
        (r"\btreatment(s)? for\b", 2), (r"\bhow is .* treated\b", 2),
        (r"\bmanagement of\b", 1), (r"\btherapy\b", 1),
    ),
    MEDICATION: _cues(
        (r"\bmedications?\b", 2), (r"\bdrugs?\b", 1), (r"\bdosage\b", 2),
        (r"\bprescri(be|ption)\b", 1), (r"\badminister(ed|ing)? .* (drug|medication)\b", 1),
    ),
    PROCEDURE: _cues(
        (r"\bhow (do|to) (you |i )?(perform|do)\b", 2), (r"\bprocedure for\b", 2),
        (r"\bsteps? (to|for)\b", 1), (r"\btechnique\b", 1),
        (r"\bwhen suctioning\b", 2), (r"\bhow to suction\b", 2),
    ),
    PREVENTION: _cues(
        (r"\bhow (can|do) .* prevent\b", 2), (r"\bprevention of\b", 2),
        (r"\bprophylaxis\b", 2), (r"\bavoid(ing)?\b", 1), (r"\breduce the risk\b", 1),
        (r"\bto prevent\b", 2), (r"\bprevention\b", 1),
    ),
    COMPLICATIONS: _cues(
        (r"\bcomplications? of\b", 2), (r"\bwhat (can|could) (go wrong|happen)\b", 1),
        (r"\badverse (effects?|events?)\b", 1), (r"\bcomplications?\b", 1),
    ),
    CONTRAINDICATIONS: _cues(
        (r"\bcontraindications?\b", 2), (r"\bcontraindicated\b", 2),
        (r"\bwhen (should|not to)\b.*\b(use|give)\b", 1),
        (r"\bwho should not\b", 1),
    ),
    EDUCATION: _cues(
        (r"\bpatient education\b", 2), (r"\bteach(ing)? the patient\b", 2),
        (r"\bwhat should (the )?patient know\b", 1),
        (r"\binstructions? (should be provided|for)\b", 2),
        (r"\bcaregiver (education|instructions?)\b", 2),
    ),
    MONITORING: _cues(
        (r"\bwhat should .* monitor\b", 2), (r"\bmonitoring\b", 2),
        (r"\bwarning signs? (to watch|of)\b", 2), (r"\bwatch for\b", 1),
        (r"\bthreshold for\b", 2), (r"\bholding .* due to\b", 2),
    ),
    PROGNOSIS: _cues(
        (r"\bprognosis\b", 2), (r"\blong[- ]term outcome\b", 2),
        (r"\bexpected outcome\b", 1), (r"\brecovery time\b", 1),
    ),
    COMPARISON: _cues(
        (r"\bdifference between\b", 2), (r"\bversus\b", 2), (r"\bvs\.?\b", 2),
        (r"\bcompared? (to|with)\b", 2), (r"\bhow is .* compared\b", 2),
        (r"\bdifference\b", 1),
    ),
    INDICATIONS: _cues(
        (r"\bindications? for\b", 2), (r"\bwhen (is|to) .* (used|indicated)\b", 2),
        (r"\bappropriate for\b", 1),
    ),
}

# Short exemplar phrase per intent, used ONLY for the optional embedding
# tiebreak. Deliberately terse -- these are not enrichment terms fed to
# retrieval, just an embedding anchor for this classifier.
_INTENT_EXEMPLARS: dict[str, str] = {
    DEFINITION: "definition and meaning of a clinical term",
    SIGNS_SYMPTOMS: "signs and symptoms and clinical findings",
    CAUSES_ETIOLOGY: "causes and etiology",
    RISK_FACTORS: "risk factors and who is at risk",
    ASSESSMENT: "nursing assessment and evaluation",
    DIAGNOSIS: "nursing diagnosis and defining characteristics",
    NURSING_INTERVENTION: "nursing interventions and nursing care",
    TREATMENT: "treatment and management",
    MEDICATION: "medication and dosage",
    PROCEDURE: "clinical procedure steps and technique",
    PREVENTION: "prevention and prophylaxis",
    COMPLICATIONS: "complications and adverse effects",
    CONTRAINDICATIONS: "contraindications and when not to use",
    EDUCATION: "patient education and teaching",
    MONITORING: "monitoring and warning signs to watch",
    PROGNOSIS: "prognosis and long term outcome",
    COMPARISON: "comparison between two conditions",
    INDICATIONS: "indications for use",
}

DEFAULT_SECONDARY_RATIO = 0.5
DEFAULT_MAX_SECONDARY = 2
DEFAULT_TIEBREAK_MARGIN = 0.2  # rule scores within 20% of each other -> near-tie

# Generic-definition precedence guard: "what is" / "define" is a weak,
# highly ambiguous cue that co-fires on almost any clinical question
# ("what is the treatment for X", "what is the prognosis for X", ...).
# When DEFINITION comes out as the rule-based top score but a more
# specific clinical intent also fired (any nonzero score), the specific
# intent must win -- a generic "what is" phrasing should never mask a
# concrete treatment/diagnosis/prognosis/medication/risk-factor question.
_DEFINITION_OVERRIDE_INTENTS = frozenset(
    INTENT_TAXONOMY
) - {DEFINITION}


def _raw_intent_scores(query: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for intent, cues in INTENT_CUES.items():
        total = 0.0
        for pattern, weight in cues:
            if pattern.search(query):
                total += weight
        if total:
            scores[intent] = total
    return scores


def _embedding_tiebreak(
    query: str, candidates: Sequence[str], model: Any
) -> str | None:
    """Return the candidate intent whose exemplar phrase is most similar
    to the query, or None if embedding utilities/model are unavailable.
    """
    if model is None or cosine_similarity is None or embed_unique_texts is None:
        return None
    if not candidates:
        return None

    exemplar_texts = [_INTENT_EXEMPLARS[c] for c in candidates]
    query_map = embed_unique_texts(model, [query])
    exemplar_map = embed_unique_texts(model, exemplar_texts)

    query_key = " ".join(query.split()).strip()
    query_vec = query_map.get(query_key)
    if query_vec is None:
        return None

    best_intent = None
    best_score = float("-inf")
    for intent in candidates:
        exemplar_key = " ".join(_INTENT_EXEMPLARS[intent].split()).strip()
        vec = exemplar_map.get(exemplar_key)
        if vec is None:
            continue
        score = cosine_similarity(query_vec, vec)
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent


def classify_intent(
    query: str,
    *,
    model: Any = None,
    secondary_ratio: float = DEFAULT_SECONDARY_RATIO,
    max_secondary: int = DEFAULT_MAX_SECONDARY,
) -> dict[str, Any]:
    """Classify a raw nurse query into the Phase 8 intent taxonomy.

    Returns
    -------
    {
      "phase": PHASE8_INTENT_VERSION,
      "query": query,
      "primary_intent": str,          # one of INTENT_TAXONOMY or UNKNOWN
      "secondary_intents": [str, ...],
      "confidence": float,             # in [0, 1]
      "method": "rule" | "rule+embedding_tiebreak" | "embedding_fallback" | "none",
      "scores": {intent: raw_score, ...},  # audit trail, rule-based only
    }

    Never raises on an empty/unusual query -- returns UNKNOWN with
    confidence 0.0 instead, since intent is an auxiliary signal, not a
    hard gate.
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "phase": PHASE8_INTENT_VERSION,
            "query": query,
            "primary_intent": UNKNOWN,
            "secondary_intents": [],
            "confidence": 0.0,
            "method": "none",
            "scores": {},
        }

    scores = _raw_intent_scores(query)
    method = "rule"

    if not scores:
        # Zero rule cues fired -- try the embedding fallback across the
        # full taxonomy before giving up to UNKNOWN.
        fallback = _embedding_tiebreak(query, INTENT_TAXONOMY, model)
        if fallback is not None:
            return {
                "phase": PHASE8_INTENT_VERSION,
                "query": query,
                "primary_intent": fallback,
                "secondary_intents": [],
                "confidence": 0.3,  # low-confidence fallback, never overstate
                "method": "embedding_fallback",
                "scores": {},
            }
        return {
            "phase": PHASE8_INTENT_VERSION,
            "query": query,
            "primary_intent": UNKNOWN,
            "secondary_intents": [],
            "confidence": 0.0,
            "method": "none",
            "scores": {},
        }

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_intent, top_score = ranked[0]

    # Deterministic precedence: a generic DEFINITION win never masks a
    # more specific clinical intent that also fired (see
    # _DEFINITION_OVERRIDE_INTENTS above).
    definition_override_applied = False
    if top_intent == DEFINITION:
        override_candidates = [
            (intent, score)
            for intent, score in ranked
            if intent in _DEFINITION_OVERRIDE_INTENTS and score > 0
        ]
        if override_candidates:
            override_candidates.sort(key=lambda kv: (-kv[1], kv[0]))
            top_intent, top_score = override_candidates[0]
            definition_override_applied = True

    # Near-tie tiebreak: only when there IS a second candidate close to
    # the top rule-based score. A clear rule-based winner is never
    # second-guessed by the embedding signal. Likewise, once the
    # deterministic definition-precedence guard above has picked a
    # specific clinical intent, that decision is final -- the embedding
    # signal must not be allowed to reverse it back to DEFINITION (or
    # anything else).
    if len(ranked) > 1 and not definition_override_applied:
        second_intent, second_score = ranked[1]
        if second_score >= top_score * (1 - DEFAULT_TIEBREAK_MARGIN):
            tied = [i for i, s in ranked if s >= top_score * (1 - DEFAULT_TIEBREAK_MARGIN)]
            embed_choice = _embedding_tiebreak(query, tied, model)
            if embed_choice is not None and embed_choice != top_intent:
                top_intent = embed_choice
                top_score = scores[embed_choice]
                method = "rule+embedding_tiebreak"

    total_score = sum(scores.values())
    confidence = min(1.0, top_score / total_score) if total_score > 0 else 0.0

    secondary = [
        intent
        for intent, score in ranked
        if intent != top_intent and score >= secondary_ratio * top_score
    ][:max_secondary]

    return {
        "phase": PHASE8_INTENT_VERSION,
        "query": query,
        "primary_intent": top_intent,
        "secondary_intents": secondary,
        "confidence": round(confidence, 4),
        "method": method,
        "scores": scores,
    }

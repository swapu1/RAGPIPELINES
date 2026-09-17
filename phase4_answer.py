"""
Phase 4 — LLM Answer Generation Layer (Google Gemini)
=======================================================

Pipeline:

    user question  +  Phase 3 reranked evidence chunks
        -> build a grounding prompt (evidence + question + strict rules)
        -> Gemini generate_content
        -> parse the model's structured JSON response
        -> cross-check claimed sources against the ACTUAL supplied chunks
        -> return {"question", "answer", "sources"}

WHAT THIS MODULE DOES **NOT** DO
---------------------------------
This module performs ONLY answer generation. It does NOT:
    - run BM25, semantic/FAISS search, hybrid fusion, or reranking
      (that is Phase 2/3's job -- phase2_bm25.py, phase3_search.py,
      phase3_hybrid.py, phase3_rerank.py -- all untouched by this file)
    - perform web search
    - retrieve any additional evidence on its own
    - build agents, maintain conversation memory, or do query rewriting

It takes whatever evidence chunks it is handed (typically the output of
`phase3_rerank.run_rerank_search(...)`) and answers strictly from them.

GROUNDING CONTRACT
-------------------
The prompt sent to Gemini explicitly instructs it to:
    - use ONLY the supplied evidence excerpts
    - never invent medical facts not present in the evidence
    - never use outside knowledge, web search, or additional retrieval
    - explicitly say the evidence is insufficient when it is
    - preserve uncertainty present in the evidence rather than flattening it

Because an LLM's own claims about "which sources it used" cannot be
trusted blindly, this module treats the model's `used_chunk_ids` as a
CLAIM to verify, not a fact to pass through. Every entry in the final
"sources" list is cross-checked against the actual `reranked_chunks` this
function was called with; page numbers in the output always come from the
original chunk data (never from anything the model wrote), and any
chunk_id the model mentions that wasn't actually in the supplied evidence
is silently dropped rather than fabricated into a source.

API KEY HANDLING
------------------
The Gemini API key is read from the GEMINI_API_KEY environment variable
by default. `api_key` may be passed explicitly (e.g. for testing); it is
never hardcoded anywhere in this module.

Usage (module):
    from phase4_answer import answer_question
    result = answer_question(
        question="What are nursing interventions for decreased activity tolerance?",
        reranked_chunks=phase3_reranked_results,   # list[dict] from phase3_rerank.py
    )

Usage (CLI):
    python phase4_answer.py \
        --question "What are nursing interventions for decreased activity tolerance?" \
        --chunks-file reranked_results.json \
        --model gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

ANSWER_PHASE_VERSION = "phase4_answer_v1"

# Configurable, but a sensible current default. Override via --model /
# model_name= for any other Gemini model the caller has access to.
DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"

# --------------------------------------------------------------------------
# Grounding / system instructions sent to Gemini
# --------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """You are answering a clinical/nursing question using ONLY the evidence \
excerpts provided below. The evidence was retrieved automatically from a nursing \
textbook by an upstream retrieval pipeline (BM25 + semantic search + reranking) \
-- you did not choose it and cannot retrieve anything else.

STRICT RULES:
1. Use ONLY the supplied evidence excerpts to answer. Do not use any outside \
   medical knowledge, training data, or general facts not present in the evidence.
2. Do NOT invent, guess, or extrapolate medical facts that are not stated in the \
   evidence.
3. Do NOT perform, simulate, or claim to perform a web search or any additional \
   retrieval. Answer only from what is given below.
4. If the supplied evidence is insufficient, incomplete, or irrelevant to the \
   question, you MUST explicitly say the evidence is insufficient to answer -- \
   do not fill the gap with outside knowledge, and do not pretend the evidence \
   says something it does not say.
5. If the evidence itself is uncertain, partial, or conflicting, preserve that \
   uncertainty in your answer rather than stating something with false confidence.
6. Every claim in your answer must be traceable to one or more of the evidence \
   excerpts below, identified by their chunk_id."""

RESPONSE_FORMAT_INSTRUCTIONS = """Respond with ONLY a single JSON object (no markdown code fences, \
no extra commentary before or after it) with exactly these keys:
{
  "direct_answer": "<a direct, concise answer to the question, OR a clear statement that the evidence is insufficient to answer it>",
  "explanation": "<brief explanation / key nursing points supporting the direct answer, grounded only in the evidence above; empty string if not applicable>",
  "insufficient_evidence": <true or false>,
  "used_chunk_ids": [<chunk_id strings from the evidence above that actually support direct_answer -- empty list if none, or if insufficient_evidence is true>]
}"""


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _validate_question(question: Any) -> str:
    """Reject empty/malformed questions before doing anything else (no
    prompt building, no API key resolution, no API call).
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return question.strip()


def _validate_chunks(reranked_chunks: Any) -> list[dict[str, Any]]:
    """Validate the reranked-chunks input. `None` and `[]` are both
    treated as "no evidence" (valid, but will short-circuit to an
    insufficient-evidence answer). Anything else must be a list of dicts,
    each with a non-empty chunk_id -- fail loudly rather than silently
    dropping malformed entries, since a dropped chunk could silently
    change what evidence the model sees.
    """
    if reranked_chunks is None:
        return []
    if not isinstance(reranked_chunks, list):
        raise ValueError(
            f"reranked_chunks must be a list, got {type(reranked_chunks).__name__}"
        )

    validated: list[dict[str, Any]] = []
    for position, chunk in enumerate(reranked_chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"reranked_chunks[{position}] must be a dict, got {type(chunk).__name__}"
            )
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"reranked_chunks[{position}] is missing a valid non-empty 'chunk_id'"
            )
        validated.append(chunk)
    return validated


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def _format_page_range(chunk: dict[str, Any]) -> str:
    start = chunk.get("pdf_page_start")
    end = chunk.get("pdf_page_end")
    if start is None and end is None:
        return "page unknown"
    if start == end:
        return f"p.{start}"
    return f"pp.{start}-{end}"


def build_evidence_block(chunks: list[dict[str, Any]]) -> str:
    """Render the supplied chunks into a plain-text evidence block, each
    clearly tagged with its chunk_id (so the model can cite it back) and
    page range (so the final output can report real page numbers).
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", "")
        page_range = _format_page_range(chunk)
        section_title = chunk.get("section_title")
        text = chunk.get("text")
        text = text if isinstance(text, str) else ""

        header = f"[Evidence {i}] chunk_id={chunk_id} ({page_range}"
        if section_title:
            header += f", section: {section_title}"
        header += ")"

        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Assemble the full prompt: grounding rules + evidence + question +
    required response format.
    """
    evidence_block = build_evidence_block(chunks)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"EVIDENCE (retrieved from a nursing textbook; use ONLY this text):\n"
        f"{evidence_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"{RESPONSE_FORMAT_INSTRUCTIONS}\n"
    )


# --------------------------------------------------------------------------
# API key / client handling
# --------------------------------------------------------------------------

def resolve_api_key(api_key: str | None = None) -> str | None:
    """Resolve the Gemini API key: explicit `api_key` argument takes
    precedence (useful for testing); otherwise fall back to the
    GEMINI_API_KEY environment variable. Returns None if neither is set
    -- callers decide how to react (this module raises a clear error only
    at the point a real API call is actually about to be made).

    The key is never hardcoded anywhere in this module.
    """
    if api_key:
        return api_key
    return os.environ.get("GEMINI_API_KEY")


def _build_client(api_key: str):
    """Lazily import and construct the official google-genai client.
    Imported lazily (not at module top-level) so this module can be
    imported, and its pure functions (build_prompt, parse_model_response,
    validation, etc.) exercised in tests, even in environments where the
    google-genai package isn't installed.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "The 'google-genai' package is not installed. Install it with: "
            "pip install google-genai"
        ) from exc
    return genai.Client(api_key=api_key)


def _call_gemini(client: Any, model_name: str, prompt: str) -> str:
    """Call Gemini with structured JSON output when supported.

    Falls back to a plain generate_content call for simple test doubles.
    """
    response_schema = {
        "type": "object",
        "properties": {
            "direct_answer": {"type": "string"},
            "explanation": {"type": "string"},
            "insufficient_evidence": {"type": "boolean"},
            "used_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "direct_answer",
            "explanation",
            "insufficient_evidence",
            "used_chunk_ids",
        ],
    }

    try:
        from google.genai import types
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
    except (ImportError, TypeError, AttributeError):
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Gemini API returned an empty or invalid response")
    return text


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?|```\s*$")


def _strip_code_fences(raw_text: str) -> str:
    """Models frequently wrap JSON in ```json ... ``` even when told not
    to. Strip that defensively before parsing.
    """
    stripped = raw_text.strip()
    stripped = _CODE_FENCE_RE.sub("", stripped)
    return stripped.strip()


def parse_model_response(
    raw_text: str,
    chunks_by_id: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Parse Gemini's raw text response into (answer_text, sources).

    `chunks_by_id` maps chunk_id -> original chunk dict from the
    `reranked_chunks` this request was actually built from. This is the
    single source of truth for what counts as a valid source: any
    chunk_id the model claims to have used that is NOT a key in
    `chunks_by_id` is dropped rather than trusted, and page numbers in
    the output always come from `chunks_by_id`, never from the model's
    own text.

    Falls back gracefully (never raises) if the model's response isn't
    valid JSON: the raw text becomes the answer and sources is empty,
    since we cannot safely confirm which chunks (if any) were actually
    used in that case.
    """
    cleaned = _strip_code_fences(raw_text)

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw_text.strip(), []

    if not isinstance(parsed, dict):
        return raw_text.strip(), []

    direct_answer = parsed.get("direct_answer")
    direct_answer = direct_answer if isinstance(direct_answer, str) else ""

    explanation = parsed.get("explanation")
    explanation = explanation if isinstance(explanation, str) else ""

    used_chunk_ids = parsed.get("used_chunk_ids")
    if not isinstance(used_chunk_ids, list):
        used_chunk_ids = []

    answer_text = direct_answer.strip()
    if explanation.strip():
        if answer_text:
            answer_text = f"{answer_text}\n\nKey nursing points:\n{explanation.strip()}"
        else:
            answer_text = explanation.strip()

    if not answer_text:
        # Model returned valid JSON but no usable text in either field --
        # fall back to the raw response rather than returning an empty
        # answer.
        answer_text = raw_text.strip()

    sources: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for chunk_id in used_chunk_ids:
        if not isinstance(chunk_id, str):
            continue
        if chunk_id in seen_ids:
            continue
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            # The model claimed a chunk_id that was never part of the
            # supplied evidence -- never fabricate a source for it.
            continue
        seen_ids.add(chunk_id)
        sources.append(
            {
                "chunk_id": chunk_id,
                "pdf_page_start": chunk.get("pdf_page_start"),
                "pdf_page_end": chunk.get("pdf_page_end"),
            }
        )

    return answer_text, sources


# --------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------

def answer_question(
    question: str,
    reranked_chunks: list[dict[str, Any]] | None,
    api_key: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    client: Any = None,
) -> dict[str, Any]:
    """Generate a grounded answer to `question` using ONLY the evidence in
    `reranked_chunks` (the output of Phase 3's phase3_rerank.py).

    This function performs NO retrieval of its own: it never queries
    BM25, FAISS, or the web. It only calls Gemini once, with a prompt
    built strictly from the chunks it was handed.

    Args:
        question: the user's question. Must be a non-empty string.
        reranked_chunks: list of Phase 3 reranked result dicts (each with
            at least chunk_id, text, pdf_page_start, pdf_page_end).
            None or [] is treated as "no evidence available".
        api_key: optional explicit Gemini API key (for testing). If not
            given, falls back to the GEMINI_API_KEY environment variable.
        model_name: Gemini model to use.
        client: optional pre-built client object exposing
            `.models.generate_content(model=..., contents=...)`. Injected
            in tests to avoid real network calls / needing the SDK
            installed; if omitted, a real google-genai client is built
            from the resolved API key.

    Returns:
        {"question": str, "answer": str, "sources": list[dict]}
        where each source dict is {"chunk_id", "pdf_page_start",
        "pdf_page_end"} and corresponds to a chunk that was ACTUALLY in
        `reranked_chunks` -- never a fabricated or hallucinated chunk_id.

    Raises:
        ValueError: for a malformed question or malformed reranked_chunks,
            or a missing API key when no client was injected.
        RuntimeError: if the Gemini API call itself fails.
    """
    validated_question = _validate_question(question)
    validated_chunks = _validate_chunks(reranked_chunks)

    if not validated_chunks:
        # No evidence at all -- this is a known, deterministic case that
        # doesn't require calling the model: there is nothing it could
        # possibly ground an answer in.
        return {
            "question": validated_question,
            "answer": (
                "The supplied evidence is insufficient to answer this question: "
                "no evidence chunks were provided."
            ),
            "sources": [],
        }

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in validated_chunks}
    prompt = build_prompt(validated_question, validated_chunks)

    if client is None:
        resolved_key = resolve_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "Gemini API key not found. Set the GEMINI_API_KEY environment "
                "variable, or pass api_key explicitly."
            )
        client = _build_client(resolved_key)

    try:
        raw_text = _call_gemini(client, model_name, prompt)
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    answer_text, sources = parse_model_response(raw_text, chunks_by_id)

    return {
        "question": validated_question,
        "answer": answer_text,
        "sources": sources,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4: generate a grounded answer (Gemini) from Phase 3 "
            "reranked evidence chunks. Performs NO retrieval of its own -- "
            "supply evidence via --chunks-file (e.g. the JSON output of "
            "phase3_rerank.py)."
        )
    )
    parser.add_argument("--question", required=True, help="The user's question.")
    parser.add_argument(
        "--chunks-file",
        required=True,
        help=(
            "Path to a JSON file containing a list of Phase 3 reranked "
            "chunk dicts (e.g. output of phase3_rerank.py)."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key. Defaults to the GEMINI_API_KEY environment variable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        with open(args.chunks_file, "r", encoding="utf-8") as f:
            reranked_chunks = json.load(f)

        result = answer_question(
            question=args.question,
            reranked_chunks=reranked_chunks,
            api_key=args.api_key,
            model_name=args.model,
        )
    except Exception as exc:
        print(f"Phase 4 answer generation failed: {exc}", file=sys.stderr)
        return 1

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
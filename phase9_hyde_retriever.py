"""
Phase 9 — Hypothetical Document Embedding (HyDE) Retriever
===========================================================
Implements HyDE for clinical nursing document retrieval:
  1. Prompts LLM (Gemini 3.5 Flash-Lite) to draft a hypothetical textbook passage answering the clinical query.
  2. Embeds the hypothetical passage (blended with query) via PubMedBERT.
  3. Searches the augmented FAISS index with the hypothetical embedding.
  4. Fuses with BM25 lexical search over augmented chunks/questions.
  5. Returns mapped top-k parent chunk candidates with provenance.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase9_hyde_retriever")

DEFAULT_AUGMENTED_JSON = "chunks_augmented.json"
DEFAULT_BM25_AUGMENTED = "bm25_augmented_index.pkl"
DEFAULT_FAISS_AUGMENTED = "index_augmented.faiss"
DEFAULT_METADATA_AUGMENTED = "index_augmented_metadata.json"
DEFAULT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"

TOKEN_RE = re.compile(r"[a-z0-9]+")

HYDE_PROMPT_TEMPLATE = """You are an expert clinical nursing educator and textbook author.
Given the clinical nursing question below, write a detailed hypothetical textbook passage or clinical guide excerpt that directly and authoritatively answers the question.

Include relevant nursing terminology, standard nursing process steps (assessment, diagnosis, planning, implementation, evaluation), clinical rationale, and typical textbook phraseology.

The passage should read like an authentic section from "Ackley and Ladwig's Nursing Diagnosis Handbook".

QUESTION:
{query}

HYPOTHETICAL TEXTBOOK PASSAGE:"""


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


_SENTENCE_MODEL_CACHE: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    global _SENTENCE_MODEL_CACHE
    if _SENTENCE_MODEL_CACHE is None:
        _SENTENCE_MODEL_CACHE = SentenceTransformer(model_name)
    return _SENTENCE_MODEL_CACHE


def generate_hypothetical_document(
    query: str,
    model_name: str = DEFAULT_LLM_MODEL,
    client: Any = None,
    max_retries: int = 5,
) -> str:
    """Generate a hypothetical clinical document answering the user query using Gemini."""
    if client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        client = genai.Client(api_key=api_key)

    from google.genai import types
    prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
    cfg = types.GenerateContentConfig(temperature=0.0)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=cfg,
            )
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception as e:
            err_str = str(e)
            match = re.search(r"retry in (\d+(\.\d+)?)s", err_str)
            sleep_sec = float(match.group(1)) + 4.0 if match else (4.0 * attempt)
            logger.warning("HyDE generation attempt %d failed (%s). Sleeping %.1fs...", attempt, err_str[:60], sleep_sec)
            time.sleep(sleep_sec)

    return query


def run_hyde_search(
    query: str,
    hypothetical_doc: Optional[str] = None,
    top_k: int = 5,
    candidate_k: int = 15,
    bm25_index_path: str = DEFAULT_BM25_AUGMENTED,
    faiss_index_path: str = DEFAULT_FAISS_AUGMENTED,
    metadata_path: str = DEFAULT_METADATA_AUGMENTED,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    bm25_weight: float = 0.4,
    semantic_weight: float = 0.6,
    query_blend_weight: float = 0.3,
    client: Any = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Execute HyDE Hybrid Search:
      1. If hypothetical_doc is None, generate it.
      2. BM25 search with query tokens over augmented text.
      3. FAISS dense search with embedding of (query * blend + hypothetical_doc * (1-blend)).
      4. Fuse and return top_k candidates.
    """
    if not hypothetical_doc:
        hypothetical_doc = generate_hypothetical_document(query, client=client)

    # 1. BM25 Search
    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    bm25_cids = bm25_data["chunk_ids"]

    query_tokens = tokenize(query)
    bm25_raw = bm25.get_scores(query_tokens)
    b_min, b_max = float(np.min(bm25_raw)), float(np.max(bm25_raw))
    bm25_norm = (bm25_raw - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(bm25_raw)
    bm25_map = {bm25_cids[i]: float(bm25_norm[i]) for i in range(len(bm25_cids))}

    # 2. Dense Semantic Search with HyDE
    index = faiss.read_index(faiss_index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    entries = meta["entries"]

    embed_model = get_embedding_model(embedding_model_name)

    if query_blend_weight > 0.0:
        # Blend query embedding with hypothetical document embedding
        embs = embed_model.encode([query, hypothetical_doc], normalize_embeddings=True)
        q_vec = embs[0]
        hyde_vec = embs[1]
        search_vec = query_blend_weight * q_vec + (1.0 - query_blend_weight) * hyde_vec
        # Re-normalize
        norm = np.linalg.norm(search_vec)
        if norm > 0:
            search_vec = search_vec / norm
    else:
        search_vec = embed_model.encode([hypothetical_doc], normalize_embeddings=True)[0]

    search_vec = np.ascontiguousarray(search_vec.reshape(1, -1), dtype=np.float32)

    num_search = min(candidate_k * 4, index.ntotal)
    D, I = index.search(search_vec, num_search)

    faiss_score_map: Dict[str, float] = {}
    matched_qs_map: Dict[str, List[Dict[str, Any]]] = {}

    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(entries):
            continue
        entry = entries[idx]
        cid = entry["chunk_id"]
        norm_score = max(0.0, float(score))
        if cid not in faiss_score_map or norm_score > faiss_score_map[cid]:
            faiss_score_map[cid] = norm_score

        if entry.get("type") == "synthetic_question":
            if cid not in matched_qs_map:
                matched_qs_map[cid] = []
            matched_qs_map[cid].append({
                "intent": entry.get("intent", "clinical"),
                "question": entry.get("text", ""),
                "score": float(score),
            })

    # 3. Hybrid Fusion
    all_cids = set(bm25_map.keys()) | set(faiss_score_map.keys())
    scored = []
    for cid in all_cids:
        s_b = bm25_map.get(cid, 0.0)
        s_s = faiss_score_map.get(cid, 0.0)
        fused = bm25_weight * s_b + semantic_weight * s_s
        scored.append({
            "chunk_id": cid,
            "fused_score": fused,
            "bm25_score": s_b,
            "semantic_score": s_s,
            "matched_questions": matched_qs_map.get(cid, []),
        })

    scored.sort(key=lambda x: x["fused_score"], reverse=True)
    return scored[:top_k], hypothetical_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HyDE Search")
    parser.add_argument("query", nargs="?", default="What are the five steps of the nursing process, and what is the purpose of each step?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results, hypo = run_hyde_search(args.query, top_k=args.top_k)
    print("Hypothetical Document:\n" + "-" * 50)
    print(hypo)
    print("-" * 50 + "\nTop Retrieved Chunks:")
    for rank, r in enumerate(results, 1):
        print(f"[{rank}] {r['chunk_id']} | Score: {r['fused_score']:.4f} (BM25: {r['bm25_score']:.4f}, FAISS: {r['semantic_score']:.4f})")

"""
App Engine: Unified Multi-Pipeline Search & Grounded Generation Layer
====================================================================
Provides retrieval and answer generation for:
  1. Plain BM25 + FAISS (Baseline)
  2. Baseline + Concept Tagging (SNOMED CT, MeSH, DOID, SYMP)
  3. Baseline + Intent Tagging (Phase 8 Reranking)
  4. Baseline + Contextual Embeddings (Phase 9 Augmented Chunks)
  5. Baseline + Contextual Embeddings + HyDE
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Suppress noisy logs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger("app_engine")

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

# Import pipeline modules
import phase5_ontology as p5
from phase3_concept import ConceptIndex, extract_query_concept_keys, score_candidates
from phase8_intent_classification import classify_intent
from phase8_intent_relevance import rerank_with_intent
from phase9_abstract_summary_extractor import get_chunk_abstract_context
from phase9_hyde_retriever import generate_hypothetical_document
from phase4_answer import answer_question

# Pipelines Metadata
PIPELINES = {
    "baseline": {
        "id": "baseline",
        "name": "1. Plain BM25 + FAISS (Baseline)",
        "badge": "Baseline",
        "description": "Bi-modal hybrid fusion (40% BM25 + 60% PubMedBERT Dense Cosine) over textbook corpus.",
        "icon": "⚖️"
    },
    "concept": {
        "id": "concept",
        "name": "2. Baseline + Concept Tagging",
        "badge": "Concepts",
        "description": "Tri-modal hybrid fusion combining BM25, PubMedBERT, and 428k SNOMED/MeSH/DOID/SYMP ontology concept weights.",
        "icon": "🏷️"
    },
    "intent": {
        "id": "intent",
        "name": "3. Baseline + Intent Tagging",
        "badge": "Phase 8 Intent",
        "description": "Rule-based 18-class clinical intent classification + intent-relevance multiplicative reranking and topical entity anchoring.",
        "icon": "🎯"
    },
    "augmented": {
        "id": "augmented",
        "name": "4. Contextual Embeddings (Augmented: Context + Fake Queries)",
        "badge": "Augmented (Context + Synthetic Qs)",
        "description": "Multi-vector index: Chunk text + Contextual abstract enrichment + Synthetic intent queries (fake clinical questions) embedded together with PubMedBERT.",
        "icon": "🧠"
    },
    "hyde": {
        "id": "hyde",
        "name": "5. Contextual Embeddings + HyDE (Hypothetical Doc + Augmented)",
        "badge": "HyDE + Augmented",
        "description": "Zero-shot hypothetical clinical textbook passage generation (HyDE) + Blended PubMedBERT search over the Augmented index (Context + Synthetic Queries).",
        "icon": "⚡"
    }
}


class PipelineEngine:
    _instance: Optional["PipelineEngine"] = None

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent
        else:
            self.base_dir = Path(base_dir)
        logger.info("Initializing Pipeline Engine from %s...", self.base_dir)
        t0 = time.time()

        # 1. Load Baseline Chunks (pages 22-51, 46 chunks)
        chunks_path = self.base_dir / "chunks.json"
        with open(chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
            self.baseline_chunks = [
                c for c in all_chunks
                if 22 <= c.get("pdf_page_start", 0) and c.get("pdf_page_end", 0) <= 51
            ]
        self.baseline_cids = [c["chunk_id"] for c in self.baseline_chunks]
        self.baseline_chunk_map = {c["chunk_id"]: c for c in self.baseline_chunks}

        # 2. Build In-Memory Baseline BM25 strictly over the 46 chunks
        logger.info("Building BM25 for %d baseline chunks...", len(self.baseline_chunks))
        corpus_tokens = [tokenize(c["text"]) for c in self.baseline_chunks]
        self.baseline_bm25 = BM25Okapi(corpus_tokens)
        self.baseline_bm25_cids = self.baseline_cids

        # 3. Load PubMedBERT SentenceTransformer
        logger.info("Loading PubMedBERT embedding model...")
        self.embed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

        # 4. Build/Load Baseline FAISS Index over the 46 chunks
        logger.info("Building FAISS index over %d baseline chunks...", len(self.baseline_chunks))
        texts = [c["text"] for c in self.baseline_chunks]
        embs = self.embed_model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        embs = np.ascontiguousarray(embs, dtype=np.float32)
        self.baseline_faiss = faiss.IndexFlatIP(embs.shape[1])
        self.baseline_faiss.add(embs)

        # 5. Load Concept Tagging Index & Ontology Store
        concept_salient_path = self.base_dir / "chunk_concepts_salient.json"
        if concept_salient_path.exists():
            self.concept_index = ConceptIndex.from_path(str(concept_salient_path))
        else:
            self.concept_index = None

        ontology_path = self.base_dir / "ontology_store.pkl"
        lite_path = self.base_dir / "ontology_store_lite.pkl"
        if ontology_path.exists():
            with open(ontology_path, "rb") as f:
                self.ontology_store = pickle.load(f)
        elif lite_path.exists():
            with open(lite_path, "rb") as f:
                self.ontology_store = pickle.load(f)
        else:
            self.ontology_store = None

        # 6. Load Phase 9 Augmented Assets
        aug_json_path = self.base_dir / "chunks_augmented.json"
        with open(aug_json_path, "r", encoding="utf-8") as f:
            self.augmented_chunks = json.load(f)
        self.augmented_chunk_map = {c["chunk_id"]: c for c in self.augmented_chunks}

        aug_bm25_path = self.base_dir / "bm25_augmented_index.pkl"
        with open(aug_bm25_path, "rb") as f:
            aug_bm25_data = pickle.load(f)
            self.aug_bm25 = aug_bm25_data["bm25"]
            self.aug_bm25_cids = aug_bm25_data["chunk_ids"]

        aug_faiss_path = self.base_dir / "index_augmented.faiss"
        self.aug_faiss = faiss.read_index(str(aug_faiss_path))

        aug_meta_path = self.base_dir / "index_augmented_metadata.json"
        if aug_meta_path.exists():
            with open(aug_meta_path, "r", encoding="utf-8") as f:
                self.aug_metadata = json.load(f).get("entries", [])
        else:
            self.aug_metadata = []

        self.abstract_db_path = str(self.base_dir / "context_chunks_abstract.db")

        logger.info("Pipeline Engine fully loaded in %.2fs!", time.time() - t0)

    @classmethod
    def get_instance(cls) -> "PipelineEngine":
        if cls._instance is None:
            cls._instance = PipelineEngine()
        return cls._instance

    # -------------------------------------------------------------------------
    # Fast Concept Resolution
    # -------------------------------------------------------------------------
    def fast_resolve_query(self, query: str) -> Dict[str, Any]:
        if not self.ontology_store:
            return {"query": query, "concepts": []}

        tokens = p5.tokenize(query)
        n = len(tokens)
        matches = []
        covered = []

        for length in range(min(n, 6), 0, -1):
            for start in range(n - length + 1):
                end = start + length
                span = tuple(range(start, end))
                if any(p5._span_overlaps(span, existing) for existing in covered):
                    continue
                phrase_tokens = tokens[start:end]
                if p5._is_single_token_stopword(phrase_tokens):
                    continue
                phrase = " ".join(phrase_tokens)
                records = self.ontology_store.exact_lookup(phrase)
                if records:
                    for record in records:
                        matches.append({
                            "ontology": record.ontology,
                            "concept_id": record.concept_id,
                            "preferred_name": record.preferred_name,
                            "matched_text": phrase,
                            "match_type": "exact",
                            "ambiguous": False,
                        })
                    covered.append((start, end))

        for start in range(n):
            if any(start in range(s, e) for s, e in covered):
                continue
            phrase_tokens = [tokens[start]]
            if p5._is_single_token_stopword(phrase_tokens):
                continue
            phrase = tokens[start]
            records = self.ontology_store.stem_lookup(phrase)
            if records:
                for record in records:
                    matches.append({
                        "ontology": record.ontology,
                        "concept_id": record.concept_id,
                        "preferred_name": record.preferred_name,
                        "matched_text": phrase,
                        "match_type": "stemmed",
                        "ambiguous": False,
                    })
                covered.append((start, start + 1))

        seen = set()
        deduped = []
        for m in matches:
            key = (m["ontology"], m["concept_id"])
            if key not in seen:
                seen.add(key)
                deduped.append(m)

        return {"concepts": deduped, "tokens": tokens}

    # -------------------------------------------------------------------------
    # Pipeline 1: Plain Baseline (BM25 + FAISS)
    # -------------------------------------------------------------------------
    def search_baseline(self, query: str, top_k: int = 5, bm25_weight: float = 0.4, semantic_weight: float = 0.6) -> List[Dict[str, Any]]:
        q_toks = tokenize(query)
        b_scores = self.baseline_bm25.get_scores(q_toks)
        b_min, b_max = float(np.min(b_scores)), float(np.max(b_scores))
        b_norm = (b_scores - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(b_scores)
        bm25_map = {self.baseline_bm25_cids[i]: float(b_norm[i]) for i in range(len(self.baseline_bm25_cids))}

        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
        D, I = self.baseline_faiss.search(q_emb, len(self.baseline_chunks))

        sem_score_map = {}
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.baseline_cids):
                cid = self.baseline_cids[idx]
                sem_score_map[cid] = max(0.0, float(score))

        fused = []
        for cid in self.baseline_cids:
            s_b = bm25_map.get(cid, 0.0)
            s_s = sem_score_map.get(cid, 0.0)
            score = bm25_weight * s_b + semantic_weight * s_s
            chunk_data = self.baseline_chunk_map.get(cid, {})
            fused.append({
                "chunk_id": cid,
                "text": chunk_data.get("text", ""),
                "score": score,
                "fused_score": score,
                "bm25_score": s_b,
                "semantic_score": s_s,
                "concept_score": 0.0,
                "pdf_page_start": chunk_data.get("pdf_page_start"),
                "pdf_page_end": chunk_data.get("pdf_page_end"),
                "section_title": chunk_data.get("section_title", "General"),
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

    # -------------------------------------------------------------------------
    # Pipeline 2: Baseline + Concept Tagging
    # -------------------------------------------------------------------------
    def search_concept(self, query: str, top_k: int = 5, concept_weight: float = 0.10) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        base_results = self.search_baseline(query, top_k=len(self.baseline_cids))
        base_map = {r["chunk_id"]: r for r in base_results}

        res = self.fast_resolve_query(query)
        q_concept_weights = extract_query_concept_keys(res)
        c_scores, matched_c = score_candidates(q_concept_weights, self.baseline_cids, self.concept_index)
        c_min = min(c_scores.values()) if c_scores else 0.0
        c_max = max(c_scores.values()) if c_scores else 0.0
        c_norm = {
            cid: (c_scores.get(cid, 0.0) - c_min) / (c_max - c_min) if c_max > c_min else 0.0
            for cid in self.baseline_cids
        }

        alpha = 1.0 - concept_weight
        blended = []
        for cid in self.baseline_cids:
            item = dict(base_map[cid])
            s_concept = c_norm.get(cid, 0.0)
            fused = alpha * item["fused_score"] + concept_weight * s_concept
            item["score"] = fused
            item["fused_score"] = fused
            item["concept_score"] = s_concept
            blended.append(item)

        blended.sort(key=lambda x: x["score"], reverse=True)
        return blended[:top_k], {"detected_concepts": res["concepts"], "concept_weights": q_concept_weights}

    # -------------------------------------------------------------------------
    # Pipeline 3: Baseline + Intent Tagging (Phase 8 Reranking)
    # -------------------------------------------------------------------------
    def search_intent(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        candidates = self.search_baseline(query, top_k=len(self.baseline_chunks))
        intent_info = classify_intent(query)

        formatted = []
        for c in candidates:
            formatted.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "final_score": c["fused_score"],
                "bm25_score": c["bm25_score"],
                "semantic_score": c["semantic_score"],
                "pdf_page_start": c.get("pdf_page_start"),
                "pdf_page_end": c.get("pdf_page_end"),
                "section_title": c.get("section_title"),
            })

        reranked = rerank_with_intent(query, intent_info, formatted)

        final_results = []
        for r in reranked[:top_k]:
            final_results.append({
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": r.get("combined_score", r.get("final_score", 0.0)),
                "fused_score": r.get("combined_score", r.get("final_score", 0.0)),
                "bm25_score": r.get("bm25_score", 0.0),
                "semantic_score": r.get("semantic_score", 0.0),
                "intent_relevance": r.get("intent_relevance", 0.0),
                "pdf_page_start": r.get("pdf_page_start"),
                "pdf_page_end": r.get("pdf_page_end"),
                "section_title": r.get("section_title", "General"),
            })

        return final_results, intent_info

    # -------------------------------------------------------------------------
    # Pipeline 4: Baseline + Contextual Embeddings (Phase 9 Augmented)
    # -------------------------------------------------------------------------
    def search_augmented(self, query: str, top_k: int = 5, candidate_k: int = 15, bm25_weight: float = 0.4, semantic_weight: float = 0.6) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        bm25_raw = self.aug_bm25.get_scores(query_tokens)
        b_min, b_max = float(np.min(bm25_raw)), float(np.max(bm25_raw))
        bm25_norm = (bm25_raw - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(bm25_raw)
        bm25_map = {self.aug_bm25_cids[i]: float(bm25_norm[i]) for i in range(len(self.aug_bm25_cids))}

        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
        num_search = min(candidate_k * 4, self.aug_faiss.ntotal)
        D, I = self.aug_faiss.search(q_emb, num_search)

        faiss_score_map: Dict[str, float] = {}
        matched_qs_map: Dict[str, List[Dict[str, Any]]] = {}

        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.aug_metadata):
                entry = self.aug_metadata[idx]
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

        all_cids = set(bm25_map.keys()) | set(faiss_score_map.keys())
        scored = []
        for cid in all_cids:
            s_b = bm25_map.get(cid, 0.0)
            s_s = faiss_score_map.get(cid, 0.0)
            fused = bm25_weight * s_b + semantic_weight * s_s
            c_data = self.augmented_chunk_map.get(cid, {})
            scored.append({
                "chunk_id": cid,
                "text": c_data.get("text", ""),
                "score": fused,
                "fused_score": fused,
                "bm25_score": s_b,
                "semantic_score": s_s,
                "matched_questions": matched_qs_map.get(cid, []),
                "pdf_page_start": c_data.get("pdf_page_start"),
                "pdf_page_end": c_data.get("pdf_page_end"),
                "section_title": c_data.get("section_title", "General"),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # -------------------------------------------------------------------------
    # Pipeline 5: Baseline + Contextual Embeddings + HyDE
    # -------------------------------------------------------------------------
    def search_hyde(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 15,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
        query_blend_weight: float = 0.3,
        hypo_doc: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        if not hypo_doc:
            hypo_doc = generate_hypothetical_document(query)

        query_tokens = tokenize(query)
        bm25_raw = self.aug_bm25.get_scores(query_tokens)
        b_min, b_max = float(np.min(bm25_raw)), float(np.max(bm25_raw))
        bm25_norm = (bm25_raw - b_min) / (b_max - b_min) if b_max > b_min else np.zeros_like(bm25_raw)
        bm25_map = {self.aug_bm25_cids[i]: float(bm25_norm[i]) for i in range(len(self.aug_bm25_cids))}

        if query_blend_weight > 0.0:
            embs = self.embed_model.encode([query, hypo_doc], normalize_embeddings=True)
            search_vec = query_blend_weight * embs[0] + (1.0 - query_blend_weight) * embs[1]
            norm = np.linalg.norm(search_vec)
            if norm > 0:
                search_vec = search_vec / norm
        else:
            search_vec = self.embed_model.encode([hypo_doc], normalize_embeddings=True)[0]

        search_vec = np.ascontiguousarray(search_vec.reshape(1, -1), dtype=np.float32)
        num_search = min(candidate_k * 4, self.aug_faiss.ntotal)
        D, I = self.aug_faiss.search(search_vec, num_search)

        faiss_score_map: Dict[str, float] = {}
        matched_qs_map: Dict[str, List[Dict[str, Any]]] = {}

        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.aug_metadata):
                entry = self.aug_metadata[idx]
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

        all_cids = set(bm25_map.keys()) | set(faiss_score_map.keys())
        scored = []
        for cid in all_cids:
            s_b = bm25_map.get(cid, 0.0)
            s_s = faiss_score_map.get(cid, 0.0)
            fused = bm25_weight * s_b + semantic_weight * s_s
            c_data = self.augmented_chunk_map.get(cid, {})
            scored.append({
                "chunk_id": cid,
                "text": c_data.get("text", ""),
                "score": fused,
                "fused_score": fused,
                "bm25_score": s_b,
                "semantic_score": s_s,
                "matched_questions": matched_qs_map.get(cid, []),
                "pdf_page_start": c_data.get("pdf_page_start"),
                "pdf_page_end": c_data.get("pdf_page_end"),
                "section_title": c_data.get("section_title", "General"),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k], hypo_doc

    # -------------------------------------------------------------------------
    # Answer Generation Orchestration
    # -------------------------------------------------------------------------
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        use_abstract_enrichment: bool = False,
        model_name: str = "gemini-3.5-flash-lite",
    ) -> Dict[str, Any]:
        if not retrieved_chunks:
            return {
                "direct_answer": "No relevant evidence chunks were found.",
                "explanation": "",
                "sources": [],
                "raw_response": "",
            }

        if use_abstract_enrichment:
            enriched_chunks = []
            for r in retrieved_chunks:
                item = dict(r)
                cid = item["chunk_id"]
                actx = get_chunk_abstract_context(cid, db_path=self.abstract_db_path)
                if actx:
                    item["abstract_context"] = actx["formatted_abstract_context"]
                enriched_chunks.append(item)
            return answer_question(
                question=query,
                reranked_chunks=enriched_chunks,
                model_name=model_name,
            )
        else:
            return answer_question(
                question=query,
                reranked_chunks=retrieved_chunks,
                model_name=model_name,
            )

    # -------------------------------------------------------------------------
    # Unified Query Entrypoint
    # -------------------------------------------------------------------------
    def execute_query(
        self,
        query: str,
        pipeline_id: str,
        top_k: int = 5,
        generate_llm_answer: bool = True,
        model_name: str = "gemini-3.5-flash-lite",
    ) -> Dict[str, Any]:
        t_start = time.time()
        extra_meta: Dict[str, Any] = {}
        use_abstract_enrichment = False

        if pipeline_id == "baseline":
            retrieved = self.search_baseline(query, top_k=top_k)
        elif pipeline_id == "concept":
            retrieved, extra_meta = self.search_concept(query, top_k=top_k)
        elif pipeline_id == "intent":
            retrieved, extra_meta = self.search_intent(query, top_k=top_k)
        elif pipeline_id == "augmented":
            retrieved = self.search_augmented(query, top_k=top_k)
            use_abstract_enrichment = True
        elif pipeline_id == "hyde":
            retrieved, hypo_doc = self.search_hyde(query, top_k=top_k)
            extra_meta["hypothetical_document"] = hypo_doc
            use_abstract_enrichment = True
        else:
            raise ValueError(f"Unknown pipeline ID: {pipeline_id}")

        retrieval_ms = (time.time() - t_start) * 1000

        answer_data = {}
        gen_ms = 0.0
        if generate_llm_answer:
            t_gen_start = time.time()
            answer_data = self.generate_answer(
                query=query,
                retrieved_chunks=retrieved,
                use_abstract_enrichment=use_abstract_enrichment,
                model_name=model_name,
            )
            gen_ms = (time.time() - t_gen_start) * 1000

        total_ms = (time.time() - t_start) * 1000

        return {
            "pipeline_id": pipeline_id,
            "pipeline_name": PIPELINES[pipeline_id]["name"],
            "query": query,
            "retrieved_chunks": retrieved,
            "extra_meta": extra_meta,
            "answer_data": answer_data,
            "timing": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": gen_ms,
                "total_ms": total_ms,
            }
        }

# Clinical Decision-Support RAG: Multi-Pipeline Architecture & Evaluation

An evidence-grounded Retrieval-Augmented Generation (RAG) system developed for clinical nursing using *Ackley and Ladwig's Nursing Diagnosis Handbook*. 

This project benchmarks and integrates **5 distinct retrieval paradigms** into a unified **Streamlit localhost application**, demonstrating how progressive architectural additions (Ontological Concepts, Intent Reranking, Contextual Embeddings, and HyDE) optimize retrieval precision, eliminate medical hallucinations, and reduce LLM token spending.

---

## 🏛️ The 5 Core Pipeline Architectures

1. **Pipeline 1: Baseline (Plain BM25 + FAISS Dense)**
   - Bi-modal hybrid fusion (40% BM25 Okapi lexical + 60% PubMedBERT cosine similarity).
   - Domain Encoder: `pritamdeka/S-PubMedBert-MS-MARCO` (768d).

2. **Pipeline 2: Baseline + Concept Tagging**
   - Tri-modal hybrid fusion combining BM25, PubMedBERT, and biomedical ontologies.
   - Leverages **428k concepts** across SNOMED CT, MeSH 2024, DOID, and SYMP with fast O(1) hash resolution.

3. **Pipeline 3: Baseline + Intent Tagging (Phase 8)**
   - Rule-based 18-class clinical nursing intent classification.
   - Applies multiplicative score gating and topical entity anchoring to penalize off-topic diagnostic entries.

4. **Pipeline 4: Contextual Embeddings (Augmented: Context + Synthetic Queries)**
   - Multi-vector indexing: Original chunk text + structural abstract enrichment (`context_chunks_abstract.db`) + synthetic clinical intent queries.
   - Surfaces relevant parent chunks even under complex phrasing or severe lexical mismatch.

5. **Pipeline 5: Contextual Embeddings + HyDE (Hypothetical Document Embeddings)**
   - Zero-shot LLM prompting to draft an authentic textbook excerpt answering the clinical query.
   - Dense blended query-hypothetical embedding searched over the augmented multi-vector index.

---

## 📊 Benchmark Evaluation Results (30 Clinical Questions)

| Metric | 1. Baseline | 2. + Concepts | 3. + Intent | 4. Contextual (Fake Qs) | 5. HyDE + Aug |
|---|:---:|:---:|:---:|:---:|:---:|
| **MRR** | 0.4639 | 0.4444 | **0.4722** | 0.5978 | **0.6000** |
| **MAP** | 0.3323 | 0.2951 | **0.3536** | 0.5008 | **0.5083** |
| **nDCG@5** | 0.4021 | 0.3662 | **0.4240** | 0.5853 | **0.5867** |
| **Recall@3** | 0.3222 | 0.2944 | **0.3667** | 0.5500 | **0.5500** |
| **Recall@5** | 0.4611 | 0.4222 | **0.4889** | **0.7222** | 0.7111 |
| **Precision@1** | 0.4000 | 0.3333 | 0.4000 | **0.4667** | **0.4667** |
| **Context Sufficiency** | 60.0% | — | 63.3% | **86.7%** | 83.3% |
| **Noise Ratio** | 80.0% | — | 78.7% | **72.7%** | 73.3% |
| **Faithfulness** | 97.8% | — | 98.3% | **100.0%** | **100.0%** |
| **Citation Precision** | 100.0% | — | 100.0% | **100.0%** | **100.0%** |
| **Citation Coverage** | 52.2% | — | 57.2% | **79.4%** | 75.0% |

---

## 📁 Repository Structure

```
├── app.py                                # Streamlit interactive localhost comparison app
├── app_engine.py                         # Unified search engine & orchestrator for all 5 pipelines
├── phase3_hybrid.py                      # Baseline hybrid fusion (BM25 + Dense)
├── phase3_concept.py                     # Tri-modal concept index & overlap scoring
├── phase5_ontology.py                    # Medical ontology resolution (SNOMED, MeSH, DOID, SYMP)
├── phase8_intent_classification.py       # Clinical intent classifier (18-class taxonomy)
├── phase8_intent_relevance.py            # Intent-relevance multiplicative reranker & entity anchoring
├── phase9_hyde_retriever.py              # HyDE hypothetical document generator & blended retriever
├── phase9_abstract_summary_extractor.py  # Structural preceding/following abstract context enrichment
├── phase4_answer.py                      # Grounded Gemini 3.5 Flash-Lite generation & citation verification
├── nlp_rag_metrics.py                    # Full metrics evaluation harness (ROUGE, BLEU, Faithfulness, etc.)
│
├── pipeline_metric_results.pdf           # Publication-ready single-page comparative benchmark PDF
├── clinical_ai_internship_report.tex     # Comprehensive Overleaf LaTeX technical report
│
├── eval_queries_30_baseline.json         # 30 standardized clinical benchmark queries with gold chunk IDs
├── baseline_30_eval_results.json         # Evaluation results for Pipeline 1
├── unified_concept_intent_evaluation_results.json # Evaluation results for Pipeline 2
├── phase8_30_eval_results.json           # Evaluation results for Pipeline 3
├── augmented_30_eval_results.json        # Evaluation results for Pipeline 4
└── hyde_30_eval_results.json             # Evaluation results for Pipeline 5
```

---

## 🚀 Running the Streamlit App

1. Install dependencies:
   ```bash
   pip install streamlit pandas faiss-cpu rank-bm25 sentence-transformers google-genai
   ```
2. Set your Google Gemini API key:
   ```bash
   set GEMINI_API_KEY="your_api_key_here"
   ```
3. Launch the dashboard:
   ```bash
   python -m streamlit run app.py
   ```
4. Navigate to `http://localhost:8501`.

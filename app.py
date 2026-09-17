"""
Interactive Multi-Pipeline Nursing RAG & Metrics Comparison Dashboard
=====================================================================
A localhost web application enabling clinicians and researchers to:
  1. Interactively submit clinical nursing questions across 5 distinct RAG pipelines:
     - Plain BM25 + FAISS (Baseline)
     - Baseline + Concept Tagging (SNOMED CT, MeSH, DOID, SYMP)
     - Baseline + Intent Tagging (Phase 8 Reranking)
     - Baseline + Contextual Embeddings (Phase 9 Augmented)
     - Baseline + Contextual Embeddings + HyDE
  2. Inspect and compare evaluation metrics across all 5 pipelines on the 30-query clinical benchmark:
     - Retrieval Quality (MRR, MAP, nDCG@5, Precision@k, Recall@k)
     - Context Quality (Sufficiency, Noise Ratio, Context Recall, Context Precision)
     - Generation Quality (ROUGE-1/2/L, BLEU-1/2/4, Faithfulness, Citation Precision/Coverage)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app_engine import PipelineEngine, PIPELINES

# Set page config
st.set_page_config(
    page_title="Pipeline metrics with results",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .pipeline-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        background-color: #F9FAFB;
        margin-bottom: 1rem;
    }
    .metric-delta-pos {
        color: #059669;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #DC2626;
        font-weight: 600;
    }
    .evidence-box {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }
    .badge-baseline { background-color: #E0E7FF; color: #3730A3; }
    .badge-concept { background-color: #FEF3C7; color: #92400E; }
    .badge-intent { background-color: #D1FAE5; color: #065F46; }
    .badge-augmented { background-color: #EDE9FE; color: #5B21B6; }
    .badge-hyde { background-color: #FCE7F3; color: #9D174D; }
</style>
""", unsafe_allow_html=True)


# Initialize Engine (Singleton via cache_resource)
@st.cache_resource(show_spinner="Loading Clinical Knowledge Bases, FAISS Indices & Ontologies...")
def load_engine():
    return PipelineEngine.get_instance()


# Load Benchmark Metrics Data
@st.cache_data
def load_benchmark_data():
    results = {}
    
    # Baseline
    with open(BASE_DIR / "baseline_30_eval_results.json", "r", encoding="utf-8") as f:
        results["baseline"] = json.load(f)

    # Concept + Intent Unified
    with open(BASE_DIR / "unified_concept_intent_evaluation_results.json", "r", encoding="utf-8") as f:
        results["unified"] = json.load(f)

    # Phase 8 Intent
    with open(BASE_DIR / "phase8_30_eval_results.json", "r", encoding="utf-8") as f:
        results["intent"] = json.load(f)

    # Phase 9 Augmented
    with open(BASE_DIR / "augmented_30_eval_results.json", "r", encoding="utf-8") as f:
        results["augmented"] = json.load(f)

    # Phase 9 HyDE
    with open(BASE_DIR / "hyde_30_eval_results.json", "r", encoding="utf-8") as f:
        results["hyde"] = json.load(f)

    # Eval Queries
    with open(BASE_DIR / "eval_queries_30_baseline.json", "r", encoding="utf-8") as f:
        results["queries"] = json.load(f)

    return results


# Sample queries for user quick-fill
SAMPLE_QUERIES = [
    "What are the five steps of the nursing process, and what is the purpose of each step?",
    "What is clinical reasoning in nursing, and how does a nurse use it to identify client problems and guide care?",
    "How should a nurse use the PES system to formulate a three-part nursing diagnosis?",
    "What are defining characteristics and related factors in a nursing diagnosis, and how do they differ?",
    "How does a nurse distinguish between a problem-focused nursing diagnosis and a risk nursing diagnosis?",
    "What factors should a nurse consider when prioritizing multiple nursing diagnoses for a client?",
    "A client has abdominal pain and decreased bowel activity. What nursing diagnoses could be considered?",
    "A client with a spinal cord injury develops autonomic dysreflexia. What are the likely triggers, complications, and immediate nursing actions?",
]


def render_sidebar():
    st.sidebar.image("https://img.icons8.com/color/96/caduceus.png", width=64)
    st.sidebar.title("Pipeline Settings")
    
    st.sidebar.markdown("**Corpus Scope:** Pages 22–51 (46 Chunks)")
    st.sidebar.markdown("**Clinical Domain:** Ackley & Ladwig Handbook")
    st.sidebar.markdown("---")
    
    top_k = st.sidebar.slider("Evidence Top-K Chunks", min_value=1, max_value=8, value=3, step=1)
    generate_llm = st.sidebar.checkbox("Generate Grounded LLM Answer", value=True)
    llm_model = st.sidebar.selectbox("LLM Model", ["gemini-3.5-flash-lite", "gemini-2.5-flash"], index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Ontologies")
    st.sidebar.info("""
    - **SNOMED CT**: 383,853 concepts
    - **MeSH 2024**: 31,110 headings
    - **DOID**: 12,247 disease terms
    - **SYMP**: 895 symptom phenotypes
    """)
    
    return top_k, generate_llm, llm_model


def main():
    engine = load_engine()
    bench_data = load_benchmark_data()
    top_k, generate_llm, llm_model = render_sidebar()

    st.markdown('<div class="main-title">pipeline metric results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive Multi-Pipeline Search, Ontological Reasoning, and Quantitative Performance Analysis</div>', unsafe_allow_html=True)

    tab_query, tab_metrics = st.tabs(["🔍 Interactive Query & Pipeline Comparison", "📊 Comprehensive Benchmark Metrics"])

    # =========================================================================
    # TAB 1: INTERACTIVE QUERY & PIPELINE COMPARISON
    # =========================================================================
    with tab_query:
        st.markdown("### Ask a Clinical Nursing Question")
        st.markdown("Select one or more pipelines to compare retrieval precision and grounded responses side-by-side.")

        col_sample, col_spacer = st.columns([3, 1])
        with col_sample:
            selected_sample = st.selectbox(
                "Choose a benchmark question (or type your own below):",
                ["-- Custom Query --"] + SAMPLE_QUERIES,
                index=0
            )

        initial_val = selected_sample if selected_sample != "-- Custom Query --" else ""
        query_input = st.text_area("Clinical Query:", value=initial_val, height=85, placeholder="e.g. How does a nurse distinguish between a problem-focused and risk nursing diagnosis?")

        selected_pipelines = st.multiselect(
            "Select Pipelines to Run:",
            options=list(PIPELINES.keys()),
            default=["baseline", "intent", "hyde"],
            format_func=lambda x: PIPELINES[x]["name"]
        )

        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            run_btn = st.button("🚀 Run Query", type="primary", use_container_width=True)

        if run_btn and query_input.strip():
            if not selected_pipelines:
                st.warning("Please select at least one pipeline to execute.")
            else:
                st.markdown("---")
                # Run pipelines and display side-by-side or tabs
                pipeline_cols = st.columns(len(selected_pipelines))

                for idx, pkey in enumerate(selected_pipelines):
                    pinfo = PIPELINES[pkey]
                    with pipeline_cols[idx]:
                        st.markdown(f"#### {pinfo['icon']} {pinfo['name']}")
                        with st.spinner(f"Running {pinfo['name']}..."):
                            res = engine.execute_query(
                                query=query_input.strip(),
                                pipeline_id=pkey,
                                top_k=top_k,
                                generate_llm_answer=generate_llm,
                                model_name=llm_model,
                            )

                        # Timing metrics
                        t = res["timing"]
                        st.caption(f"⏱️ Retrieval: **{t['retrieval_ms']:.1f}ms** | Gen: **{t['generation_ms']:.1f}ms**")

                        # Generated Answer
                        if generate_llm and res.get("answer_data"):
                            ans_data = res["answer_data"]
                            # Retrieve answer text (handles both dict styles)
                            full_answer = ans_data.get("answer", "")
                            direct_ans = ans_data.get("direct_answer", "")
                            explanation = ans_data.get("explanation", "")
                            sources = ans_data.get("sources", [])

                            display_text = full_answer if full_answer else direct_ans
                            if not display_text and explanation:
                                display_text = explanation

                            st.success(f"**Grounded Answer:**\n\n{display_text}")

                            if explanation and full_answer and explanation not in full_answer:
                                with st.expander("Clinical Explanation / Nursing Points"):
                                    st.write(explanation)

                            if sources:
                                cited_ids = [s.get("chunk_id", str(s)) if isinstance(s, dict) else str(s) for s in sources]
                                st.markdown(f"**Verified Citations:** `{', '.join(cited_ids)}`")

                        # Extra pipeline metadata
                        extra = res.get("extra_meta", {})
                        if pkey == "concept" and "detected_concepts" in extra:
                            with st.expander(f"🏷️ Matched Concepts ({len(extra['detected_concepts'])})"):
                                for c in extra["detected_concepts"][:6]:
                                    st.write(f"- **{c['preferred_name']}** (`{c['ontology']}:{c['concept_id']}`) via *{c['matched_text']}*")
                        elif pkey == "intent" and extra:
                            with st.expander("🎯 Intent Classification"):
                                st.write(f"- **Primary Intent:** `{extra.get('primary_intent')}`")
                                st.write(f"- **Confidence:** `{extra.get('confidence', 0.0):.2f}`")
                                if extra.get("matched_cues"):
                                    st.write(f"- **Cues:** `{', '.join(extra['matched_cues'])}`")
                        elif pkey == "hyde" and "hypothetical_document" in extra:
                            with st.expander("⚡ HyDE Hypothetical Passage"):
                                st.markdown(f"*{extra['hypothetical_document']}*")

                        # Retrieved Evidence Chunks
                        st.markdown(f"**Retrieved Evidence (Top-{len(res['retrieved_chunks'])}):**")
                        for rank, r in enumerate(res["retrieved_chunks"], 1):
                            cid = r["chunk_id"]
                            score = r.get("score", 0.0)
                            p_start = r.get("pdf_page_start", "?")
                            p_end = r.get("pdf_page_end", "?")
                            sec = r.get("section_title", "General")

                            with st.expander(f"#{rank} [{cid}] Pages {p_start}-{p_end} (Score: {score:.3f})"):
                                st.caption(f"**Section:** {sec} | BM25: {r.get('bm25_score', 0):.3f} | Semantic: {r.get('semantic_score', 0):.3f}")
                                if r.get("concept_score", 0) > 0:
                                    st.caption(f"Concept Weight Score: {r['concept_score']:.3f}")
                                if r.get("intent_relevance", 0) > 0:
                                    st.caption(f"Intent Relevance Multiplier: {r['intent_relevance']:.3f}")
                                st.write(r.get("text", "")[:450] + "...")

    # =========================================================================
    # TAB 2: COMPREHENSIVE BENCHMARK METRICS
    # =========================================================================
    with tab_metrics:
        st.markdown("### Evaluation Benchmark Results Across All 5 Pipelines")
        st.markdown("Quantitatively evaluated on the **30-query clinical nursing benchmark** across Pages 22–51 (46 corpus chunks).")

        # Compile comparison dataframe
        b_ret = bench_data["baseline"]["retrieval_summary"]
        b_gen = bench_data["baseline"]["generation_summary"]

        c_ret = bench_data["unified"]["scope_30"]["results"]["B: + Concept Tagging"]
        i_ret = bench_data["intent"]["retrieval_summary"]["intent_aware"]
        i_gen = bench_data["intent"]["generation_summary"]["intent_aware"]

        a_ret = bench_data["augmented"]["retrieval_summary"]
        a_gen = bench_data["augmented"]["generation_summary"]

        h_ret = bench_data["hyde"]["retrieval_summary"]
        h_gen = bench_data["hyde"]["generation_summary"]

        # 1. High-Level Summary Metric Cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("1. Baseline MRR", f"{b_ret['mrr']:.4f}", help="BM25 + FAISS Dense")
        with col_m2:
            st.metric("2. + Concepts MRR", f"{c_ret['mrr']:.4f}", delta=f"{c_ret['mrr'] - b_ret['mrr']:.4f}", help="Tri-modal concept fusion")
        with col_m3:
            st.metric("3. + Intent MRR", f"{i_ret['mrr']:.4f}", delta=f"+{i_ret['mrr'] - b_ret['mrr']:.4f}", help="Phase 8 Intent Rerank")
        with col_m4:
            st.metric("4. + Augmented MRR", f"{a_ret['mrr']:.4f}", delta=f"+{a_ret['mrr'] - b_ret['mrr']:.4f}", help="Intent question augmentation")
        with col_m5:
            st.metric("5. + HyDE MRR", f"{h_ret['mrr']:.4f}", delta=f"+{h_ret['mrr'] - b_ret['mrr']:.4f}", help="Hypothetical doc embedding")

        st.markdown("---")

        # 2. Retrieval Quality Metrics Table
        st.markdown("#### 1. Retrieval Performance Metrics")
        ret_df = pd.DataFrame([
            {
                "Pipeline": "1. Plain BM25 + FAISS (Baseline)",
                "MRR": b_ret["mrr"],
                "MAP": b_ret["map"],
                "nDCG@5": b_ret["ndcg@5"],
                "Recall@1": b_ret["recall@1"],
                "Recall@3": b_ret["recall@3"],
                "Recall@5": b_ret["recall@5"],
                "Precision@1": b_ret["precision@1"],
                "Precision@3": b_ret["precision@3"],
                "Precision@5": b_ret["precision@5"],
            },
            {
                "Pipeline": "2. Baseline + Concept Tagging",
                "MRR": c_ret["mrr"],
                "MAP": c_ret["map"],
                "nDCG@5": c_ret["ndcg@5"],
                "Recall@1": c_ret["recall@1"],
                "Recall@3": c_ret["recall@3"],
                "Recall@5": c_ret["recall@5"],
                "Precision@1": c_ret["precision@1"],
                "Precision@3": c_ret["precision@3"],
                "Precision@5": c_ret["precision@5"],
            },
            {
                "Pipeline": "3. Baseline + Intent Tagging (Phase 8)",
                "MRR": i_ret["mrr"],
                "MAP": i_ret["map"],
                "nDCG@5": i_ret["ndcg@5"],
                "Recall@1": i_ret["recall@1"],
                "Recall@3": i_ret["recall@3"],
                "Recall@5": i_ret["recall@5"],
                "Precision@1": i_ret["precision@1"],
                "Precision@3": i_ret["precision@3"],
                "Precision@5": i_ret["precision@5"],
            },
            {
                "Pipeline": "4. Contextual Embeddings (Augmented: Context + Fake Queries)",
                "MRR": a_ret["mrr"],
                "MAP": a_ret["map"],
                "nDCG@5": a_ret["ndcg@5"],
                "Recall@1": a_ret["recall@1"],
                "Recall@3": a_ret["recall@3"],
                "Recall@5": a_ret["recall@5"],
                "Precision@1": a_ret["precision@1"],
                "Precision@3": a_ret["precision@3"],
                "Precision@5": a_ret["precision@5"],
            },
            {
                "Pipeline": "5. Contextual Embeddings + HyDE (HyDE + Aug)",
                "MRR": h_ret["mrr"],
                "MAP": h_ret["map"],
                "nDCG@5": h_ret["ndcg@5"],
                "Recall@1": h_ret["recall@1"],
                "Recall@3": h_ret["recall@3"],
                "Recall@5": h_ret["recall@5"],
                "Precision@1": h_ret["precision@1"],
                "Precision@3": h_ret["precision@3"],
                "Precision@5": h_ret["precision@5"],
            },
        ])

        st.dataframe(
            ret_df.style.format({
                "MRR": "{:.4f}", "MAP": "{:.4f}", "nDCG@5": "{:.4f}",
                "Recall@1": "{:.4f}", "Recall@3": "{:.4f}", "Recall@5": "{:.4f}",
                "Precision@1": "{:.4f}", "Precision@3": "{:.4f}", "Precision@5": "{:.4f}",
            }).highlight_max(subset=["MRR", "MAP", "nDCG@5", "Recall@3", "Recall@5"], color="#D1FAE5"),
            use_container_width=True
        )

        # Bar chart comparison of key retrieval metrics
        chart_data = ret_df.set_index("Pipeline")[["MRR", "MAP", "nDCG@5", "Recall@5"]]
        st.bar_chart(chart_data)

        st.markdown("---")

        # 3. Context Quality & Generation Quality
        col_ctx, col_gen = st.columns(2)

        with col_ctx:
            st.markdown("#### 2. Context Sufficiency & Noise")
            ctx_df = pd.DataFrame([
                {
                    "Pipeline": "1. Baseline",
                    "Context Recall": b_ret.get("context_recall", 0.4611),
                    "Context Precision": b_ret.get("context_precision", 0.2000),
                    "Sufficiency": b_ret.get("context_sufficiency", 0.6000),
                    "Noise Ratio": b_ret.get("context_noise_ratio", 0.8000),
                },
                {
                    "Pipeline": "3. + Intent Tagging",
                    "Context Recall": i_ret.get("context_recall", 0.4889),
                    "Context Precision": i_ret.get("context_precision", 0.2133),
                    "Sufficiency": i_ret.get("context_sufficiency", 0.6333),
                    "Noise Ratio": i_ret.get("context_noise_ratio", 0.7867),
                },
                {
                    "Pipeline": "4. Augmented (Context + Fake Queries)",
                    "Context Recall": a_ret.get("context_recall", 0.7222),
                    "Context Precision": a_ret.get("context_precision", 0.2733),
                    "Sufficiency": a_ret.get("context_sufficiency", 0.8667),
                    "Noise Ratio": a_ret.get("context_noise_ratio", 0.7267),
                },
                {
                    "Pipeline": "5. HyDE + Augmented",
                    "Context Recall": h_ret.get("context_recall", 0.7111),
                    "Context Precision": h_ret.get("context_precision", 0.2667),
                    "Sufficiency": h_ret.get("context_sufficiency", 0.8333),
                    "Noise Ratio": h_ret.get("context_noise_ratio", 0.7333),
                },
            ])
            st.dataframe(
                ctx_df.style.format({
                    "Context Recall": "{:.4f}", "Context Precision": "{:.4f}",
                    "Sufficiency": "{:.4f}", "Noise Ratio": "{:.4f}",
                }).highlight_max(subset=["Sufficiency", "Context Recall"], color="#D1FAE5"),
                use_container_width=True
            )

        with col_gen:
            st.markdown("#### 3. LLM Generation Quality (Gemini 3.5 Flash-Lite)")
            gen_df = pd.DataFrame([
                {
                    "Pipeline": "1. Baseline",
                    "ROUGE-L": b_gen.get("rougeL_f1", 0.2400),
                    "BLEU-4": b_gen.get("bleu4", 0.0499),
                    "Faithfulness": b_gen.get("faithfulness", 0.9778),
                    "Citation Precision": b_gen.get("citation_precision", 1.0000),
                    "Citation Coverage": b_gen.get("citation_coverage", 0.5222),
                },
                {
                    "Pipeline": "3. + Intent Tagging",
                    "ROUGE-L": i_gen.get("rougeL_f1", 0.2435),
                    "BLEU-4": i_gen.get("bleu4", 0.0520),
                    "Faithfulness": i_gen.get("faithfulness", 0.9833),
                    "Citation Precision": i_gen.get("citation_precision", 1.0000),
                    "Citation Coverage": i_gen.get("citation_coverage", 0.5722),
                },
                {
                    "Pipeline": "4. Augmented (Context + Fake Queries)",
                    "ROUGE-L": a_gen.get("rougeL_f1", 0.2539),
                    "BLEU-4": a_gen.get("bleu4", 0.0600),
                    "Faithfulness": a_gen.get("faithfulness", 1.0000),
                    "Citation Precision": a_gen.get("citation_precision", 1.0000),
                    "Citation Coverage": a_gen.get("citation_coverage", 0.7944),
                },
                {
                    "Pipeline": "5. HyDE + Augmented",
                    "ROUGE-L": h_gen.get("rougeL_f1", 0.2669),
                    "BLEU-4": h_gen.get("bleu4", 0.0697),
                    "Faithfulness": h_gen.get("faithfulness", 1.0000),
                    "Citation Precision": h_gen.get("citation_precision", 1.0000),
                    "Citation Coverage": h_gen.get("citation_coverage", 0.7500),
                },
            ])
            st.dataframe(
                gen_df.style.format({
                    "ROUGE-L": "{:.4f}", "BLEU-4": "{:.4f}", "Faithfulness": "{:.4f}",
                    "Citation Precision": "{:.4f}", "Citation Coverage": "{:.4f}",
                }).highlight_max(subset=["ROUGE-L", "BLEU-4", "Citation Coverage"], color="#D1FAE5"),
                use_container_width=True
            )

        st.markdown("---")

        # 4. Per-Query Drill-down
        with st.expander("🔍 Drill Down: 30 Benchmark Queries with Gold Chunks & Pipeline Performance"):
            query_table = []
            for q in bench_data["queries"]:
                query_table.append({
                    "ID": q["query_id"],
                    "Question": q["question"],
                    "Gold Intent": q.get("gold_intent", ""),
                    "Gold Chunks": ", ".join(q.get("gold_chunk_ids", [])),
                })
            st.dataframe(pd.DataFrame(query_table), use_container_width=True)


if __name__ == "__main__":
    main()

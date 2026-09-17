"""
Phase 9 — Isolated Abstract Summary Context Extractor
=====================================================
Extracts strictly grounded 1-2 sentence abstract summaries of preceding
and following sections using Google Gemini 3.5 Flash-Lite.

Scope: First 30 content pages of the nursing handbook (PDF pages 22 to 51),
excluding front matter (author bios, contributor lists, Roman-numeral prefaces).
Stored in an isolated database: context_chunks_abstract.db.
Zero modification to existing production code or prior context databases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase9_abstract_summary")

DEFAULT_CHUNKS_PATH = "chunks.json"
DEFAULT_BASE_DB = "context_chunks.db"
DEFAULT_ABSTRACT_DB = "context_chunks_abstract.db"
DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"

# First 30 content pages: starts at Section I (page 22) through page 51
PAGE_START = 22
PAGE_END = 51

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a clinical document summarizer. "
    "Summarize the following clinical nursing textbook excerpt in 1-2 concise, strictly factual sentences. "
    "STRICT RULES:\n"
    "1. Include ONLY facts explicitly stated in the excerpt below.\n"
    "2. Do NOT extrapolate, add outside background medical knowledge, or infer unmentioned details.\n"
    "3. Focus on the core clinical concepts, assessment findings, or nursing interventions discussed.\n"
    "4. Return ONLY the 1-2 sentence summary, no preamble, no commentary, no quotes."
)


def filter_target_chunks(chunks: List[Dict[str, Any]], page_start: int = PAGE_START, page_end: int = PAGE_END) -> List[Dict[str, Any]]:
    """Filter chunks to the first 30 content pages (pages 22 to 51)."""
    target = []
    for c in chunks:
        p_start = c.get("pdf_page_start", 0)
        p_end = c.get("pdf_page_end", 0)
        if p_start >= page_start and p_end <= page_end:
            target.append(c)
    return target


def format_abstract_context_block(
    section: str,
    heading_path: List[str],
    prev_section_title: Optional[str],
    prev_abstract_summary: Optional[str],
    next_section_title: Optional[str],
    next_abstract_summary: Optional[str],
) -> str:
    """Format the enriched structural + abstract summary context block."""
    hpath = " > ".join(heading_path) if heading_path else section
    lines = [
        "[STRUCTURAL CONTEXT]",
        f"Current section: {section}",
        f"Hierarchy: {hpath}",
    ]

    if prev_abstract_summary and prev_abstract_summary.strip():
        title = prev_section_title or "Preceding Section"
        lines.append(f"\nPreceding Section Abstract Summary ({title}):\n\"{prev_abstract_summary.strip()}\"")

    if next_abstract_summary and next_abstract_summary.strip():
        title = next_section_title or "Following Section"
        lines.append(f"\nFollowing Section Abstract Summary ({title}):\n\"{next_abstract_summary.strip()}\"")

    return "\n".join(lines)


def summarize_excerpt(
    text: str,
    client: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    max_retries: int = 5,
) -> str:
    """Generate a strictly grounded 1-2 sentence abstract summary using Gemini."""
    if not text or not text.strip():
        return ""

    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=SUMMARIZATION_SYSTEM_PROMPT,
        temperature=0.0,  # Deterministic, grounded output
    )

    prompt = f"Excerpt to summarize:\n\"\"\"\n{text.strip()}\n\"\"\""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            summary = getattr(response, "text", "") or ""
            return summary.strip().strip('"')
        except Exception as e:
            err_str = str(e)
            match = re.search(r"retry in (\d+(\.\d+)?)s", err_str)
            sleep_sec = float(match.group(1)) + 3.0 if match else (4.0 * attempt)
            logger.warning("Summarization attempt %d failed (%s). Sleeping %.1fs...", attempt, err_str[:80], sleep_sec)
            time.sleep(sleep_sec)

    logger.error("Failed to summarize excerpt after %d attempts.", max_retries)
    return ""


def init_abstract_db(db_path: str | Path = DEFAULT_ABSTRACT_DB) -> sqlite3.Connection:
    """Initialize SQLite database for abstract context summaries."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks_abstract_context (
            chunk_id TEXT PRIMARY KEY,
            original_text TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            section TEXT,
            heading_path TEXT,
            prev_chunk_id TEXT,
            prev_section_title TEXT,
            prev_abstract_summary TEXT,
            next_chunk_id TEXT,
            next_section_title TEXT,
            next_abstract_summary TEXT,
            formatted_abstract_context TEXT
        )
        """
    )
    conn.commit()
    return conn


def get_chunk_abstract_context(chunk_id: str, db_path: str = DEFAULT_ABSTRACT_DB) -> Optional[Dict[str, Any]]:
    """Retrieve the abstract context record for a given chunk."""
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM chunks_abstract_context WHERE chunk_id = ?
        """,
        (chunk_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def build_abstract_summary_database(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    base_db_path: str = DEFAULT_BASE_DB,
    out_db_path: str = DEFAULT_ABSTRACT_DB,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Build context_chunks_abstract.db for the 46 chunks in the first 30 content pages."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to generate abstract summaries.")

    from google import genai
    client = genai.Client(api_key=api_key)

    logger.info("Loading chunk corpus from %s", chunks_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    chunk_map = {c["chunk_id"]: c for c in all_chunks}
    target_chunks = filter_target_chunks(all_chunks, PAGE_START, PAGE_END)
    total_target = len(target_chunks)
    logger.info("Filtered %d target chunks in pages %d-%d.", total_target, PAGE_START, PAGE_END)

    # Connect to structural base DB for accurate headings
    base_conn = sqlite3.connect(base_db_path) if Path(base_db_path).exists() else None
    base_cur = base_conn.cursor() if base_conn else None

    # Connect to target abstract DB
    out_conn = init_abstract_db(out_db_path)
    out_cur = out_conn.cursor()

    # Check already completed chunks to support resuming
    out_cur.execute("SELECT chunk_id FROM chunks_abstract_context")
    completed_ids = {row[0] for row in out_cur.fetchall()}
    logger.info("%d chunks already completed in %s.", len(completed_ids), out_db_path)

    # Cache for chunk summaries so neighbor chunks don't summarize the same text multiple times
    summary_cache: Dict[str, str] = {}

    processed_count = 0
    for idx, chunk in enumerate(target_chunks, 1):
        cid = chunk["chunk_id"]
        if cid in completed_ids:
            continue

        p_start = chunk.get("pdf_page_start", 0)
        p_end = chunk.get("pdf_page_end", 0)
        orig_text = chunk.get("text", "")

        # Get structural metadata
        section = chunk.get("section_title") or "General"
        hpath = [section]
        if base_cur:
            base_cur.execute("SELECT section, heading_path FROM chunks_context WHERE chunk_id = ?", (cid,))
            brow = base_cur.fetchone()
            if brow:
                section = brow[0] or section
                try:
                    hpath = json.loads(brow[1]) if brow[1] else [section]
                except Exception:
                    hpath = [section]

        # 1. Preceding neighbor
        prev_cid = chunk.get("previous_chunk_id")
        prev_chunk = chunk_map.get(prev_cid) if prev_cid else None
        prev_title = prev_chunk.get("section_title") if prev_chunk else None
        prev_summary = ""
        if prev_chunk:
            if prev_cid in summary_cache:
                prev_summary = summary_cache[prev_cid]
            else:
                logger.info("[%02d/%02d] Summarizing preceding chunk %s...", idx, total_target, prev_cid)
                prev_summary = summarize_excerpt(prev_chunk.get("text", ""), client=client, model_name=model_name)
                summary_cache[prev_cid] = prev_summary
                time.sleep(1.5)  # gentle pacing

        # 2. Following neighbor
        next_cid = chunk.get("next_chunk_id")
        next_chunk = chunk_map.get(next_cid) if next_cid else None
        next_title = next_chunk.get("section_title") if next_chunk else None
        next_summary = ""
        if next_chunk:
            if next_cid in summary_cache:
                next_summary = summary_cache[next_cid]
            else:
                logger.info("[%02d/%02d] Summarizing following chunk %s...", idx, total_target, next_cid)
                next_summary = summarize_excerpt(next_chunk.get("text", ""), client=client, model_name=model_name)
                summary_cache[next_cid] = next_summary
                time.sleep(1.5)  # gentle pacing

        formatted_block = format_abstract_context_block(
            section=section,
            heading_path=hpath,
            prev_section_title=prev_title,
            prev_abstract_summary=prev_summary,
            next_section_title=next_title,
            next_abstract_summary=next_summary,
        )

        out_cur.execute(
            """
            INSERT OR REPLACE INTO chunks_abstract_context (
                chunk_id, original_text, page_start, page_end,
                section, heading_path,
                prev_chunk_id, prev_section_title, prev_abstract_summary,
                next_chunk_id, next_section_title, next_abstract_summary,
                formatted_abstract_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                orig_text,
                p_start,
                p_end,
                section,
                json.dumps(hpath, ensure_ascii=False),
                prev_cid,
                prev_title,
                prev_summary,
                next_cid,
                next_title,
                next_summary,
                formatted_block,
            ),
        )
        out_conn.commit()
        processed_count += 1
        logger.info("[%02d/%02d] Stored abstract context for %s", idx, total_target, cid)

    if base_conn:
        base_conn.close()
    out_conn.close()

    logger.info("Successfully built abstract context database with %d chunks at %s.", total_target, out_db_path)
    return {"total_target": total_target, "newly_processed": processed_count, "out_db": out_db_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 Abstract Context Summarizer")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--base-db", default=DEFAULT_BASE_DB)
    parser.add_argument("--out-db", default=DEFAULT_ABSTRACT_DB)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    build_abstract_summary_database(
        chunks_path=args.chunks,
        base_db_path=args.base_db,
        out_db_path=args.out_db,
        model_name=args.model,
    )

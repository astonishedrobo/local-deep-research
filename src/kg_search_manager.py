#!/usr/bin/env python3
"""
knowledge_graph_searcher.py

Drop-in module implementing KnowledgeGraphSearcher with plain-markdown
post-processing of GraphRAG/DRIFT SearchResult responses to replace
numeric citation placeholders (e.g., "Reports (1)", "Sources (0)")
with human-readable labels and a "### References" block.

Usage:
    from knowledge_graph_searcher import get_searcher, async_query_example
    # or call async_query_example() directly for a short demo
"""

import os
import re
import asyncio
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import tiktoken
from dotenv import load_dotenv

from graphrag.config.enums import ModelType
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.drift_search.drift_context import (
    DRIFTSearchContextBuilder,
)
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.query.structured_search.drift_search.state import QueryState
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# --- Load environment and basic config (adjust as needed) ---
CWD = Path.cwd()
dotenv_path = CWD / ".env"
load_dotenv(dotenv_path)

API_KEY = os.environ.get("GRAPHRAG_API_KEY")
LLM_MODEL = os.environ.get("GRAPHRAG_LLM_MODEL")
EMBED_MODEL = os.environ.get("GRAPHRAG_EMBEDDING_MODEL")


# ---------------- Citation extraction + formatter (plain markdown) ----------------
def _extract_reports_map(context_data: dict) -> dict:
    """Return {int_id: human_title} from any 'reports' DataFrame(s) in context_data."""
    reports_map: Dict[int, str] = {}
    if not context_data:
        return reports_map
    for ctx in context_data.values():
        rpt_df = ctx.get("reports")
        if rpt_df is None:
            continue
        # expect a pandas DataFrame
        for _, row in rpt_df.iterrows():
            # robust id extraction
            rid = None
            try:
                # row can be a Series/dict-like
                rid = row.get("id") if hasattr(row, "get") else None
            except Exception:
                rid = None
            try:
                rid = int(rid)
            except Exception:
                rid = int(row.name)
            # title candidates
            title = None
            for k in ("title", "name", "headline"):
                if k in row and row[k]:
                    title = str(row[k]).strip()
                    break
            if not title and "content" in row and row["content"]:
                content = str(row["content"]).strip().replace("\n", " ")
                title = content[:120] + "..."
            reports_map[rid] = title or f"Report {rid}"
    return reports_map


def _extract_sources_map(context_data: dict) -> dict:
    """Return {int_id: human_title} parsed from 'sources' text-blobs in context_data."""
    sources_map: Dict[int, str] = {}
    if not context_data:
        return sources_map
    for ctx in context_data.values():
        src_df = ctx.get("sources")
        if src_df is None:
            continue
        for _, row in src_df.iterrows():
            sid = None
            try:
                sid = row.get("id") if hasattr(row, "get") else None
            except Exception:
                sid = None
            try:
                sid = int(sid)
            except Exception:
                sid = int(row.name)
            tb = row.get("text") if "text" in row else None
            human = None
            if tb:
                tb = str(tb)
                m = re.search(r"title:\s*([^\n\r]+)", tb, re.IGNORECASE)
                if m:
                    human = m.group(1).strip()
                else:
                    # take first non-empty line as fallback
                    for line in tb.splitlines():
                        s = line.strip()
                        if s:
                            human = s[:120]
                            break
            sources_map[sid] = human or f"Source {sid}"
    return sources_map


def format_drift_response_plain(resp: Any) -> str:
    """
    Input: object with .response (str) and .context_data (dict).
    Output: markdown string with inline human labels and a '### References' block.
    """
    raw = getattr(resp, "response", "") or ""
    ctx = getattr(resp, "context_data", None) or {}

    reports_map = _extract_reports_map(ctx)
    sources_map = _extract_sources_map(ctx)

    maps_by_label = {
        "Reports": reports_map,
        "Sources": sources_map,
        # extend with "TextUnits", "Entities" if you implement extractors
    }

    pattern = re.compile(r"\b(Reports|Sources|TextUnits|Entities)\s*\(\s*(\d+)\s*\)")
    footnotes = []
    seen = {}

    def replace(m):
        label = m.group(1)
        idx = int(m.group(2))
        text = maps_by_label.get(label, {}).get(idx, f"{label} {idx}")
        key = (label, idx)
        if key not in seen:
            seen[key] = len(seen) + 1
            footnotes.append((seen[key], text, label, idx))
        # inline human title and bracketed numeric ref: "Title [1]"
        return f"{text} [{seen[key]}]"

    formatted = pattern.sub(replace, raw)

    if footnotes:
        formatted += "\n\n### References\n"
        for num, text, label, idx in footnotes:
            formatted += f"{num}. {text} ({label} id={idx})\n"

    return formatted


# ---------------- KnowledgeGraphSearcher class ----------------
class KnowledgeGraphSearcher:
    """
    Encapsulates loading a GraphRAG project and performing DRIFT searches.
    The search() method returns a plain-markdown string with readable citations.
    """

    def __init__(self, project_name: str, community_level: int = 2):
        self.project_name = project_name
        self.community_level = community_level
        self.project_dir = CWD / self.project_name
        self.input_dir = self.project_dir / "output"
        self.lancedb_uri = str(self.input_dir / "lancedb")
        self.drift_searcher: DRIFTSearch | None = None
        print(f"✅ KnowledgeGraphSearcher created for '{self.project_name}'. Will initialize lazily on first search.")

    async def _initialize(self) -> None:
        """Load data and initialize the DRIFT searcher (expensive, run once per instance)."""
        print(f"🚀 Initializing DRIFT searcher for '{self.project_name}'...")

        # load dataframes
        entity_df = pd.read_parquet(self.input_dir / "entities.parquet")
        community_df = pd.read_parquet(self.input_dir / "communities.parquet")
        relationship_df = pd.read_parquet(self.input_dir / "relationships.parquet")
        text_unit_df = pd.read_parquet(self.input_dir / "text_units.parquet")
        report_df = pd.read_parquet(self.input_dir / "community_reports.parquet")

        # build indexer adapters
        entities = read_indexer_entities(entity_df, community_df, self.community_level)
        relationships = read_indexer_relationships(relationship_df)
        text_units = read_indexer_text_units(text_unit_df)
        reports = read_indexer_reports(
            report_df, community_df, self.community_level, content_embedding_col="full_content_embeddings"
        )

        # vector stores
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=self.lancedb_uri)

        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=self.lancedb_uri)
        read_indexer_report_embeddings(reports, full_content_embedding_store)

        # models
        chat_config = LanguageModelConfig(api_key=API_KEY, type=ModelType.OpenAIChat, model=LLM_MODEL, max_retries=20)
        embed_config = LanguageModelConfig(api_key=API_KEY, type=ModelType.OpenAIEmbedding, model=EMBED_MODEL, max_retries=20)

        model_manager = ModelManager()
        chat_model = model_manager.get_or_create_chat_model(
            name=f"{self.project_name}_chat", model_type=ModelType.OpenAIChat, config=chat_config
        )
        text_embedder = model_manager.get_or_create_embedding_model(
            name=f"{self.project_name}_embed", model_type=ModelType.OpenAIEmbedding, config=embed_config
        )
        token_encoder = tiktoken.encoding_for_model(LLM_MODEL)

        drift_params = DRIFTSearchConfig(temperature=0, max_tokens=12_000, primer_folds=1, drift_k_followups=3, n_depth=3, n=1)

        context_builder = DRIFTSearchContextBuilder(
            model=chat_model,
            text_embedder=text_embedder,
            entities=entities,
            relationships=relationships,
            reports=reports,
            entity_text_embeddings=description_embedding_store,
            text_units=text_units,
            token_encoder=token_encoder,
            config=drift_params,
        )

        self.drift_searcher = DRIFTSearch(model=chat_model, context_builder=context_builder, token_encoder=token_encoder)
        print(f"🎉 Initialization complete for '{self.project_name}'.")

    async def search(self, query: str) -> str:
        """Perform a DRIFT search and return a plain-markdown formatted response (readable citations)."""
        if self.drift_searcher is None:
            await self._initialize()

        # ensure a fresh query state
        self.drift_searcher.query_state = QueryState()
        result = await self.drift_searcher.search(query)

        # format to plain markdown with readable citations
        try:
            formatted = format_drift_response_plain(result)
        except Exception:
            formatted = getattr(result, "response", "") or ""
        return formatted


# ---------------- Instance management / factory ----------------
_SEARCHER_CACHE: Dict[str, KnowledgeGraphSearcher] = {}


def get_searcher(project_name: str, **kwargs) -> KnowledgeGraphSearcher:
    """Return a cached KnowledgeGraphSearcher or create+cache it."""
    if project_name not in _SEARCHER_CACHE:
        _SEARCHER_CACHE[project_name] = KnowledgeGraphSearcher(project_name, **kwargs)
    return _SEARCHER_CACHE[project_name]


# ---------------- Optional demo function ----------------
async def async_query_example(project_name: str = "kg", query: str | None = None) -> None:
    """Simple demo showing how to call the searcher. Adjust project_name and query."""
    if query is None:
        query = "Has government prioritised the development of small nuclear-powered reactors?"
    searcher = get_searcher(project_name)
    resp_md = await searcher.search(query)
    print(resp_md)


# ---------------- CLI entrypoint ----------------
if __name__ == "__main__":
    # Simple CLI: python knowledge_graph_searcher.py "project_name" "your query"
    import sys

    proj = sys.argv[1] if len(sys.argv) > 1 else "kg"
    q = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(async_query_example(proj, q))

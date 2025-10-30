import os
import asyncio
from pathlib import Path
from typing import Dict

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

# --- Global Configuration (Load Once) ---
CWD = Path.cwd()
dotenv_path = CWD / ".env"
load_dotenv(dotenv_path)

# Load API keys and model names from environment variables
API_KEY = os.environ.get("GRAPHRAG_API_KEY")
LLM_MODEL = os.environ.get("GRAPHRAG_LLM_MODEL")
EMBED_MODEL = os.environ.get("GRAPHRAG_EMBEDDING_MODEL")


class KnowledgeGraphSearcher:
    """
    An encapsulated class to handle loading and searching a specific GraphRAG project.
    Each instance corresponds to one project directory (e.g., 'kg' or 'kg_news').
    The setup logic is directly mirrored from the original working script.
    """

    def __init__(self, project_name: str, community_level: int = 2):
        """
        Initializes the searcher for a specific project directory.

        Args:
            project_name (str): The name of the project directory (e.g., "kg").
            community_level (int): The community level to use for analysis.
        """
        self.project_name = project_name
        self.community_level = community_level
        self.project_dir = CWD / self.project_name
        self.input_dir = self.project_dir / "output"
        self.lancedb_uri = str(self.input_dir / "lancedb")

        # This will be initialized lazily on the first search call
        self.drift_searcher = None
        print(f"✅ KnowledgeGraphSearcher instance created for '{self.project_name}'. Initialization will occur on first search.")

    async def _initialize(self):
        """
        Loads all necessary data and initializes the DRIFT searcher.
        This is an expensive, one-time operation per instance.
        """
        print(f"🚀 Initializing DRIFT searcher for '{self.project_name}'...")

        # Load static data from the specific project directory
        entity_df = pd.read_parquet(self.input_dir / "entities.parquet")
        community_df = pd.read_parquet(self.input_dir / "communities.parquet")
        relationship_df = pd.read_parquet(self.input_dir / "relationships.parquet")
        text_unit_df = pd.read_parquet(self.input_dir / "text_units.parquet")
        report_df = pd.read_parquet(self.input_dir / "community_reports.parquet")

        # Build indexer adapters using the instance's community level
        entities = read_indexer_entities(entity_df, community_df, self.community_level)
        relationships = read_indexer_relationships(relationship_df)
        text_units = read_indexer_text_units(text_unit_df)
        reports = read_indexer_reports(
            report_df, community_df, self.community_level, content_embedding_col="full_content_embeddings"
        )

        # Setup vector stores for this project
        description_embedding_store = LanceDBVectorStore(collection_name="default-entity-description")
        description_embedding_store.connect(db_uri=self.lancedb_uri)

        full_content_embedding_store = LanceDBVectorStore(collection_name="default-community-full_content")
        full_content_embedding_store.connect(db_uri=self.lancedb_uri)
        read_indexer_report_embeddings(reports, full_content_embedding_store)

        # Manually create model configs, just like the original script
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

        # Manually create DRIFT search config, like the original script
        drift_params = DRIFTSearchConfig(temperature=0, max_tokens=12_000, primer_folds=1, drift_k_followups=3, n_depth=3, n=1)

        # Build the DRIFT context builder and searcher
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

        self.drift_searcher = DRIFTSearch(
            model=chat_model, context_builder=context_builder, token_encoder=token_encoder
        )
        print(f"🎉 Initialization complete for '{self.project_name}'.")

    async def search(self, query: str) -> str:
        """Performs a DRIFT search. Initializes the searcher on the first call."""
        if self.drift_searcher is None:
            await self._initialize()

        self.drift_searcher.query_state = QueryState()
        result = await self.drift_searcher.search(query)
        return result.response


# --- Instance Management (Factory Function) ---

# A cache to store initialized searchers to avoid reloading data.
_SEARCHER_CACHE: Dict[str, KnowledgeGraphSearcher] = {}

def get_searcher(project_name: str, **kwargs) -> KnowledgeGraphSearcher:
    """
    Factory function to get or create a KnowledgeGraphSearcher instance.
    This prevents re-creating instances for the same project directory.
    Passes any extra arguments (like community_level) to the constructor.
    """
    if project_name not in _SEARCHER_CACHE:
        _SEARCHER_CACHE[project_name] = KnowledgeGraphSearcher(project_name, **kwargs)
    return _SEARCHER_CACHE[project_name]
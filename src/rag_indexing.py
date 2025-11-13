"""
Parliamentary Questions RAG System with Qdrant
Hybrid retrieval using dense embeddings + BM25 keyword search
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from openai import OpenAI, AsyncOpenAI
from rank_bm25 import BM25Okapi
import uuid
import asyncio
import tiktoken


@dataclass
class QuestionDocument:
    """Represents a parliamentary question document"""
    q_no: str
    date: str
    subject: str
    ministry: str
    question_text: str
    answer_text: str
    pdf_url: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QuestionDocument':
        """Create QuestionDocument from dictionary"""
        return cls(
            q_no=data.get("Q.No.", ""),
            date=data.get("Date", ""),
            subject=data.get("Subject", ""),
            ministry=data.get("Ministry", ""),
            question_text=data.get("Question Text", ""),
            answer_text=data.get("Answer Text", ""),
            pdf_url=data.get("PDF URL", "")
        )
    
    def get_search_text(self) -> str:
        """Get combined text for indexing (Subject + Question)"""
        return f"{self.subject}\n{self.question_text}"


class ParliamentaryQARetriever:
    """
    RAG system for parliamentary questions with hybrid retrieval
    """
    
    def __init__(
        self,
        collection_name: str = "parliamentary_questions",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize the retriever
        
        Args:
            collection_name: Name of Qdrant collection
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            openai_api_key: OpenAI API key for embeddings
            embedding_model: OpenAI embedding model to use
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize OpenAI clients (both sync and async)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.async_openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Initialize tokenizer for text truncation
        try:
            self.tokenizer = tiktoken.encoding_for_model(embedding_model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Maximum tokens for embedding model (will be set per operation)
        self.max_tokens = None
        
        # BM25 index (loaded when needed)
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        print(f"Warning: Text truncated from {len(tokens)} to {max_tokens} tokens")
        return truncated_text
    
    def _get_embedding(self, text: str, max_tokens: int = 8000) -> List[float]:
        """Get embedding for text using OpenAI"""
        # Truncate text if necessary
        text = self._truncate_text(text, max_tokens)
        
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    async def _get_embedding_async(self, text: str, max_tokens: int = 8000) -> List[float]:
        """Get embedding for text using OpenAI (async)"""
        # Truncate text if necessary
        text = self._truncate_text(text, max_tokens)
        
        response = await self.async_openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def _get_embedding_dimension(self, max_tokens: int = 8000) -> int:
        """Get the dimension of the embedding model"""
        sample_embedding = self._get_embedding("test", max_tokens)
        return len(sample_embedding)
    
    def index_documents(
        self,
        json_file_path: str,
        reset_old: bool = False,
        batch_size: int = 100,
        max_tokens: int = 8000
    ) -> None:
        """
        Index documents from JSON file into Qdrant
        
        Args:
            json_file_path: Path to JSON file with questions
            reset_old: If True, delete existing collection and recreate
            batch_size: Number of documents to process in each batch
            max_tokens: Maximum tokens for embedding text (default: 8000)
        """
        # Load documents
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = [QuestionDocument.from_dict(item) for item in data]
        print(f"Loaded {len(documents)} documents")
        
        # Handle collection creation/deletion
        collection_exists = self.qdrant_client.collection_exists(self.collection_name)
        
        if reset_old and collection_exists:
            print(f"Deleting existing collection: {self.collection_name}")
            self.qdrant_client.delete_collection(self.collection_name)
            collection_exists = False
        
        # Create collection if it doesn't exist
        if not collection_exists:
            embedding_dim = self._get_embedding_dimension(max_tokens)
            print(f"Creating collection with embedding dimension: {embedding_dim}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
        
        # Get existing IDs if appending
        existing_ids = set()
        if not reset_old and collection_exists:
            # Scroll through existing points to get IDs
            offset = None
            while True:
                records, offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                existing_ids.update([rec.payload.get('q_no') for rec in records if rec.payload.get('q_no')])
                if offset is None:
                    break
            print(f"Found {len(existing_ids)} existing documents")
        
        # Index documents in batches
        points = []
        indexed_count = 0
        skipped_count = 0
        
        for i, doc in enumerate(documents):
            # Skip if document already exists (when appending)
            if not reset_old and doc.q_no in existing_ids:
                skipped_count += 1
                continue
            
            # Get search text and embedding
            search_text = doc.get_search_text()
            embedding = self._get_embedding(search_text, max_tokens)
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "q_no": doc.q_no,
                    "date": doc.date,
                    "subject": doc.subject,
                    "ministry": doc.ministry,
                    "question_text": doc.question_text,
                    "answer_text": doc.answer_text,
                    "pdf_url": doc.pdf_url,
                    "search_text": search_text
                }
            )
            points.append(point)
            indexed_count += 1
            
            # Upload batch
            if len(points) >= batch_size:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Indexed {indexed_count} documents...")
                points = []
        
        # Upload remaining points
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        print(f"Indexing complete! Indexed: {indexed_count}, Skipped: {skipped_count}")
        
        # Build BM25 index
        self._build_bm25_index()
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from Qdrant collection"""
        print("Building BM25 index...")
        
        # Retrieve all documents
        all_docs = []
        offset = None
        
        while True:
            records, offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            all_docs.extend(records)
            if offset is None:
                break
        
        # Tokenize documents for BM25
        self.bm25_documents = []
        self.bm25_doc_ids = []
        
        for record in all_docs:
            search_text = record.payload.get('search_text', '')
            tokenized = search_text.lower().split()
            self.bm25_documents.append(tokenized)
            self.bm25_doc_ids.append(record.payload.get('q_no'))
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(self.bm25_documents)
        print(f"BM25 index built with {len(self.bm25_documents)} documents")
    
    def load_index(self) -> None:
        """Load existing index and build BM25"""
        if not self.qdrant_client.collection_exists(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist. Please index documents first.")
        
        print(f"Loading collection: {self.collection_name}")
        self._build_bm25_index()
        print("Index loaded successfully")
    
    def _dense_retrieval(self, query: str, k: int, max_tokens: int = 8000) -> List[Dict]:
        """Perform dense retrieval using embeddings"""
        query_embedding = self._get_embedding(query, max_tokens)
        
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        return [hit.payload for hit in results]
    
    async def _dense_retrieval_async(self, query: str, k: int, max_tokens: int = 8000) -> List[Dict]:
        """Perform dense retrieval using embeddings (async)"""
        query_embedding = await self._get_embedding_async(query, max_tokens)
        
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        return [hit.payload for hit in results]
    
    def _keyword_retrieval(self, query: str, k: int) -> List[Dict]:
        """Perform keyword retrieval using BM25"""
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded. Call load_index() first.")
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Retrieve documents from Qdrant by q_no
        results = []
        for idx in top_k_indices:
            if idx < len(self.bm25_doc_ids):
                q_no = self.bm25_doc_ids[idx]
                
                # Search in Qdrant by q_no
                search_results = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="q_no",
                                match=MatchValue(value=q_no)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )
                
                if search_results[0]:
                    results.append(search_results[0][0].payload)
        
        return results
    
    def retrieve(self, query: str, k: int = 5, max_tokens: int = 8000) -> List[Dict]:
        """
        Retrieve documents using hybrid retrieval
        
        Args:
            query: Search query
            k: Number of documents to retrieve per method
            max_tokens: Maximum tokens for query embedding (default: 8000)
            
        Returns:
            List of retrieved documents (deduplicated)
        """
        # Perform both retrievals
        dense_results = self._dense_retrieval(query, k, max_tokens)
        keyword_results = self._keyword_retrieval(query, k)
        
        # Combine and deduplicate by q_no
        seen_q_nos = set()
        combined_results = []
        
        # Add dense results first
        for doc in dense_results:
            q_no = doc.get('q_no')
            if q_no not in seen_q_nos:
                seen_q_nos.add(q_no)
                combined_results.append(doc)
        
        # Add keyword results
        for doc in keyword_results:
            q_no = doc.get('q_no')
            if q_no not in seen_q_nos:
                seen_q_nos.add(q_no)
                combined_results.append(doc)
        
        return combined_results
    
    async def retrieve_async(self, query: str, k: int = 5, max_tokens: int = 8000) -> List[Dict]:
        """
        Retrieve documents using hybrid retrieval (async)
        
        Args:
            query: Search query
            k: Number of documents to retrieve per method
            max_tokens: Maximum tokens for query embedding (default: 8000)
            
        Returns:
            List of retrieved documents (deduplicated)
        """
        # Perform both retrievals (dense is async, keyword is sync)
        dense_results = await self._dense_retrieval_async(query, k, max_tokens)
        keyword_results = self._keyword_retrieval(query, k)
        
        # Combine and deduplicate by q_no
        seen_q_nos = set()
        combined_results = []
        
        # Add dense results first
        for doc in dense_results:
            q_no = doc.get('q_no')
            if q_no not in seen_q_nos:
                seen_q_nos.add(q_no)
                combined_results.append(doc)
        
        # Add keyword results
        for doc in keyword_results:
            q_no = doc.get('q_no')
            if q_no not in seen_q_nos:
                seen_q_nos.add(q_no)
                combined_results.append(doc)
        
        return combined_results
    
    def get_formatted_context(self, query: str, k: int = 5, max_tokens: int = 8000) -> str:
        """
        Get formatted context string for retrieved documents
        
        Args:
            query: Search query
            k: Number of documents to retrieve per method
            max_tokens: Maximum tokens for query embedding (default: 8000)
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k, max_tokens)
        
        if not results:
            return "No relevant documents found."
        
        formatted_parts = []
        for i, doc in enumerate(results, 1):
            # Build metadata section
            metadata_lines = [
                f"Q.No.: {doc.get('q_no', 'N/A')}",
                f"Date: {doc.get('date', 'N/A')}",
                f"Ministry: {doc.get('ministry', 'N/A')}",
                f"Subject: {doc.get('subject', 'N/A')}"
            ]
            
            # Add PDF URL if available
            if doc.get('pdf_url'):
                metadata_lines.append(f"PDF URL: {doc.get('pdf_url')}")
            
            metadata_str = " | ".join(metadata_lines)
            
            part = f"""{'='*80}
SOURCE {i}
{'='*80}

{metadata_str}

QUESTION:
{doc.get('question_text', 'N/A')}

ANSWER:
{doc.get('answer_text', 'N/A')}
"""
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    async def get_formatted_context_async(self, query: str, k: int = 5, max_tokens: int = 8000) -> str:
        """
        Get formatted context string for retrieved documents (async)
        
        Args:
            query: Search query
            k: Number of documents to retrieve per method
            max_tokens: Maximum tokens for query embedding (default: 8000)
            
        Returns:
            Formatted context string
        """
        results = await self.retrieve_async(query, k, max_tokens)
        
        if not results:
            return "No relevant documents found."
        
        formatted_parts = []
        for i, doc in enumerate(results, 1):
            # Build metadata section
            metadata_lines = [
                f"Q.No.: {doc.get('q_no', 'N/A')}",
                f"Date: {doc.get('date', 'N/A')}",
                f"Ministry: {doc.get('ministry', 'N/A')}",
                f"Subject: {doc.get('subject', 'N/A')}"
            ]
            
            # Add PDF URL if available
            if doc.get('pdf_url'):
                metadata_lines.append(f"PDF URL: {doc.get('pdf_url')}")
            
            metadata_str = " | ".join(metadata_lines)
            
            part = f"""{'='*80}
SOURCE {i}
{'='*80}

{metadata_str}

QUESTION:
{doc.get('question_text', 'N/A')}

ANSWER:
{doc.get('answer_text', 'N/A')}
"""
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize retriever
    retriever = ParliamentaryQARetriever(
        collection_name="parliamentary_questions",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Index documents (first time or reset)
    # retriever.index_documents("questions.json", reset_old=True)
    
    # OR append to existing index
    # retriever.index_documents("new_questions.json", reset_old=False)
    
    # OR load existing index
    retriever.load_index()
    
    # Query and get formatted context
    query = "small modular reactors nuclear energy"
    context = retriever.get_formatted_context(query, k=3)
    print(context)
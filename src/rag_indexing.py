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
from openai import OpenAI
from rank_bm25 import BM25Okapi
import uuid


@dataclass
class QuestionDocument:
    """Represents a parliamentary question document"""
    q_no: str
    date: str
    subject: str
    ministry: str
    question_text: str
    answer_text: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QuestionDocument':
        """Create QuestionDocument from dictionary"""
        return cls(
            q_no=data.get("Q.No.", ""),
            date=data.get("Date", ""),
            subject=data.get("Subject", ""),
            ministry=data.get("Ministry", ""),
            question_text=data.get("Question Text", ""),
            answer_text=data.get("Answer Text", "")
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
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # BM25 index (loaded when needed)
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def _get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        sample_embedding = self._get_embedding("test")
        return len(sample_embedding)
    
    def index_documents(
        self,
        json_file_path: str,
        reset_old: bool = False,
        batch_size: int = 100
    ) -> None:
        """
        Index documents from JSON file into Qdrant
        
        Args:
            json_file_path: Path to JSON file with questions
            reset_old: If True, delete existing collection and recreate
            batch_size: Number of documents to process in each batch
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
            embedding_dim = self._get_embedding_dimension()
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
            embedding = self._get_embedding(search_text)
            
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
    
    def _dense_retrieval(self, query: str, k: int) -> List[Dict]:
        """Perform dense retrieval using embeddings"""
        query_embedding = self._get_embedding(query)
        
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
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve documents using hybrid retrieval
        
        Args:
            query: Search query
            k: Number of documents to retrieve per method
            
        Returns:
            List of retrieved documents (deduplicated)
        """
        # Perform both retrievals
        dense_results = self._dense_retrieval(query, k)
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
    
    def get_formatted_context(self, query: str, k: int = 5) -> str:
        """
        Get formatted context string for retrieved documents
        
        Args:
            query: Search query
            k: Number of documents to retrieve per method
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k)
        
        if not results:
            return "No relevant documents found."
        
        formatted_parts = []
        for i, doc in enumerate(results, 1):
            part = f"""##### Source: {i} #####
Q.No. {doc.get('q_no', 'N/A')}; Date: {doc.get('date', 'N/A')}; Subject: {doc.get('subject', 'N/A')}; Ministry: {doc.get('ministry', 'N/A')}

Question:
{doc.get('question_text', 'N/A')}

Answer:
{doc.get('answer_text', 'N/A')}
"""
            formatted_parts.append(part)
        
        return "\n\n".join(formatted_parts)


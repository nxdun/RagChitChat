"""
ChromaDB Vector Store Implementation
"""
import os
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB vector store implementation for document storage and retrieval"""
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.collection_name = "lecture_notes"
        
        # 💡 Using sentence-transformers for embeddings - fast and accurate for semantic search
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        self._init_client()
        
    def _init_client(self) -> None:
        """Initialize the ChromaDB client"""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "CTSE Lecture Notes"}
            )
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to vector store
        
        Args:
            documents: List of document dicts with 'content' and 'metadata' keys
        """
        try:
            # Extract required fields
            ids = [f"doc_{i}" for i in range(len(documents))]
            texts = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # Add documents to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
    
    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents
        
        Args:
            query_text: The query string
            n_results: Number of results to return
            
        Returns:
            List of document dicts with content and metadata
        """
        try:
            # TODO: Add support for metadata filtering for more specific searches
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Format results
            documents = []
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i],
                    'distance': results.get('distances', [[0] * n_results])[0][i]
                })
            
            return documents
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get stats about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}

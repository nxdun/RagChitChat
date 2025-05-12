"""
Haystack retriever integration for the RAG pipeline
"""
import logging
from typing import List, Dict, Any, Optional
import urllib3

from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HaystackRetriever:
    """Haystack-based document retriever for RAG"""
    
    def __init__(self, vector_store=None):
        """Initialize Haystack retriever
        
        Args:
            vector_store: Optional ChromaVectorStore to use for retrieval
        """
        # Force disable telemetry and warnings before initializing Haystack components
        import os
        os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"
        
        # Silence urllib3 warnings
        urllib3.disable_warnings()
        
        # Disable PostHog warnings by patching
        try:
            import posthog
            # Replace PostHog's capture method with a no-op function
            posthog.capture = lambda *args, **kwargs: None
            # Set the client as disabled
            if hasattr(posthog, 'disabled'):
                posthog.disabled = True
        except ImportError:
            pass
        
        
        self.vector_store = vector_store

        # ! This is an in-memory approach - for larger datasets consider a persistent store
        self.document_store = InMemoryDocumentStore()
        self.bm25_retriever = InMemoryBM25Retriever(document_store=self.document_store)
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the retriever
        
        Args:
            documents: List of document dicts with 'content' and 'metadata'
        """
        haystack_docs = []
        
        for doc in documents:
            haystack_docs.append(
                Document(content=doc['content'], meta=doc['metadata'])
            )
        
        try:
            # Add to document store
            self.document_store.write_documents(haystack_docs)
            logger.info(f"Added {len(haystack_docs)} documents to Haystack document store")
        except Exception as e:
            logger.error(f"Failed to add documents to Haystack: {str(e)}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve relevant documents using Haystack
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of Haystack Document objects
        """
        try:
            # Use BM25 retriever
            results = self.bm25_retriever.run(query=query, top_k=top_k)
            documents = results.get("documents", [])  # Ensure documents is a list
            if not documents:
                logger.info("BM25 retriever found no documents.")
                return []
            logger.info(f"Retrieved {len(documents)} documents using BM25 retriever")
            return documents
        except Exception as e:
            logger.error(f"Haystack retrieval error: {str(e)}")
            return []
    
    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid retrieval using both BM25 and vector search
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of document dicts
        """
        if self.vector_store is None:
            logger.warning("Vector store not available for hybrid search. Falling back to BM25.")
            bm25_results = self.retrieve(query, top_k)
            if not bm25_results:
                return []
            return self._convert_to_dicts(bm25_results)
        
        try:
            # Get sparse results (BM25)
            sparse_results = self.retrieve(query, top_k)
            sparse_docs = self._convert_to_dicts(sparse_results) if sparse_results else []
            
            # Get dense results (Vector)
            dense_docs = self.vector_store.query(query, top_k)  # Assuming vector_store.query returns an empty list if nothing is found or below threshold
            
            if not sparse_docs and not dense_docs:
                logger.info("Hybrid search (BM25 and Vector) found no documents.")
                return []

            # Combine results (simple approach - could be improved)
            seen_contents = set()
            hybrid_results = []
            
            # 💡 Hybrid search combines semantic similarity (vectors) with keyword matching (BM25)
            # This gives better results than either method alone
            
            # Add dense results first (prioritize semantic search)
            for doc in dense_docs:
                content_hash = hash(doc['content'])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc['retrieval_method'] = 'vector'
                    hybrid_results.append(doc)
            
            # Add sparse results that aren't duplicates
            for doc in sparse_docs:
                content_hash = hash(doc['content'])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc['retrieval_method'] = 'bm25'
                    hybrid_results.append(doc)
            
            if not hybrid_results:
                logger.info("Hybrid search yielded no combined results after filtering.")
                return []
            
            # Trim to top_k
            hybrid_results = hybrid_results[:top_k]
            
            logger.info(f"Retrieved {len(hybrid_results)} documents using hybrid search")
            return hybrid_results
        except Exception as e:
            logger.error(f"Hybrid retrieval error: {str(e)}")
            # Fallback to BM25 if hybrid fails, and ensure it handles empty results
            bm25_fallback_results = self.retrieve(query, top_k)
            if not bm25_fallback_results:
                return []
            return self._convert_to_dicts(bm25_fallback_results)
    
    @staticmethod
    def _convert_to_dicts(documents: List[Document]) -> List[Dict[str, Any]]:
        """Convert Haystack Document objects to dicts"""
        return [
            {
                'content': doc.content,
                'metadata': doc.meta,
                'id': doc.id,
                'score': doc.score if hasattr(doc, 'score') else None
            }
            for doc in documents
        ]

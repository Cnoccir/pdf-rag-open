"""
Vector store implementation for technical documents with LangGraph integration.
Provides efficient storage and retrieval of document vectors using Pinecone.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Pydantic
from pydantic import BaseModel, Field

# Updated Langchain imports for v0.2
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Pinecone imports
from langchain_openai import OpenAIEmbeddings
import pinecone
from langchain_pinecone import PineconeVectorStore

# App imports
from app.chat.types import (
    SearchQuery,
    SearchResult,
    ProcessingResult,
    ContentElement
)

logger = logging.getLogger(__name__)


class VectorStoreMetrics(BaseModel):
    """Metrics for vector store operations."""
    total_embeddings: int = 0
    total_queries: int = 0
    total_retrievals: int = 0
    avg_query_time: float = 0.0
    query_times: List[float] = Field(default_factory=list)
    batch_sizes: List[int] = Field(default_factory=list)
    error_count: int = 0
    last_error: Optional[str] = None
    total_filter_ops: int = 0

    def record_query_time(self, time_seconds: float) -> None:
        """Record time taken for a query."""
        self.query_times.append(time_seconds)
        self.total_queries += 1
        self.avg_query_time = sum(self.query_times) / len(self.query_times)

    def record_batch(self, batch_size: int) -> None:
        """Record batch size for embedding operations."""
        self.batch_sizes.append(batch_size)
        self.total_embeddings += batch_size

    def record_error(self, error_message: str) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error_message

    def record_retrieval(self) -> None:
        """Record a retrieval operation."""
        self.total_retrievals += 1

    def record_filter_op(self) -> None:
        """Record a filter operation."""
        self.total_filter_ops += 1


class CachedEmbeddings:
    """
    Embedding wrapper with caching to avoid redundant embedding calculations.
    """

    def __init__(self, embedding, pdf_id: str, cache_size: int = 1000):
        """
        Initialize with embedding model and cache.

        Args:
            embedding: Base embedding model
            pdf_id: Document ID for cache partitioning
            cache_size: LRU cache size
        """
        self.embedding = embedding
        self.pdf_id = pdf_id
        self.cache = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Check which texts are cached
        results = []
        texts_to_embed = []
        indices = []

        for i, text in enumerate(texts):
            # Create cache key (must be hashable)
            key = self._create_cache_key(text)

            if key in self.cache:
                # Cache hit
                results.append(self.cache[key])
                self.hits += 1
            else:
                # Cache miss - need to embed
                texts_to_embed.append(text)
                indices.append(i)
                self.misses += 1

        # If we have texts to embed, do it
        if texts_to_embed:
            # Embed new texts
            new_embeddings = self.embedding.embed_documents(texts_to_embed)

            # Store in cache
            for j, embedding in enumerate(new_embeddings):
                text = texts_to_embed[j]
                key = self._create_cache_key(text)
                self.cache[key] = embedding

            # Insert new embeddings into correct positions
            for j, idx in enumerate(indices):
                results.insert(idx, new_embeddings[j])

        # Manage cache size
        self._manage_cache()

        return results

    def embed_query(self, text: str) -> List[float]:
        """
        Embed query with caching.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        # Create cache key
        key = self._create_cache_key(text, is_query=True)

        # Check cache
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        # Cache miss
        self.misses += 1
        embedding = self.embedding.embed_query(text)

        # Store in cache
        self.cache[key] = embedding

        # Manage cache size
        self._manage_cache()

        return embedding

    def _create_cache_key(self, text: str, is_query: bool = False) -> str:
        """
        Create stable cache key from text.

        Args:
            text: Text to create key for
            is_query: Whether this is a query embedding

        Returns:
            Cache key string
        """
        # Use a stable hash that's specific to this document and text
        # Include a prefix to differentiate between query and document embeddings
        prefix = "q_" if is_query else "d_"
        return f"{prefix}{self.pdf_id}_{hash(text)}"

    def _manage_cache(self) -> None:
        """
        Manage cache size to prevent memory issues.
        Removes oldest entries when cache exceeds maximum size.
        """
        if len(self.cache) > self.cache_size:
            # Simple LRU implementation - remove oldest 25% of entries
            remove_count = int(0.25 * self.cache_size)
            keys_to_remove = list(self.cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.cache[key]


class TechDocVectorStore:
    """Vector store implementation for technical documents with LangGraph integration."""
    
    def __init__(self):
        """Initialize the vector store with Pinecone."""
        self.metrics = VectorStoreMetrics()
        self.initialized = False
        self.pc = None
        self.vectorstore = None
        self.embedding = None
        self.index_name = None
        
        try:
            # Initialize the embedding model
            self.embedding = OpenAIEmbeddings()
            
            # Get Pinecone configuration
            pc_api_key = os.getenv("PINECONE_API_KEY")
            pc_environment = os.getenv("PINECONE_ENVIRONMENT")
            self.index_name = os.getenv("PINECONE_INDEX", "tech-rag-v1")
            
            if not pc_api_key or not pc_environment:
                logger.error("Pinecone API key or environment not configured")
                return
                
            # Initialize Pinecone with new SDK approach
            self.pc = pinecone.Pinecone(api_key=pc_api_key)
            logger.info("Pinecone configuration loaded with new SDK approach")
            
            # Check if index exists
            available_indexes = self.pc.list_indexes().names()
            if not available_indexes:
                logger.error("No indexes found in Pinecone account")
                return
                
            if self.index_name not in available_indexes:
                logger.warning(f"Index {self.index_name} not found in Pinecone. Available indexes: {available_indexes}")
                # If we have at least one index available, use the first one
                if len(available_indexes) > 0:
                    self.index_name = available_indexes[0]
                    logger.info(f"Using alternative index: {self.index_name}")
                else:
                    return
                
            logger.info(f"Using Pinecone index: {self.index_name}")
            
            # Connect to the vector store using the index
            index = self.pc.Index(self.index_name)
            
            # Initialize PineconeVectorStore with the index
            self.vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embedding,
                text_key="text",
                namespace="default"
            )
            
            self.initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
            self.initialized = False
    
    def similarity_search(self, query: str, k: int = 5, pdf_id: str = None):
        """
        Perform a similarity search using the vector store.
        
        Args:
            query: The search query
            k: Number of results to return
            pdf_id: Optional PDF ID to filter results
            
        Returns:
            List of Documents
        """
        if not self.initialized or not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        # Track timing
        start_time = time.time()
        
        try:
            # Create filter dict if pdf_id is specified
            filter_dict = {"pdf_id": pdf_id} if pdf_id else None
            
            # Perform similarity search
            docs = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # Record metrics
            query_time = time.time() - start_time
            self.metrics.record_query_time(query_time)
            self.metrics.total_queries += 1
            
            logger.info(f"Similarity search completed in {query_time:.2f}s, found {len(docs)} documents")
            
            return docs
            
        except Exception as e:
            error_msg = f"Error in similarity search: {str(e)}"
            logger.error(error_msg)
            self.metrics.record_error(error_msg)
            return []
    
    def add_documents(self, documents: List[Document], namespace: str = "default"):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            namespace: Namespace to add documents to
        """
        if not self.initialized or not self.vectorstore:
            logger.error("Vector store not initialized")
            return
        
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            # Track batch size for metrics
            self.metrics.record_batch(len(documents))
            
            # Add documents to vector store
            self.vectorstore.add_documents(
                documents=documents,
                namespace=namespace
            )
            
            # Update metrics
            self.metrics.total_embeddings += len(documents)
            
            logger.info(f"Added {len(documents)} documents to vector store in namespace '{namespace}'")
            
        except Exception as e:
            error_msg = f"Error adding documents: {str(e)}"
            logger.error(error_msg)
            self.metrics.record_error(error_msg)
    
    def process_document_elements(self, elements: List[ContentElement], pdf_id: str):
        """
        Process document elements and add them to the vector store.
        
        Args:
            elements: List of content elements from document processing
            pdf_id: ID of the PDF document
        """
        if not elements:
            logger.warning(f"No elements to process for PDF {pdf_id}")
            return
        
        # Convert elements to documents
        documents = []
        
        for element in elements:
            # Create document with metadata
            doc = Document(
                page_content=element.content,
                metadata={
                    "id": element.id,
                    "pdf_id": pdf_id,
                    "content_type": element.content_type,
                    "page_number": element.page_number,
                    "section": element.section,
                    "section_title": element.metadata.get("section_title", ""),
                    "chunk_id": element.metadata.get("chunk_id", "")
                }
            )
            documents.append(doc)
        
        # Add documents to vector store
        namespace = f"pdf_{pdf_id}" if pdf_id else "default"
        self.add_documents(documents, namespace=namespace)
        
        logger.info(f"Processed {len(documents)} elements for PDF {pdf_id}")

    def process_processing_result(self, result: ProcessingResult):
        """
        Process a processing result and add its elements to the vector store.
        
        Args:
            result: Processing result with document content
        """
        if not result or not result.elements:
            logger.warning("No elements in processing result")
            return
        
        # Process elements
        self.process_document_elements(result.elements, result.pdf_id)
        
        logger.info(f"Processed result for PDF {result.pdf_id} with {len(result.elements)} elements")
    
    def retrieve(self, query: str, k: int = 5, pdf_id: str = None, filter_content_types: List[str] = None):
        """
        Retrieve documents matching the query with enhanced search result formatting.
        
        Args:
            query: The search query
            k: Number of results to return
            pdf_id: Optional PDF ID to filter results
            filter_content_types: Optional list of content types to filter results
            
        Returns:
            SearchResult object with documents and metadata
        """
        # Create filter dict
        filter_dict = {}
        
        if pdf_id:
            filter_dict["pdf_id"] = pdf_id
            
        if filter_content_types:
            filter_dict["content_type"] = {"$in": filter_content_types}
        
        # Perform search
        try:
            # Track timing
            start_time = time.time()
            
            # Use the vectorstore directly or with filter
            if filter_dict:
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
                self.metrics.record_filter_op()
            else:
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )
            
            # Record metrics
            query_time = time.time() - start_time
            self.metrics.record_query_time(query_time)
            self.metrics.total_queries += 1
            self.metrics.record_retrieval()
            
            # Create SearchResult
            result = SearchResult(
                query=query,
                documents=docs,
                total_results=len(docs),
                search_time=query_time
            )
            
            logger.info(f"Retrieved {len(docs)} documents in {query_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in retrieval: {str(e)}"
            logger.error(error_msg)
            self.metrics.record_error(error_msg)
            
            # Return empty result
            return SearchResult(
                query=query,
                documents=[],
                total_results=0,
                search_time=0,
                error=error_msg
            )
    
    def get_cached_embedding(self, pdf_id: str = None):
        """
        Get a cached embedding instance for the given PDF ID.
        
        Args:
            pdf_id: PDF ID to use for cache partitioning
            
        Returns:
            CachedEmbeddings instance
        """
        return CachedEmbeddings(self.embedding, pdf_id or "general")


# Singleton instance
_vector_store = None

def get_vector_store() -> TechDocVectorStore:
    """
    Get the singleton vector store instance.
    
    Returns:
        Vector store instance
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = TechDocVectorStore()
        
    return _vector_store

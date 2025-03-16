from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import Dict, List, Optional, Set, Any, Tuple, Iterator, Union
from datetime import datetime
from collections import defaultdict
import logging
import json
import re
from pathlib import Path
import os
import time
import uuid
from functools import lru_cache

# Updated Langchain imports for v0.2
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Pinecone imports
from langchain_openai import OpenAIEmbeddings
import pinecone
from langchain_pinecone import PineconeVectorStore

from pydantic import BaseModel, PrivateAttr, Field, root_validator

from app.chat.types import (
    ContentType,
    ProcessingResult,
    ContentElement,
    ContentMetadata,
    ProcessingConfig,
    SearchQuery,
    SearchResult,
    ResearchMode,
    ResearchManager,
    ConceptNetwork,
    ConceptRelationship,
    Concept,
    ResearchResult,
    ResearchContext,
    RelationType
)
from app.chat.errors import VectorStoreError
from app.chat.utils import (
    extract_technical_terms,
    detect_content_types,
    extract_concept_relationships,
    extract_hierarchy_relationships,
    normalize_metadata_for_vectorstore,
    setup_logging
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
            Cache key
        """
        # Create a stable hash
        prefix = "q_" if is_query else "d_"
        return f"{prefix}_{self.pdf_id}_{hash(text)}"

    def _manage_cache(self) -> None:
        """Manage cache size by removing oldest entries."""
        if len(self.cache) > self.cache_size:
            # Remove oldest entries (25% of cache)
            remove_count = self.cache_size // 4
            keys_to_remove = list(self.cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.cache[key]

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        total = self.hits + self.misses
        hit_rate = self.hits / max(1, total)

        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "pdf_id": self.pdf_id
        }


class TechnicalDocumentStore(BaseRetriever):
    """
    Enhanced vector store for technical documents with hierarchical understanding.
    Optimized for document structure preservation, cross-document insights,
    and efficient multi-modal retrieval.
    """

    _store = PrivateAttr(default=None)
    _research_manager = PrivateAttr()
    _config = PrivateAttr()
    _openai_client = PrivateAttr()
    _index_name = PrivateAttr(default=None)
    _embeddings = PrivateAttr(default=None)
    _registries = PrivateAttr(default=None)
    _metrics = PrivateAttr(default=None)

    def __init__(
        self,
        research_manager: ResearchManager,
        processing_config: ProcessingConfig,
        openai_client: Optional[Any] = None
    ):
        """Initialize the document store with enhanced configuration."""
        from app.chat.utils import normalize_pdf_id

        super().__init__()
        self._research_manager = research_manager
        self._config = processing_config
        self._openai_client = openai_client
        self._metrics = VectorStoreMetrics()

        # Ensure PDF ID is a string
        if hasattr(self._config, 'pdf_id'):
            self._config.pdf_id = normalize_pdf_id(self._config.pdf_id)

        # 1) Initialize vector store
        self._initialize_store()

        # 2) Setup internal registries
        self._setup_registries()

        # 3) Initialize embeddings
        self._init_embeddings()

        logger.info(f"TechnicalDocumentStore initialized with config: {processing_config.dict(exclude={'embedding_dimensions'})}")

    def _initialize_store(self):
        """
        Initialize Pinecone vector store with a shared namespace strategy for better multi-document search.
        """
        try:
            # Initialize the embeddings first
            embeddings = self._get_embeddings()

            # Setup Pinecone client
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("Missing PINECONE_API_KEY in environment variables")

            # Initialize Pinecone client
            pc = pinecone.Pinecone(api_key=api_key)

            # Use a common index for all documents
            self._index_name = os.getenv("PINECONE_INDEX_NAME", "tech-rag-v1")

            # Set index dimensions based on embedding model
            dimensions = self._config.embedding_dimensions

            # Check if index exists
            existing_indexes = [idx["name"] for idx in pc.list_indexes()]

            # Create index if needed with error handling and retries
            if self._index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self._index_name}")

                # Retry logic for index creation
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        pc.create_index(
                            name=self._index_name,
                            dimension=dimensions,
                            metric="cosine",
                            spec=pinecone.ServerlessSpec(
                                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                region=os.getenv("PINECONE_REGION", "us-west-2")
                            )
                        )
                        break
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Retrying index creation ({retry_count}/{max_retries}): {e}")
                        if retry_count >= max_retries:
                            # If all retries fail, try to use existing index as fallback
                            if existing_indexes:
                                self._index_name = existing_indexes[0]
                                logger.warning(f"Using existing index as fallback: {self._index_name}")
                            else:
                                raise
                        time.sleep(2)

                # Wait for index to be ready
                self._wait_for_index_readiness(pc)

            # Connect to the index
            index = pc.Index(self._index_name)

            # Create the vector store with metadata
            self._store = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text",
                # Use a common namespace for all documents instead of PDF-specific namespaces
                namespace="all_documents"
            )

            logger.info(f"Vector store initialized with index: {self._index_name}, namespace: all_documents")

        except (ValueError, pinecone.core.client.exceptions.ApiException) as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            self._metrics.record_error(f"Pinecone initialization: {str(e)}")
            raise VectorStoreError(self._config.pdf_id, f"Failed to initialize Pinecone: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error initializing vector store: {str(e)}")
            self._metrics.record_error(f"Unexpected initialization: {str(e)}")
            raise VectorStoreError(self._config.pdf_id, f"Vector store initialization failed: {str(e)}")

    def _wait_for_index_readiness(self, pinecone_client, max_retries=10, wait_time=2):
        """
        Wait for Pinecone index to be ready with retry logic.

        Args:
            pinecone_client: Pinecone client
            max_retries: Maximum number of retries
            wait_time: Time to wait between retries (seconds)
        """
        is_ready = False
        retries = 0

        while not is_ready and retries < max_retries:
            try:
                # Use the new Pinecone SDK method to check index status
                index_description = pinecone_client.describe_index(name=self._index_name)
                # The ready status is now directly accessible in the response
                is_ready = index_description.status.ready

                if not is_ready:
                    logger.info(f"Waiting for Pinecone index to be ready (attempt {retries+1}/{max_retries})...")
                    retries += 1
                    time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"Error checking index readiness: {e}")
                retries += 1
                time.sleep(wait_time)

        if not is_ready:
            logger.warning("Index may not be fully ready, proceeding anyway...")

    def _setup_registries(self):
        """
        Initialize internal registries for content tracking and metadata.
        """
        self._registries = {
            # Core content registries
            "concepts": defaultdict(dict),            # Concept ID -> concept info
            "content": defaultdict(dict),             # Element ID -> content metadata
            "relationships": defaultdict(list),       # Concept ID -> list of relationships
            "primary_concepts": [],                   # List of primary concepts

            # Cross-reference registries
            "technical_terms": defaultdict(set),      # Term -> set of element IDs
            "section_elements": defaultdict(list),    # Section path -> list of element IDs
            "page_elements": defaultdict(list),       # Page number -> list of element IDs
            "section_concepts": defaultdict(list),    # Section path -> list of concept IDs

            # Visual content registries
            "visual_context": defaultdict(dict),      # Image ID -> visual context
            "image_references": {},                   # Image ID -> image metadata
            "table_references": {},                   # Table ID -> table metadata

            # Enhanced contextual registries
            "embedding_cache": {},                    # Element ID -> embedding vector
            "document_metadata": {},                  # Document-level metadata
            "hierarchical_index": defaultdict(list),  # Section -> list of child sections

            # Enhanced relationship registries
            "relationship_types": defaultdict(list),  # Relationship type -> list of relationships
            "relationship_contexts": defaultdict(list),  # Context term -> list of relationships
            "section_relationship_map": defaultdict(list),  # Section -> list of relationships
        }

    def _init_embeddings(self):
        """Initialize embedding model with caching and error handling."""
        self._embeddings = self._get_embeddings()

    def _get_embeddings(self):
        """
        Create and return advanced embeddings model with caching.

        Returns:
            Embeddings model
        """
        try:
            # Create embeddings with caching for efficiency
            return CachedEmbeddings(
                embedding=OpenAIEmbeddings(
                    model=self._config.embedding_model,
                    dimensions=self._config.embedding_dimensions,
                    timeout=60.0  # Increased timeout for batch operations
                ),
                pdf_id=self._config.pdf_id
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            self._metrics.record_error(f"Embeddings initialization: {str(e)}")
            raise VectorStoreError(self._config.pdf_id, f"Embeddings initialization failed: {str(e)}")

    async def ingest_processed_content(self, result: ProcessingResult) -> None:
        """
        Ingest processed content with optimized batching and hierarchical tracking.

        Args:
            result: Processing result with document content
        """
        start_time = time.time()
        try:
            logger.info(f"Starting enhanced content ingestion for document {result.pdf_id}")

            # 1. Process content elements for vectorization
            element_docs = []

            for element in result.elements:
                # Create document from element
                doc = self._element_to_document(element)
                if doc:
                    element_docs.append(doc)

                    # Track element by section for hierarchical retrieval
                    if element.metadata.section_headers:
                        section_path = tuple(element.metadata.section_headers)
                        self._registries["section_elements"][section_path].append(element.element_id)

                    # Track element by page
                    if element.metadata.page_number:
                        self._registries["page_elements"][element.metadata.page_number].append(element.element_id)

                    # Track technical terms
                    for term in element.metadata.technical_terms:
                        self._registries["technical_terms"][term].add(element.element_id)

                    # Index element in content registry
                    self._registries["content"][element.element_id] = {
                        "content_type": element.content_type,
                        "page_number": element.metadata.page_number or 0,
                        "hierarchy_level": element.metadata.hierarchy_level or 0,
                        "technical_terms": element.metadata.technical_terms[:15] if element.metadata.technical_terms else [],
                        "source_type": "element"
                    }

            logger.info(f"Processing {len(element_docs)} content elements for ingestion")

            # 2. Process chunks with enhanced metadata
            chunk_docs = []
            for chunk in result.chunks:
                doc = self._chunk_to_document(chunk)
                if doc:
                    chunk_docs.append(doc)
            logger.info(f"Processing {len(chunk_docs)} chunks for ingestion")

            # 3. Add documents to vector store in optimized batches
            all_docs = element_docs + chunk_docs

            # Determine optimal batch size based on document size
            batch_size = self._get_optimal_batch_size(all_docs)
            logger.info(f"Using batch size {batch_size} for {len(all_docs)} documents")

            successful_batches = 0
            failed_batches = 0
            total_batches = (len(all_docs) + batch_size - 1) // batch_size

            # Process in batches with improved error handling
            for i in range(0, len(all_docs), batch_size):
                batch = all_docs[i:i+batch_size]
                batch_index = i // batch_size + 1
                logger.info(f"Adding batch {batch_index}/{total_batches}: {len(batch)} documents")

                # Normalize metadata for each document in the batch
                for doc in batch:
                    if hasattr(doc, 'metadata') and doc.metadata:
                        doc.metadata = normalize_metadata_for_vectorstore(doc.metadata)

                success = False
                for retry in range(3):  # Try up to 3 times
                    try:
                        batch_start = time.time()
                        await self._store.aadd_documents(batch)
                        batch_time = time.time() - batch_start
                        logger.info(f"Batch {batch_index} completed in {batch_time:.2f}s")
                        self._metrics.record_batch(len(batch))
                        successful_batches += 1
                        success = True
                        break
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Error adding batch {batch_index} (attempt {retry+1}/3): {error_msg}")
                        # If metadata error, try to remove the problematic field
                        if "Metadata value must be" in error_msg:
                            field_match = re.search(r"for field '([^']+)'", error_msg)
                            if field_match:
                                problem_field = field_match.group(1)
                                logger.warning(f"Removing problematic field '{problem_field}' from all documents in batch {batch_index}")
                                for doc in batch:
                                    if problem_field in doc.metadata:
                                        doc.metadata.pop(problem_field, None)
                        # Wait before retry with exponential backoff (1s, 2s)
                        if retry < 2:
                            await asyncio.sleep(2 ** retry)
                if not success:
                    failed_batches += 1
                    self._metrics.record_error(f"Failed to add batch {batch_index} after 3 attempts")

            logger.info(f"Batch processing complete: {successful_batches}/{total_batches} successful ({failed_batches} failed)")

            # 4. Process concept network if available
            if result.concept_network:
                await self._process_concept_network(result.concept_network)

            # 5. Update the research manager
            self._update_research_data(result)

            ingestion_time = time.time() - start_time
            logger.info(f"Successfully ingested content for {result.pdf_id} in {ingestion_time:.2f}s")

            # Store document metadata
            self._registries["document_metadata"][result.pdf_id] = {
                "ingested_at": datetime.utcnow().isoformat(),
                "ingestion_time": ingestion_time,
                "total_elements": len(result.elements),
                "total_chunks": len(result.chunks),
                "metrics": result.processing_metrics,
                "concept_count": len(result.concept_network.concepts) if result.concept_network else 0,
                "relationship_count": len(result.concept_network.relationships) if result.concept_network else 0,
                "primary_concepts": result.concept_network.primary_concepts if result.concept_network else []
            }

        except Exception as e:
            logger.error(f"Content ingestion failed: {str(e)}", exc_info=True)
            self._metrics.record_error(f"Content ingestion: {str(e)}")
            raise VectorStoreError(self._config.pdf_id, f"Failed to ingest content: {str(e)}")

    def _get_optimal_batch_size(self, documents: List[Document]) -> int:
        """
        Calculate optimal batch size based on document characteristics.

        Args:
            documents: List of documents

        Returns:
            Optimal batch size
        """
        # Default sizes
        DEFAULT_BATCH_SIZE = 100
        MIN_BATCH_SIZE = 10
        MAX_BATCH_SIZE = 200

        # If no documents, return default
        if not documents:
            return DEFAULT_BATCH_SIZE

        # Calculate average document size
        avg_len = sum(len(doc.page_content) for doc in documents) / len(documents)

        # Scale batch size inversely with document size
        if avg_len > 5000:
            batch_size = 50  # Smaller batch for large documents
        elif avg_len > 2000:
            batch_size = 100  # Medium batch for medium documents
        else:
            batch_size = 200  # Larger batch for small documents

        # Ensure within bounds
        return max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, batch_size))

    async def _process_concept_network(self, network: ConceptNetwork) -> None:
        """
        Process and store concept network for enhanced retrieval.

        This improved implementation handles richer network data including section
        context, primary concepts, and comprehensive relationship metadata.

        Args:
            network: Concept network from document processing
        """
        try:
            # 1. Register concepts with enhanced metadata
            for concept in network.concepts:
                concept_id = concept.name
                self._registries["concepts"][concept_id] = {
                    "name": concept.name,
                    "description": concept.description,
                    "confidence": concept.confidence,
                    "category": concept.category,
                    "occurrences": concept.occurrences,
                    "sections": concept.sections,
                    "is_primary": concept.is_primary,
                    "importance_score": concept.importance_score,
                    "in_headers": concept.in_headers,
                    "first_occurrence_page": concept.first_occurrence_page
                }

            # 2. Register primary concepts for quick access
            self._registries["primary_concepts"] = network.primary_concepts.copy()

            # 3. Register section-to-concept mappings
            for section, concepts in network.section_concepts.items():
                self._registries["section_concepts"][section] = concepts.copy()

            # 4. Register relationships with detailed metadata
            for relationship in network.relationships:
                source_id = relationship.source
                target_id = relationship.target

                # Standardize relationship type
                rel_type = relationship.type
                if isinstance(rel_type, str):
                    rel_type_value = rel_type
                else:  # RelationType enum
                    rel_type_value = rel_type.value

                # Create comprehensive relationship data
                rel_data = {
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type_value,
                    "weight": relationship.weight,
                    "context": relationship.context,
                    "doc_id": self._config.pdf_id,
                    "page_number": relationship.page_number,
                    "section_path": relationship.section_path,
                    "extraction_method": relationship.extraction_method,
                    "evidence": relationship.evidence,
                    "confidence_factors": relationship.confidence_factors
                }

                # Store in main relationship registry
                self._registries["relationships"][source_id].append(rel_data)

                # Index by relationship type
                self._registries["relationship_types"][rel_type_value].append(rel_data)

                # Index by section if available
                if relationship.section_path:
                    section_key = " > ".join(relationship.section_path) if isinstance(relationship.section_path, list) else relationship.section_path
                    self._registries["section_relationship_map"][section_key].append(rel_data)

                # Store bidirectionally for efficient querying
                reverse_rel = rel_data.copy()
                reverse_rel["source"] = target_id
                reverse_rel["target"] = source_id

                # For directional relationships, adjust the reverse type
                if rel_type_value in ["contains", "defines", "has"]:
                    if rel_type_value == "contains":
                        reverse_rel["type"] = "part_of"
                    elif rel_type_value == "defines":
                        reverse_rel["type"] = "defined_by"
                    elif rel_type_value == "has":
                        reverse_rel["type"] = "belongs_to"

                self._registries["relationships"][target_id].append(reverse_rel)

            # 5. Register with research manager for cross-document analysis
            if self._research_manager:
                self._research_manager.register_concept_network(self._config.pdf_id, network)

            logger.info(f"Processed concept network with {len(network.concepts)} concepts and {len(network.relationships)} relationships")

        except Exception as e:
            logger.error(f"Failed to process concept network: {e}")
            self._metrics.record_error(f"Concept network processing: {str(e)}")

    def _element_to_document(self, element: ContentElement) -> Optional[Document]:
        """
        Convert ContentElement to Document with improved metadata handling.
        Updated to use the extract_hierarchy_relationships function for relationship extraction.
        """
        try:
            # Get searchable content with context
            content = element.get_searchable_content()

            # Create metadata optimized for retrieval
            metadata = {
                "pdf_id": element.pdf_id,
                "element_id": element.element_id,
                "content_type": element.content_type.value,
                "page_number": element.metadata.page_number or 0,
                "hierarchy_level": element.metadata.hierarchy_level or 0,
                "technical_terms": element.metadata.technical_terms[:15] if element.metadata.technical_terms else [],
                "source_type": "element"
            }

            # Add hierarchy from section headers
            if element.metadata.section_headers:
                # Convert section headers to hierarchy string
                hierarchy = " > ".join(element.metadata.section_headers)
                metadata["hierarchy"] = hierarchy
                metadata["section_path"] = element.metadata.section_headers
                metadata["section_str"] = hierarchy
            else:
                # Use empty string instead of null for hierarchy
                metadata["hierarchy"] = ""
                metadata["section_path"] = []
                metadata["section_str"] = ""

            # Add hierarchical relationships
            if element.metadata.parent_element:
                metadata["parent_element"] = element.metadata.parent_element
            else:
                metadata["parent_element"] = ""

            if element.metadata.child_elements:
                metadata["child_count"] = len(element.metadata.child_elements)
                metadata["child_elements"] = element.metadata.child_elements
            else:
                metadata["child_count"] = 0
                metadata["child_elements"] = []

            # Add type-specific metadata
            if element.content_type == ContentType.IMAGE and element.metadata.image_metadata:
                metadata["image_path"] = element.metadata.image_path or ""
                metadata["image_description"] = element.metadata.image_metadata.analysis.description or ""
                metadata["is_image"] = True

                # Add extracted technical terms from image
                if element.metadata.image_metadata.analysis.technical_concepts:
                    metadata["image_technical_terms"] = element.metadata.image_metadata.analysis.technical_concepts
                else:
                    metadata["image_technical_terms"] = []

            elif element.content_type == ContentType.TABLE and element.metadata.table_data:
                table_data = element.metadata.table_data
                metadata["table_caption"] = table_data.caption or ""
                metadata["table_headers"] = table_data.headers[:5] if table_data.headers else []
                metadata["is_table"] = True

                # Add table-specific technical terms
                if table_data.technical_concepts:
                    metadata["table_technical_terms"] = table_data.technical_concepts
                else:
                    metadata["table_technical_terms"] = []

            elif element.content_type == ContentType.CODE and element.metadata.code_block_data:
                code_data = element.metadata.code_block_data
                metadata["code_language"] = code_data.language or ""
                metadata["is_code"] = True

                # Add code-specific technical terms
                if code_data.technical_concepts:
                    metadata["code_technical_terms"] = code_data.technical_concepts
                else:
                    metadata["code_technical_terms"] = []

            # Add relationship data if available
            if hasattr(element.metadata, 'relationships') and element.metadata.relationships:
                metadata["relationships"] = element.metadata.relationships
            else:
                # Extract relationships on the fly using our new helper function
                if element.metadata.technical_terms:
                    known_concepts = set(element.metadata.technical_terms)
                    extracted_relationships = extract_hierarchy_relationships(
                        element.content,
                        known_concepts=known_concepts,
                        min_confidence=0.7
                    )
                    metadata["relationships"] = extracted_relationships
                else:
                    metadata["relationships"] = []

            # Create the document with normalized metadata
            return Document(
                page_content=content,
                metadata=normalize_metadata_for_vectorstore(metadata)
            )

        except Exception as e:
            logger.warning(f"Failed to convert element to document: {e}")
            return None

    def _chunk_to_document(self, chunk: Dict[str, Any]) -> Optional[Document]:
        """
        Convert chunk data to Document with optimized metadata.
        Ensures no null values for important fields.
        """
        try:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {}).copy()

            # Ensure required fields
            metadata["pdf_id"] = self._config.pdf_id
            metadata["source_type"] = "chunk"
            metadata["chunk_id"] = chunk.get("chunk_id", str(uuid.uuid4()))

            # Extract technical terms if not present
            if "technical_terms" not in metadata or not metadata["technical_terms"]:
                metadata["technical_terms"] = extract_technical_terms(content)

            # Extract content types if not present
            if "content_types" not in metadata or not metadata["content_types"]:
                metadata["content_types"] = detect_content_types(content)

            # Add section string if section headers are present
            if "section_headers" in metadata and metadata["section_headers"]:
                metadata["section_str"] = " > ".join(metadata["section_headers"])
                metadata["hierarchy"] = metadata["section_str"]
            else:
                metadata["section_str"] = ""
                metadata["hierarchy"] = ""
                # Ensure section_headers is an empty list instead of null
                metadata["section_headers"] = []

            # Extract relationships if not present
            if "relationships" not in metadata or not metadata["relationships"]:
                if "technical_terms" in metadata and metadata["technical_terms"]:
                    extracted_relationships = extract_hierarchy_relationships(
                        content,
                        known_concepts=set(metadata["technical_terms"]),
                        min_confidence=0.7
                    )
                    metadata["relationships"] = extracted_relationships
                else:
                    metadata["relationships"] = []

            # Create the document with normalized metadata
            return Document(
                page_content=content,
                metadata=normalize_metadata_for_vectorstore(metadata)
            )

        except Exception as e:
            logger.warning(f"Failed to convert chunk to document: {e}")
            return None

    def _update_research_data(self, result: ProcessingResult) -> None:
        """
        Update research manager with document data and cross-document insights.
        Fixed to handle primary_concepts as a list or dictionary.

        Args:
            result: Processing result with document data
        """
        pdf_id = result.pdf_id

        try:
            # Collect document-level data
            terms = set()
            docling_paths = set()
            content_counts = defaultdict(int)
            section_hierarchy = set()

            for element in result.elements:
                # Collect technical terms
                if element.metadata.technical_terms:
                    terms.update(element.metadata.technical_terms)

                # Count content types
                content_counts[element.content_type.value] += 1

                # Track docling references
                if element.metadata.docling_ref:
                    docling_paths.add(element.metadata.docling_ref)

                # Track section headers
                if element.metadata.section_headers:
                    section_hierarchy.add(tuple(element.metadata.section_headers))

            # Add document data to research manager
            self._research_manager.add_document_metadata(
                pdf_id=pdf_id,
                metadata={
                    "total_elements": len(result.elements),
                    "content_types": dict(content_counts),
                    "technical_terms": terms,
                    "hierarchies": list(map(lambda x: " > ".join(x), section_hierarchy)),
                    "primary_concepts": result.concept_network.primary_concepts if result.concept_network else []
                }
            )

            # Register technical terms and primary concepts as shared concepts
            # Set different confidence for primary vs regular concepts
            if result.concept_network:
                # Check if primary_concepts is a list or dictionary, properly handle both
                if hasattr(result.concept_network, 'primary_concepts'):
                    primary_concepts = result.concept_network.primary_concepts
                    try:
                        if isinstance(primary_concepts, list):
                            # Handle as a list
                            for concept in primary_concepts:
                                self._research_manager.register_shared_concept(
                                    concept=concept,
                                    pdf_ids={pdf_id},
                                    confidence=0.95  # High confidence for primary concepts
                                )
                        elif isinstance(primary_concepts, dict):
                            # Handle as a dictionary (if implemented that way elsewhere)
                            for concept, score in primary_concepts.items():
                                self._research_manager.register_shared_concept(
                                    concept=concept,
                                    pdf_ids={pdf_id},
                                    confidence=score
                                )
                        else:
                            # Log the type and handle safely
                            logger.warning(f"Unexpected primary_concepts type: {type(primary_concepts)}")
                            # Try to use it as a list if possible
                            if primary_concepts:
                                for concept in primary_concepts:
                                    try:
                                        self._research_manager.register_shared_concept(
                                            concept=str(concept),
                                            pdf_ids={pdf_id},
                                            confidence=0.9
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to register concept {concept}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error processing primary concepts: {str(e)}")

                # Register concepts from the concept network
                for concept in result.concept_network.concepts:
                    confidence = 0.95 if concept.is_primary else 0.7
                    self._research_manager.register_shared_concept(
                        concept=concept.name,
                        pdf_ids={pdf_id},
                        confidence=confidence
                    )

                # Prepare a safe summary for primary_concepts
                primary_concepts_summary = []
                if result.concept_network.primary_concepts:
                    if isinstance(result.concept_network.primary_concepts, list):
                        primary_concepts_summary = result.concept_network.primary_concepts[:10]
                    elif isinstance(result.concept_network.primary_concepts, dict):
                        primary_concepts_summary = list(result.concept_network.primary_concepts.keys())[:10]

                # Add complete document summary
                self._research_manager.add_document_summary(
                    pdf_id=pdf_id,
                    summary={
                        "primary_concepts": primary_concepts_summary,
                        "total_concepts": len(result.concept_network.concepts),
                        "total_relationships": len(result.concept_network.relationships),
                        "section_coverage": {
                            k: v for k, v in result.concept_network.section_concepts.items()
                        } if hasattr(result.concept_network, 'section_concepts') else {},
                        "concept_network": {
                            "total_concepts": len(result.concept_network.concepts),
                            "total_relationships": len(result.concept_network.relationships),
                            "primary_concepts": primary_concepts_summary
                        },
                        "top_technical_terms": list(terms)[:20]
                    }
                )

            logger.info(f"Updated research data with {len(terms)} technical terms")

        except Exception as e:
            logger.error(f"Failed to update research data: {e}")
            self._metrics.record_error(f"Research data update: {str(e)}")

    def _get_relevant_documents(self, query: str, filter_dict=None, k=None) -> List[Document]:
        """
        Synchronous implementation of document retrieval (required by BaseRetriever).
        Delegates to the async version by creating a temporary event loop.

        Args:
            query: Search query
            filter_dict: Optional filter to apply
            k: Number of results to return

        Returns:
            List of relevant documents
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._aget_relevant_documents(query, filter_dict, k))
        finally:
            loop.close()

    async def _aget_relevant_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: int = None
    ) -> List[Document]:
        """
        Retrieve relevant documents with advanced filtering and scoring.
        Enhanced with domain-specific term boosting and technical documentation awareness.

        Args:
            query: Search query
            filter_dict: Optional filter to apply
            k: Number of results to return

        Returns:
            List of relevant documents
        """
        search_start = time.time()
        try:
            # Apply default filter if not provided
            if filter_dict is None:
                filter_dict = self._get_default_filter()

            # Determine result count (k) if not specified
            if k is None:
                k = self._config.max_results

            # Extract technical terms from query for improved retrieval
            query_terms = extract_technical_terms(query)

            # Identify any domain-specific terms in the query for boosting
            domain_specific_matches = self._identify_domain_specific_terms(query, query_terms)

            # Check if query terms match primary concepts for boosting
            primary_match = False
            if query_terms:
                primary_match = any(term in self._registries.get("primary_concepts", []) for term in query_terms)

            # Dynamically adjust result count based on query complexity
            if primary_match or domain_specific_matches:
                # Retrieve extra results for post-filtering
                # Use higher multiplier for domain-specific queries
                k_multiplier = 1.6 if domain_specific_matches else 1.5 if primary_match else 1.2
                search_k = int(k * k_multiplier)
            else:
                search_k = k

            # Log search information
            logger.info(f"Searching for: {query}. Filter: {filter_dict}, Results: {k}, Terms: {query_terms}")
            self._metrics.record_filter_op()

            # Use hybrid search if available
            if hasattr(self._store, "asimilarity_search_with_relevance_scores"):
                results = await self._store.asimilarity_search_with_relevance_scores(
                    query,
                    k=search_k,
                    filter=filter_dict
                )

                # Extract documents and scores
                documents = []
                for doc, score in results:
                    # Add score to document metadata
                    doc.metadata["score"] = float(score)

                    # Apply enhanced domain-specific boosting
                    self._apply_domain_specific_boosting(doc, query_terms, domain_specific_matches)

                    # Add research context for multi-document mode
                    if self.is_research_active:
                        doc.metadata["research_context"] = self._get_research_context(query)

                    documents.append(doc)

                # Sort by adjusted score
                documents.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)

                # Limit to requested k
                documents = documents[:k]

            else:
                # Fallback to regular search
                documents = await self._store.asimilarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )

            # Record metrics
            search_time = time.time() - search_start
            self._metrics.record_query_time(search_time)
            self._metrics.record_retrieval()

            logger.info(f"Retrieved {len(documents)} documents in {search_time:.2f}s")

            return documents

        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            self._metrics.record_error(f"Document retrieval: {str(e)}")

            # Return empty result for graceful failure
            return []

def _identify_domain_specific_terms(self, query: str, query_terms: List[str]) -> Dict[str, List[str]]:
    """
    Identify domain-specific terms in the query for specialized boosting.
    Optimized for Tridium Niagara framework terminology.

    Args:
        query: Original search query
        query_terms: Extracted technical terms from query

    Returns:
        Dict[str, List[str]]: Dictionary mapping domain categories to matched terms
    """
    query_lower = query.lower()
    matches = {}

    # Check each domain category for matches
    for category, terms in DOMAIN_SPECIFIC_TERMS.items():
        category_matches = []
        for term in terms:
            if term.lower() in query_lower:
                category_matches.append(term)

        if category_matches:
            matches[category] = category_matches

    # Also check extracted terms against domain terms
    if query_terms:
        for term in query_terms:
            term_lower = term.lower()

            # Check if this term matches any domain category
            for category, domain_terms in DOMAIN_SPECIFIC_TERMS.items():
                if category in matches:
                    continue  # Already have matches for this category

                for domain_term in domain_terms:
                    # Direct or partial match
                    if term_lower == domain_term.lower() or term_lower in domain_term.lower() or domain_term.lower() in term_lower:
                        if category not in matches:
                            matches[category] = []
                        matches[category].append(term)
                        break

    return matches

def _apply_domain_specific_boosting(
    self,
    doc: Document,
    query_terms: List[str],
    domain_matches: Dict[str, List[str]]
) -> None:
    """
    Apply domain-specific score boosting to document results.

    This method enhances result ranking by prioritizing documents that match
    technical terms from the Tridium Niagara framework domain.

    Args:
        doc: Document to apply boosting to
        query_terms: Technical terms from query
        domain_matches: Domain-specific matches from query
    """
    # Initialize total boost factor
    total_boost = 1.0
    boost_reasons = []

    # 1. Apply basic technical term matching boost
    if query_terms and "technical_terms" in doc.metadata:
        doc_terms = doc.metadata["technical_terms"]

        # Check for primary concept matches
        primary_matches = []
        for term in query_terms:
            # Check if this term is in the document AND is a primary concept
            if term in doc_terms and term in self._registries.get("primary_concepts", []):
                primary_matches.append(term)

        # Apply primary concept boost
        if primary_matches:
            primary_boost = 1.0 + (len(primary_matches) * 0.1)  # Increased from 0.05
            total_boost *= primary_boost
            doc.metadata["matched_primary_concepts"] = primary_matches
            boost_reasons.append(f"primary_concepts:{len(primary_matches)}")

    # 2. Apply domain-specific category boosting
    if domain_matches and "technical_terms" in doc.metadata:
        doc_terms = doc.metadata["technical_terms"]

        # Track matches by category
        category_matches = {}

        # Check each domain category
        for category, terms in domain_matches.items():
            category_match_count = 0

            # Count matches within this category
            for term in terms:
                term_lower = term.lower()
                for doc_term in doc_terms:
                    if doc_term.lower() == term_lower or term_lower in doc_term.lower():
                        category_match_count += 1
                        break

            if category_match_count > 0:
                category_matches[category] = category_match_count

        # Apply category-specific boosts
        if category_matches:
            # Calculate boost based on category importance and match count
            category_weights = {
                "visualization": 0.15,
                "trend": 0.15,
                "interval": 0.15,
                "node": 0.12,
                "hierarchy": 0.12,
                "station": 0.1,
                "component": 0.1,
                # Default weight for other categories
                "_default": 0.05
            }

            # Apply boosts for each matching category
            for category, match_count in category_matches.items():
                weight = category_weights.get(category, category_weights["_default"])
                category_boost = 1.0 + (match_count * weight)
                total_boost *= category_boost
                boost_reasons.append(f"{category}:{match_count}")

    # 3. Apply section context boost
    if "section_path" in doc.metadata or "section_str" in doc.metadata:
        # Get section path
        section_path = None
        if "section_path" in doc.metadata and doc.metadata["section_path"]:
            section_path = doc.metadata["section_path"]
        elif "section_str" in doc.metadata and doc.metadata["section_str"]:
            section_path = doc.metadata["section_str"].split(" > ")

        if section_path:
            # Check if section matches query terms
            section_match = False
            for term in query_terms:
                term_lower = term.lower()
                for section in section_path:
                    if term_lower in section.lower():
                        section_match = True
                        break
                if section_match:
                    break

            # Apply section context boost
            if section_match:
                section_boost = 1.2  # 20% boost for section match
                total_boost *= section_boost
                boost_reasons.append("section_match")

    # Apply the total boost to the document score
    original_score = doc.metadata.get("score", 0)
    doc.metadata["score"] = original_score * total_boost
    doc.metadata["boost_factor"] = total_boost
    doc.metadata["boost_reasons"] = boost_reasons

    logger.debug(f"Applied boosting factor {total_boost:.2f} to document, " +
                f"raising score from {original_score:.3f} to {doc.metadata['score']:.3f}, " +
                f"reasons: {boost_reasons}")

    def _get_default_filter(self) -> Dict[str, Any]:
        """
        Get default filter based on current research mode state.

        This method has been refactored to ensure consistent filtering
        based on the research mode state, ensuring proper document
        retrieval in both single and multiple document modes.

        Returns:
            Dict[str, Any]: Filter dictionary for vector store queries
        """
        # Check research mode status using the consistent property
        research_active = self.is_research_active

        # In single document mode, filter by the primary PDF ID
        if not research_active:
            return {"pdf_id": self._config.pdf_id}

        # In research mode, filter by all active PDF IDs
        if hasattr(self._research_manager, "active_pdf_ids"):
            active_ids = list(self._research_manager.active_pdf_ids)
            if active_ids:
                return {"pdf_id": {"$in": active_ids}}

        # Default to empty filter if no active PDFs (should not happen)
        logger.warning("No active PDF IDs found for research mode filter")
        return {}

    def _sanitize_filter_for_pinecone(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the filter dictionary is Pinecone-compatible with enhanced validation.
        Removes any operators not supported by Pinecone and handles domain-specific filtering.

        Args:
            filter_dict: Original filter dictionary

        Returns:
            Dict[str, Any]: Sanitized filter dictionary compatible with Pinecone
        """
        if not filter_dict:
            logger.debug("Empty filter dictionary provided to sanitize")
            return {}

        # Pinecone supported operators (limited subset of MongoDB operators)
        # $in, $gte, $gt, $lte, $lt, $eq, $neq, $and, $or
        supported = ["$in", "$gte", "$gt", "$lte", "$lt", "$eq", "$neq", "$and", "$or"]

        # Known unsupported operators
        unsupported = ["$regex", "$text", "$search", "$where", "$elemMatch", "$exists", "$type"]

        result = {}

        for key, value in filter_dict.items():
            # Skip unsupported operator keys
            if key in unsupported:
                logger.warning(f"Removing unsupported Pinecone operator: {key}")
                continue

            # Handle domain-specific key transformations
            if key == "domain_capability" and isinstance(value, str):
                # Convert domain capability to appropriate technical terms filter
                domain_terms = []
                # Mapping for Tridium Niagara framework capabilities to search terms
                domain_mapping = {
                    "visualization": ["trend", "chart", "graph", "plot", "visualization"],
                    "time_interval": ["interval", "timeshift", "hour", "minute", "daily"],
                    "hierarchy": ["hierarchy", "nav", "tree", "navigation"],
                    "station": ["station", "jace", "supervisor", "controller"]
                }

                # Get relevant terms for this capability
                if value in domain_mapping:
                    domain_terms = domain_mapping[value]
                    # Create a filter for these terms in technical_terms field
                    if domain_terms:
                        result["technical_terms"] = {"$in": domain_terms}
                        # Log the transformation for debugging
                        logger.info(f"Transformed domain filter '{value}' to technical terms: {domain_terms}")
                        continue

            # Process nested dictionaries
            if isinstance(value, dict):
                # Check for unsupported operators
                if any(op in value for op in unsupported):
                    logger.warning(f"Skipping condition with unsupported operators in '{key}': {value}")
                    # Skip this condition entirely
                    continue
                else:
                    # Recursively sanitize nested dict
                    result[key] = self._sanitize_filter_for_pinecone(value)
            else:
                # Ensure values are proper types for Pinecone
                if isinstance(value, (list, set)):
                    # Convert sets to lists for Pinecone
                    result[key] = list(value)
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    # Keep basic types as-is
                    result[key] = value
                else:
                    # Convert other types to strings
                    try:
                        result[key] = str(value)
                        logger.warning(f"Converting non-basic type to string for key '{key}': {type(value)}")
                    except:
                        logger.warning(f"Skipping unconvertible value for key '{key}': {value}")
                        continue

        # Ensure we have at least one valid filter condition
        if not result and filter_dict:
            logger.warning(f"All filter conditions were invalid. Original filter: {filter_dict}")

            # Try to provide a safe fallback
            if "pdf_id" in filter_dict:
                result["pdf_id"] = str(filter_dict["pdf_id"])
                logger.info("Added fallback filter on pdf_id")

        return result

    def _get_research_context(self, query: str) -> Dict[str, Any]:
        """
        Get research context for multi-document analysis.

        Args:
            query: Search query

        Returns:
            Research context information
        """
        if not hasattr(self._research_manager, "get_research_context"):
            return {}

        try:
            return self._research_manager.get_research_context(query)
        except Exception as e:
            logger.error(f"Failed to get research context: {e}")
            self._metrics.record_error(f"Research context: {str(e)}")
            return {}

    def ensure_research_mode_consistency(self) -> bool:
        """
        Ensure research mode status is consistently set across components.

        This method centralizes the logic for checking and maintaining
        consistent research mode state across the system. It verifies the
        state in the research manager and returns the definitive status.

        Returns:
            bool: True if research mode is active, False otherwise
        """
        try:
            # Determine correct research mode status
            is_active = (
                self._research_manager and
                hasattr(self._research_manager, "context") and
                self._research_manager.context.mode == ResearchMode.MULTI
            )

            # Check for multiple active PDF IDs
            has_multiple_pdfs = False
            if hasattr(self._research_manager, "active_pdf_ids"):
                # Normalize PDF IDs to ensure consistent comparison
                from app.chat.utils import normalize_pdf_id
                active_ids = [normalize_pdf_id(pid) for pid in self._research_manager.active_pdf_ids]
                # Filter out empty or invalid IDs
                valid_ids = [pid for pid in active_ids if pid]
                has_multiple_pdfs = len(valid_ids) > 1

            # CRITICAL: Only true if BOTH conditions are met
            final_status = is_active and has_multiple_pdfs

            logger.info(f"Research mode consistency check: active={final_status}, " +
                       f"pdf_count={len(valid_ids) if 'valid_ids' in locals() else 0}")

            return final_status
        except Exception as e:
            logger.error(f"Error in research mode consistency check: {str(e)}")
            # Default to the property value if available
            if hasattr(self, 'is_research_active'):
                return self.is_research_active
            return False

    async def search_content(
        self,
        query: str,
        research_mode: bool = False,
        content_types: Optional[List[str]] = None,
        min_score: float = 0.65,
        section_filter: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search content with advanced filtering and research mode awareness.

        This method has been refactored to ensure consistent research mode handling
        and to improve the quality and relevance of search results.

        Args:
            query: Search query
            research_mode: Override for research mode
            content_types: Filter by content types
            min_score: Minimum relevance score
            section_filter: Filter by document section
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        try:
            start_time = time.time()

            # CRITICAL: Ensure research mode consistency before searching
            # This overrides the passed research_mode parameter with the definitive state
            is_research_active = self.ensure_research_mode_consistency()

            logger.info(f"Starting enhanced search for: '{query[:50]}...', research_mode={is_research_active}, " +
                        f"content_types={content_types}, section_filter={section_filter}")

            # Extract technical terms from query to improve search
            query_terms = extract_technical_terms(query)
            if query_terms:
                logger.info(f"Extracted technical terms: {query_terms}")

            # Determine if query suggests specific content types
            content_type_hints = self._detect_content_type_hints(query)
            is_visual_query = "image" in content_type_hints
            is_table_query = "table" in content_type_hints
            is_code_query = "code" in content_type_hints

            # Build filter dictionary with Pinecone compatibility in mind
            filter_dict = {}

            # Configure PDF filtering - ALWAYS use the definitive research mode state
            if is_research_active and hasattr(self._research_manager, "active_pdf_ids"):
                active_ids = list(self._research_manager.active_pdf_ids)
                if active_ids:
                    filter_dict["pdf_id"] = {"$in": active_ids}
            else:
                filter_dict["pdf_id"] = self._config.pdf_id

            # Configure content type filtering
            if content_types:
                filter_dict["content_type"] = {"$in": content_types}
            elif any([is_visual_query, is_table_query, is_code_query]):
                # Build prioritized content types based on query
                prioritized_types = ["text"]  # Always include text

                if is_visual_query:
                    prioritized_types.append("image")
                if is_table_query:
                    prioritized_types.append("table")
                if is_code_query:
                    prioritized_types.append("code")

                filter_dict["content_type"] = {"$in": prioritized_types}

            # Configure section filtering
            if section_filter:
                filter_dict["section_str"] = section_filter

            # Determine result count
            if not max_results:
                max_results = self._config.max_results

            # Increase results for queries matching primary concepts for better filtering
            search_k = max_results
            if query_terms and any(term in self._registries.get("primary_concepts", []) for term in query_terms):
                search_k = int(max_results * 1.5)

            # Sanitize filter to ensure Pinecone compatibility
            sanitized_filter = self._sanitize_filter_for_pinecone(filter_dict)
            logger.info(f"Sanitized Pinecone filter: {sanitized_filter}")

            # Perform search with optimized parameters
            results = await self._aget_relevant_documents(
                query=query,
                filter_dict=sanitized_filter,
                k=search_k
            )

            # Log search time and result details
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s with {len(results)} results")

            # Apply minimum score filter
            filtered_results = [
                doc for doc in results
                if doc.metadata.get("score", 0) >= min_score
            ]

            # Balance results by content type
            balanced_results = self._balance_results_by_type(
                filtered_results,
                favor_visual=is_visual_query,
                favor_tables=is_table_query,
                favor_code=is_code_query,
                max_results=max_results
            )

            # Convert to search result format
            search_results = [
                self._document_to_search_result(doc)
                for doc in balanced_results
            ]

            # Add cross-document insights if in research mode
            if is_research_active and len(search_results) > 0:
                search_results = self._enhance_with_research_insights(search_results, query)

            # Add relationship insights from concept network
            search_results = self._enhance_with_relationship_insights(search_results, query_terms)

            logger.info(f"Search completed with {len(search_results)} balanced results")
            return search_results

        except Exception as e:
            logger.error(f"Enhanced search failed: {str(e)}", exc_info=True)
            self._metrics.record_error(f"Enhanced search: {str(e)}")
            return []

    def _detect_content_type_hints(self, query: str) -> Set[str]:
        """
        Detect content type hints from query.

        Args:
            query: Search query

        Returns:
            Set of detected content types
        """
        hints = set()
        query_lower = query.lower()

        # Visual content hints
        visual_terms = {
            "image", "picture", "photo", "diagram", "figure", "illustration",
            "screenshot", "chart", "graph", "visualize", "show me", "draw", "plot"
        }

        # Table content hints
        table_terms = {
            "table", "tabular", "grid", "data", "statistics", "metrics",
            "numbers", "rows", "columns", "values", "spreadsheet", "excel"
        }

        # Code content hints
        code_terms = {
            "code", "function", "class", "method", "script", "programming",
            "algorithm", "implementation", "syntax", "snippet", "python", "javascript"
        }

        # Check for term matches
        query_terms = set(query_lower.split())

        if query_terms & visual_terms:
            hints.add("image")

        if query_terms & table_terms:
            hints.add("table")

        if query_terms & code_terms:
            hints.add("code")

        return hints

    def _balance_results_by_type(
        self,
        results: List[Document],
        favor_visual: bool = False,
        favor_tables: bool = False,
        favor_code: bool = False,
        max_results: int = 5
    ) -> List[Document]:
        """
        Balance results to include diverse content types based on query intent.
        Enhanced for technical documentation with domain-specific allocation.

        Args:
            results: List of retrieved documents
            favor_visual: Whether to prioritize visual content
            favor_tables: Whether to prioritize tabular content
            favor_code: Whether to prioritize code content
            max_results: Maximum number of results

        Returns:
            Balanced list of documents
        """
        # Group results by content type
        by_type = defaultdict(list)
        for doc in results:
            content_type = doc.metadata.get("content_type", "text")
            by_type[content_type].append(doc)

        # Track domain-specific content
        trend_results = []
        timeshift_results = []
        hierarchy_results = []

        # Identify domain-specific results
        for doc in results:
            terms = doc.metadata.get("technical_terms", [])
            if any(term.lower() in ["trend", "chart", "graph", "plot"] for term in terms):
                trend_results.append(doc)
            if any(term.lower() in ["timeshift", "interval", "hours", "minutes"] for term in terms):
                timeshift_results.append(doc)
            if any(term.lower() in ["hierarchy", "nav", "navtree"] for term in terms):
                hierarchy_results.append(doc)

        # Check if any results have primary concept matches
        has_primary_matches = any(
            "matched_primary_concepts" in doc.metadata
            for doc in results
        )

        # Determine allocation based on query intent and domain context
        if trend_results and timeshift_results:
            # This is a trend visualization with time intervals query - highly specialized
            allocation = {
                "text": 0.4,
                "code": 0.3,  # Code likely contains SeriesTransform nodes
                "image": 0.2,  # Images likely contain charts
                "table": 0.1
            }
        elif trend_results:
            # Trend visualization query
            allocation = {
                "text": 0.4,
                "image": 0.3,  # Prioritize images for visualization
                "code": 0.2,
                "table": 0.1
            }
        elif timeshift_results:
            # Time interval query
            allocation = {
                "text": 0.4,
                "code": 0.3,  # Code samples for timeshift configuration
                "image": 0.2,
                "table": 0.1
            }
        elif hierarchy_results:
            # Hierarchy navigation query
            allocation = {
                "text": 0.5,
                "image": 0.3,  # Hierarchy diagrams
                "code": 0.1,
                "table": 0.1
            }
        elif favor_visual:
            # Prioritize images
            allocation = {
                "image": 0.5,
                "text": 0.3,
                "table": 0.1,
                "code": 0.1
            }
        elif favor_tables:
            # Prioritize tables
            allocation = {
                "table": 0.5,
                "text": 0.3,
                "image": 0.1,
                "code": 0.1
            }
        elif favor_code:
            # Prioritize code
            allocation = {
                "code": 0.5,
                "text": 0.3,
                "image": 0.1,
                "table": 0.1
            }
        elif has_primary_matches:
            # Prioritize text matches with primary concepts
            allocation = {
                "text": 0.7,
                "image": 0.1,
                "table": 0.1,
                "code": 0.1
            }
        else:
            # Standard allocation - prioritize text
            allocation = {
                "text": 0.6,
                "image": 0.15,
                "table": 0.15,
                "code": 0.1
            }

        # Calculate slots for each type
        total_slots = min(max_results, len(results))
        slots = {ct: max(1, int(total_slots * alloc)) for ct, alloc in allocation.items() if ct in by_type}

        # Adjust if total exceeds max_results
        while sum(slots.values()) > total_slots:
            # Find type with most slots and reduce by 1
            max_type = max(slots.items(), key=lambda x: x[1])[0]
            slots[max_type] -= 1

        # Fill remaining slots if any
        remaining = total_slots - sum(slots.values())
        if remaining > 0:
            # Distribute remaining slots proportionally
            for ct in sorted(allocation.keys(), key=lambda x: allocation[x], reverse=True):
                if ct in slots and remaining > 0:
                    slots[ct] += 1
                    remaining -= 1

        # Create balanced results
        balanced = []

        # First, prioritize documents with primary concept matches
        if has_primary_matches:
            primary_matches = [
                doc for doc in results
                if "matched_primary_concepts" in doc.metadata
            ]
            # Take up to half of total slots for primary matches
            primary_slot_count = min(len(primary_matches), total_slots // 2)
            balanced.extend(sorted(primary_matches, key=lambda x: x.metadata.get("score", 0), reverse=True)[:primary_slot_count])

            # Adjust remaining slots
            remaining_slots = total_slots - len(balanced)
            # Scale down other type allocations proportionally
            scale_factor = remaining_slots / total_slots
            slots = {ct: max(1, int(count * scale_factor)) for ct, count in slots.items()}

            # Remove already added documents
            results = [r for r in results if r not in balanced]

        # Add highest scoring results of each type up to slot count
        for content_type, slot_count in slots.items():
            type_results = by_type.get(content_type, [])
            # Skip if we've already exhausted slots
            if len(balanced) >= total_slots:
                break

            # Calculate how many more of this type to add
            to_add = min(slot_count, total_slots - len(balanced))
            if to_add <= 0:
                continue

            # Sort by score
            sorted_results = sorted(type_results, key=lambda x: x.metadata.get("score", 0), reverse=True)
            # Add only results not already included
            type_to_add = [r for r in sorted_results if r not in balanced][:to_add]
            balanced.extend(type_to_add)

        # Fill any remaining slots with highest scoring results regardless of type
        if len(balanced) < total_slots:
            # Get all results not already included
            remaining = [r for r in results if r not in balanced]
            # Sort by score
            remaining_sorted = sorted(remaining, key=lambda x: x.metadata.get("score", 0), reverse=True)
            # Add up to the limit
            balanced.extend(remaining_sorted[:total_slots - len(balanced)])

        # Final sort by score
        return sorted(balanced, key=lambda x: x.metadata.get("score", 0), reverse=True)


    def _document_to_search_result(self, doc: Document) -> Dict[str, Any]:
        """
        Convert Document to standardized search result format.

        Args:
            doc: Document from vector store

        Returns:
            Search result dictionary
        """
        metadata = doc.metadata.copy()
        score = metadata.pop("score", 0.0)
        pdf_id = metadata.get("pdf_id", self._config.pdf_id)

        # Create base result
        result = {
            "content": doc.page_content,
            "element_id": metadata.get("element_id", metadata.get("chunk_id", str(uuid.uuid4()))),
            "pdf_id": pdf_id,
            "content_type": metadata.get("content_type", "text"),
            "page_number": metadata.get("page_number", 0),
            "score": score,
            "metadata": metadata
        }

        # Add section context
        if "section_path" in metadata:
            result["section_path"] = metadata["section_path"]
        elif "section_str" in metadata:
            result["section_path"] = metadata["section_str"].split(" > ")

        # Add content-specific fields
        content_type = metadata.get("content_type", "text")

        if content_type == "image" and metadata.get("is_image", False):
            # Add image-specific fields
            result["image_path"] = metadata.get("image_path")
            result["image_description"] = metadata.get("image_description", "")

        elif content_type == "table" and metadata.get("is_table", False):
            # Add table-specific fields
            result["table_caption"] = metadata.get("table_caption", "")
            result["table_headers"] = metadata.get("table_headers", [])

        elif content_type == "code" and metadata.get("is_code", False):
            # Add code-specific fields
            result["code_language"] = metadata.get("code_language", "")

        return result

    def _enhance_with_research_insights(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Enhance search results with cross-document research insights.

        Args:
            results: List of search results
            query: Original search query

        Returns:
            Enhanced search results
        """
        if not self.is_research_active or not results:
            return results

        try:
            # Get research context
            research_context = self._research_manager.get_research_context(query)

            # Track PDF IDs in results
            result_pdf_ids = {r["pdf_id"] for r in results}

            # Generate insights
            insights = []

            # Add shared concept insights
            if "shared_concepts" in research_context:
                shared_concepts = research_context["shared_concepts"]
                top_concepts = [c["concept"] for c in shared_concepts[:3]]
                if top_concepts:
                    insights.append(f"Concepts {', '.join(top_concepts)} appear across documents")

            # Add document relationship insights
            doc_relationships = research_context.get("document_relationships", {})
            for pdf_id in result_pdf_ids:
                if pdf_id in doc_relationships:
                    for rel in doc_relationships[pdf_id][:2]:  # Limit to top 2
                        insights.append(f"Document {pdf_id} is related to {rel['pdf_id']} through {rel['relationship_types']}")

            # Add technical overlap insights
            tech_overlaps = research_context.get("technical_overlaps", {})
            if tech_overlaps:
                # Get first overlap
                first_key = next(iter(tech_overlaps))
                first_overlap = tech_overlaps[first_key]
                if first_overlap:
                    insights.append(f"Technical overlap between documents: {', '.join(list(first_overlap)[:3])}")

            # Add general cross-document insights
            cross_insights = research_context.get("cross_document_insights", [])
            if cross_insights:
                insights.extend(cross_insights[:2])  # Add top 2

            # Add insights to results
            for result in results:
                result["research_insights"] = insights

                # Add document relationships
                pdf_id = result["pdf_id"]
                if pdf_id in doc_relationships:
                    result["related_documents"] = [
                        {
                            "pdf_id": rel["pdf_id"],
                            "relationship_type": rel["relationship_types"][0] if rel["relationship_types"] else "related",
                            "strength": rel["relationship_strength"]
                        }
                        for rel in doc_relationships[pdf_id][:3]  # Top 3 related docs
                    ]

            return results

        except Exception as e:
            logger.error(f"Failed to enhance with research insights: {e}")
            return results

    def _enhance_with_relationship_insights(
        self,
        results: List[Dict[str, Any]],
        query_terms: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Enhance search results with concept relationship insights.
        Optimized for domain-specific technical relationships.

        Args:
            results: List of search results
            query_terms: Technical terms extracted from query

        Returns:
            Enhanced search results with relationship insights
        """
        if not results or not query_terms:
            return results

        try:
            # Look for relationships involving query terms
            relevant_relationships = []

            # Check each query term for relationships
            for term in query_terms:
                # Get direct relationships
                if term in self._registries["relationships"]:
                    # Get top 5 relationships by weight
                    term_rels = sorted(
                        self._registries["relationships"][term],
                        key=lambda r: r.get("weight", 0),
                        reverse=True
                    )[:5]  # Increased from 3 to 5 for more comprehensive context

                    # Add to relevant relationships
                    relevant_relationships.extend(term_rels)

            # Add domain-specific relationships if available
            domain_specific_added = False
            domain_terms = {
                "trend": ["timeshift", "interval", "history", "chart"],
                "timeshift": ["trend", "interval", "history"],
                "station": ["jace", "supervisor", "controller"],
                "hierarchy": ["nav", "tree", "navigation"]
            }

            # Add domain-specific relationships for context
            for query_term in query_terms:
                term_lower = query_term.lower()
                # Check if this is a domain term
                for domain_term, related_terms in domain_terms.items():
                    if domain_term in term_lower or term_lower in domain_term:
                        # Add implied relationships
                        for related in related_terms:
                            relevant_relationships.append({
                                "source": domain_term,
                                "target": related,
                                "type": "related_to",
                                "weight": 0.9,
                                "context": "Domain-specific relationship"
                            })
                        domain_specific_added = True
                        break

            # Skip if no relevant relationships
            if not relevant_relationships:
                return results

            # Create relationship insights
            relationship_insights = []
            seen_pairs = set()  # Track relationships we've already added

            for rel in relevant_relationships:
                # Create a key to avoid duplicates
                rel_key = (rel["source"], rel["target"], rel["type"])
                if rel_key in seen_pairs:
                    continue
                seen_pairs.add(rel_key)

                # Format the relationship as insight
                rel_type = rel["type"]
                source = rel["source"]
                target = rel["target"]

                # Skip self-references
                if source == target:
                    continue

                insight = f"{source} {rel_type} {target}"
                if "context" in rel and rel["context"]:
                    insight += f" ({rel['context']})"

                relationship_insights.append(insight)

            # Add insights to results
            for result in results:
                # Get technical terms in this result
                result_terms = result["metadata"].get("technical_terms", [])

                # Find related terms
                related_terms = set()
                for term in result_terms:
                    # Check if term is in query
                    if term in query_terms:
                        # Get all relationships for this term
                        if term in self._registries["relationships"]:
                            for rel in self._registries["relationships"][term]:
                                # Add related concept
                                if rel["source"] == term:
                                    related_terms.add(rel["target"])
                                else:
                                    related_terms.add(rel["source"])

                # If we found related terms, add as insight
                if related_terms:
                    # Add relationship insights
                    if "concept_relationships" not in result:
                        result["concept_relationships"] = []

                    # Add only relevant relationships (increased from 3 to 5)
                    result["concept_relationships"] = relationship_insights[:5]

                    # Add related terms
                    result["related_concepts"] = list(related_terms)

                # Add domain context if this was a domain-specific query
                elif domain_specific_added:
                    # Find domain terms in the result
                    result_domain_terms = []
                    for term in result_terms:
                        term_lower = term.lower()
                        for domain_term in domain_terms:
                            if domain_term in term_lower or term_lower in domain_term:
                                result_domain_terms.append(term)

                    if result_domain_terms:
                        # Add domain relationships
                        if "concept_relationships" not in result:
                            result["concept_relationships"] = []

                        # Add domain relationships
                        if not result["concept_relationships"]:
                            result["concept_relationships"] = relationship_insights[:3]

                        # Add domain context
                        result["domain_context"] = "This content relates to Tridium Niagara framework concepts"

            return results

        except Exception as e:
            logger.error(f"Failed to enhance with relationship insights: {e}")
            return results

    def get_related_docs(self, pdf_id: str) -> List[Dict[str, Any]]:
        """
        Get documents related to specified PDF with relationship strength.

        Args:
            pdf_id: PDF ID to find related documents for

        Returns:
            List of dictionaries with related document information
        """
        # Delegate to research manager for consistency
        try:
            related_docs = self._research_manager.get_related_documents(pdf_id)

            # Format results
            result = []
            for doc in related_docs:
                result.append({
                    "id": doc["pdf_id"],
                    "shared_concepts": doc.get("shared_concepts", []),
                    "relationship_strength": doc.get("relationship_strength", 0.5)
                })

            return result

        except Exception as e:
            logger.error(f"Failed to get related documents: {e}")
            return []

    def get_section_hierarchy(self) -> Dict[str, Any]:
        """
        Get hierarchical section structure of the document.

        Returns:
            Nested dictionary representing document's section hierarchy
        """
        try:
            hierarchy = {}

            # Extract section relationships from registry
            sections = set()
            for section_path in self._registries["section_elements"].keys():
                # Add each section level to the set
                for i in range(1, len(section_path) + 1):
                    sections.add(section_path[:i])

            # Build parent-child relationships
            parent_map = defaultdict(list)
            for section in sorted(sections, key=len):
                if len(section) > 1:
                    parent = section[:-1]
                    child = section[-1]
                    parent_map[parent].append(child)
                elif len(section) == 1:
                    # Top level section
                    if () not in parent_map:
                        parent_map[()].append(section[0])

            # Build hierarchical tree
            def build_tree(parent=()):
                result = {}
                for child in parent_map.get(parent, []):
                    if parent:
                        new_path = parent + (child,)
                    else:
                        new_path = (child,)

                    # Get concepts for this section
                    section_str = " > ".join(new_path)
                    concepts = self._registries["section_concepts"].get(section_str, [])

                    result[child] = {
                        "children": build_tree(new_path),
                        "concepts": concepts[:10],  # Limit to top 10 concepts
                        "elements": len(self._registries["section_elements"].get(new_path, [])),
                    }
                return result

            return build_tree()

        except Exception as e:
            logger.error(f"Failed to get section hierarchy: {e}")
            return {}

    @property
    def is_research_active(self) -> bool:
        """
        Check if research mode is active with improved reliability.

        This property has been refactored to ensure consistent and reliable
        determination of research mode status across the entire pipeline.

        The research mode is considered active if and only if:
        1. The research manager has been initialized
        2. The research manager context mode is set to MULTI
        3. There are at least 2 active PDF IDs

        Returns:
            bool: True if research mode is active, False otherwise
        """
        if not self._research_manager:
            logger.debug("Research mode check: No research manager available")
            return False

        # Check research manager mode
        is_multi_mode = (
            hasattr(self._research_manager, "context") and
            self._research_manager.context.mode == ResearchMode.MULTI
        )

        # Check for multiple active PDF IDs
        has_multiple_pdfs = False
        if hasattr(self._research_manager, "active_pdf_ids"):
            # Normalize PDF IDs to ensure consistent comparison
            from app.chat.utils import normalize_pdf_id
            active_ids = [normalize_pdf_id(pid) for pid in self._research_manager.active_pdf_ids]
            # Filter out empty or invalid IDs
            valid_ids = [pid for pid in active_ids if pid]
            has_multiple_pdfs = len(valid_ids) > 1

            if is_multi_mode and not has_multiple_pdfs:
                logger.warning(f"Research mode inconsistency: mode=MULTI but only {len(valid_ids)} valid PDF IDs")

        # CRITICAL: Only true if BOTH conditions are met
        result = is_multi_mode and has_multiple_pdfs

        # Log the decision for debugging
        if result:
            logger.debug(f"Research mode active with {len(self._research_manager.active_pdf_ids)} PDFs")

        return result

    async def cleanup(self) -> None:
        """Clean up resources and save metrics."""
        try:
            # Save metrics
            metrics_path = Path("output") / self._config.pdf_id / "metadata" / "vector_store_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_path, "w") as f:
                f.write(json.dumps(self._metrics.dict(), default=str))

            # Clear registries
            for registry in self._registries.values():
                if hasattr(registry, "clear"):
                    registry.clear()

            logger.info(f"Storage cleanup completed for {self._config.pdf_id}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise VectorStoreError(self._config.pdf_id, f"Cleanup failed: {e}")

    async def delete_vectors(self, pdf_id: str) -> None:
        """
        Delete all vectors for a specific PDF from the vector store.

        Args:
            pdf_id: The ID of the PDF whose vectors should be deleted
        """
        if not self._store:
            logger.warning(f"Vector store not initialized, cannot delete vectors for {pdf_id}")
            return

        try:
            # Normalize PDF ID to ensure consistent filtering
            if hasattr(self, '_config') and hasattr(self._config, 'pdf_id'):
                self._config.pdf_id = str(pdf_id)

            # Create a filter to match all vectors for this PDF ID
            filter_dict = {"pdf_id": str(pdf_id)}

            logger.info(f"Deleting vectors for PDF {pdf_id} using filter")

            try:
                # For Pinecone, access the underlying index via vector store
                # This approach is safer since it doesn't rely on the exact structure
                if hasattr(self._store, '_index'):
                    namespace = "all_documents"  # Use the same namespace as in _initialize_store
                    await self._store._index.delete(filter=filter_dict, namespace=namespace)
                    logger.info(f"Successfully deleted vectors for PDF {pdf_id} using filter")
                else:
                    # Alternative approach if _index is not available
                    # Try to delete by query matching the PDF ID
                    # This fallback may vary based on the vector store implementation
                    logger.warning(f"Could not access vector store index, trying direct filter approach")

                    # Using asimple filter if available
                    if hasattr(self._store, 'adelete'):
                        await self._store.adelete(filter=filter_dict)
                        logger.info(f"Successfully deleted vectors for PDF {pdf_id} using adelete")
                    else:
                        logger.error(f"No deletion method available, vectors for PDF {pdf_id} not deleted")
            except Exception as e:
                logger.warning(f"Primary deletion method failed: {e}, trying fallback")

                # Fallback approach - delete matching documents
                results = await self._store.asimilarity_search(
                    query="",
                    filter=filter_dict,
                    k=1000  # Get a large batch of vectors
                )

                if hasattr(self._store, 'adelete'):
                    # Extract vector IDs if available in metadata
                    vector_ids = []
                    for doc in results:
                        if hasattr(doc, 'metadata') and 'id' in doc.metadata:
                            vector_ids.append(doc.metadata['id'])

                    if vector_ids:
                        await self._store.adelete(ids=vector_ids)
                        logger.info(f"Successfully deleted {len(vector_ids)} vectors for PDF {pdf_id} by ID")
                    else:
                        logger.info(f"No vector IDs found for PDF {pdf_id}, deletion may be incomplete")

            if hasattr(self, '_metrics'):
                self._metrics.record_filter_op()

            logger.info(f"Successfully deleted vectors for PDF {pdf_id}")

        except Exception as e:
            logger.error(f"Error deleting vectors for PDF {pdf_id}: {str(e)}")
            if hasattr(self, '_metrics'):
                self._metrics.record_error(f"Vector deletion: {str(e)}")
            # Don't raise the exception, log it and continue
            logger.error(f"Vector deletion failed, but continuing: {str(e)}")


# Global vector store instance
_vector_store = None

def get_storage(pdf_id=None) -> Union['TechDocVectorStore', None]:
    """
    Get the singleton vector store instance.
    
    Args:
        pdf_id: Optional PDF ID to filter by
        
    Returns:
        Vector store instance
    """
    global _vector_store
    
    try:
        # Initialize if not already done
        if _vector_store is None:
            logger.info("Initializing vector store for the first time")
            _vector_store = TechDocVectorStore()
            
            # Verify initialization was successful
            if not _vector_store.initialized:
                logger.error("Vector store failed to initialize properly")
                return None
        
        # Return the instance (it will handle PDF ID filtering internally)
        return _vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        return None


class TechDocVectorStore:
    """Vector store implementation for technical documents."""
    
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
            # Create Pinecone client instance instead of using init()
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
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
            
        # Validate inputs
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query type: {type(query)}")
            return []
            
        # Ensure k is valid
        if not isinstance(k, int) or k < 1:
            logger.warning(f"Invalid k value: {k}, using default k=5")
            k = 5
            
        try:
            # Create filter if pdf_id provided
            filter_dict = None
            if pdf_id:
                filter_dict = {"pdf_id": pdf_id}
                logger.info(f"Filtering search by PDF ID: {pdf_id}")
                
            # Perform the search with error measurement
            start_time = time.time()
            docs = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            elapsed = time.time() - start_time
            
            # Record metrics
            self.metrics.record_query_time(elapsed)
            self.metrics.total_queries += 1
            
            if not docs:
                logger.warning(f"No documents found for query: {query}")
                return []
                
            logger.info(f"Found {len(docs)} documents for query in {elapsed:.2f}s")
            return docs
            
        except Exception as e:
            self.metrics.record_error(str(e))
            logger.error(f"Error during similarity search: {str(e)}", exc_info=True)
            # Return empty list instead of propagating error
            return []

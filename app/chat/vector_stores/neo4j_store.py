# app/chat/vector_stores/neo4j_store.py
"""
Neo4j vector store implementation for LangGraph architecture.
Provides graph-based storage and retrieval for technical documents.
"""

import os
import logging
import uuid
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

import numpy as np
from neo4j import AsyncGraphDatabase, AsyncDriver
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.chat.types import (
    ContentElement, 
    ContentMetadata, 
    ContentType,
    ConceptNetwork,
    Concept,
    ConceptRelationship
)

logger = logging.getLogger(__name__)

class VectorStoreMetrics(BaseModel):
    """Metrics for Neo4j vector store operations."""
    total_queries: int = 0
    total_docs: int = 0
    total_errors: int = 0
    avg_query_time: float = 0.0
    total_query_time: float = 0.0
    
    def record_query_time(self, time_seconds: float) -> None:
        """Record query time and update average."""
        self.total_query_time += time_seconds
        self.total_queries += 1
        self.avg_query_time = self.total_query_time / self.total_queries
    
    def record_retrieval(self) -> None:
        """Record document retrieval."""
        self.total_queries += 1
    
    def record_error(self, error: str) -> None:
        """Record retrieval error."""
        self.total_errors += 1


class Neo4jVectorStore:
    """
    Neo4j vector store for document storage and retrieval with graph capabilities.
    """
    
    def __init__(
        self,
        url: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        embedding_dimension: int = 1536,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize Neo4j vector store.
        
        Args:
            url: Neo4j connection URL
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            embedding_dimension: Embedding dimension
            embedding_model: Embedding model name
        """
        self.url = url
        self.username = username
        self.password = password
        self.database = database
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model
        self.driver = None
        self.initialized = False
        self.metrics = VectorStoreMetrics()
        self.embeddings = None
        
        # Initialize connection
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Neo4j driver and database."""
        try:
            # Create Neo4j driver
            self.driver = AsyncGraphDatabase.driver(
                self.url,
                auth=(self.username, self.password)
            )
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Set initialization flag - we'll mark as true even without database setup
            # since connection is established
            self.initialized = True
            
            # Skip immediate database setup - will happen on first use
            logger.info(f"Neo4j vector store connection initialized at {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j vector store: {str(e)}")
            self.initialized = False

    async def initialize_database(self) -> bool:
        """
        Initialize database schema.
        Call this method before using the vector store in async contexts.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized:
            logger.error("Cannot initialize database - driver not initialized")
            return False
            
        try:
            await self._setup_database()
            logger.info("Neo4j database schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j database schema: {str(e)}")
            return False
    
    async def _setup_database(self) -> None:
        """Set up Neo4j database with necessary indexes and constraints."""
        try:
            async with self.driver.session(database=self.database) as session:
                # Create document node index
                await session.run("""
                    CREATE CONSTRAINT document_id IF NOT EXISTS
                    FOR (d:Document) REQUIRE d.pdf_id IS UNIQUE
                """)
                
                # Create content element index
                await session.run("""
                    CREATE INDEX content_element_id IF NOT EXISTS
                    FOR (e:ContentElement) ON (e.element_id)
                """)
                
                # Create concept index
                await session.run("""
                    CREATE INDEX concept_name IF NOT EXISTS
                    FOR (c:Concept) ON (c.name)
                """)
                
                # Create section index
                await session.run("""
                    CREATE INDEX section_path IF NOT EXISTS
                    FOR (s:Section) ON (s.path)
                """)
                
                # Create vector index if not exists
                await session.run("""
                    CALL db.index.vector.createNodeIndex(
                      'content_vector',
                      'ContentElement',
                      'embedding',
                      $dimension,
                      'cosine'
                    )
                """, {"dimension": self.embedding_dimension})
                
                logger.info("Neo4j database setup complete")
                
        except Exception as e:
            logger.error(f"Database setup error: {str(e)}")
            raise
    
    async def create_document_node(
        self,
        pdf_id: str,
        title: str = "Untitled Document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a document node in Neo4j.
        
        Args:
            pdf_id: Document ID
            title: Document title
            metadata: Additional metadata
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return
        
        try:
            async with self.driver.session(database=self.database) as session:
                # Create document node 
                await session.run("""
                    MERGE (d:Document {pdf_id: $pdf_id})
                    ON CREATE SET
                      d.title = $title,
                      d.created_at = datetime(),
                      d.metadata = $metadata
                    ON MATCH SET
                      d.title = $title,
                      d.updated_at = datetime(),
                      d.metadata = $metadata
                """, {
                    "pdf_id": pdf_id,
                    "title": title,
                    "metadata": metadata or {}
                })
                
                logger.debug(f"Created document node for {pdf_id}")
                
        except Exception as e:
            logger.error(f"Error creating document node: {str(e)}")
            raise
    
    async def add_content_element(
        self,
        element: ContentElement,
        pdf_id: str
    ) -> None:
        """
        Add a content element to Neo4j.
        
        Args:
            element: Content element
            pdf_id: Document ID
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return
        
        try:
            # Generate embedding for content
            if not element.content:
                logger.warning(f"Empty content for element {element.element_id}")
                return
            
            embedding = self.embeddings.embed_query(element.content)
            
            async with self.driver.session(database=self.database) as session:
                # Create content element node
                await session.run("""
                    MATCH (d:Document {pdf_id: $pdf_id})
                    MERGE (e:ContentElement {element_id: $element_id})
                    ON CREATE SET
                      e.content = $content,
                      e.content_type = $content_type,
                      e.page_number = $page_number,
                      e.section = $section,
                      e.embedding = $embedding,
                      e.created_at = datetime(),
                      e.metadata = $metadata
                    ON MATCH SET
                      e.content = $content,
                      e.content_type = $content_type,
                      e.page_number = $page_number,
                      e.section = $section,
                      e.embedding = $embedding,
                      e.updated_at = datetime(),
                      e.metadata = $metadata
                    MERGE (d)-[:CONTAINS]->(e)
                """, {
                    "pdf_id": pdf_id,
                    "element_id": element.element_id,
                    "content": element.content,
                    "content_type": element.content_type.value if hasattr(element.content_type, 'value') else str(element.content_type),
                    "page_number": element.metadata.page_number if hasattr(element.metadata, 'page_number') else 0,
                    "section": element.metadata.section if hasattr(element.metadata, 'section') else "",
                    "embedding": embedding,
                    "metadata": element.metadata.dict() if hasattr(element.metadata, 'dict') else {}
                })
                
                # If section headers exist, create section hierarchy
                if hasattr(element.metadata, 'section_headers') and element.metadata.section_headers:
                    # Create section path string
                    section_path = " > ".join(element.metadata.section_headers)
                    
                    # Create section node and relationship
                    await session.run("""
                        MATCH (e:ContentElement {element_id: $element_id})
                        MERGE (s:Section {path: $path})
                        ON CREATE SET
                          s.level = $level,
                          s.title = $title,
                          s.pdf_id = $pdf_id
                        MERGE (s)-[:CONTAINS]->(e)
                    """, {
                        "element_id": element.element_id,
                        "path": section_path,
                        "level": len(element.metadata.section_headers),
                        "title": element.metadata.section_headers[-1] if element.metadata.section_headers else "Untitled Section",
                        "pdf_id": pdf_id
                    })
                
                logger.debug(f"Added content element {element.element_id}")
                
        except Exception as e:
            logger.error(f"Error adding content element: {str(e)}")
            raise
    
    async def add_concept(
        self,
        concept_name: str,
        pdf_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a concept to Neo4j.
        
        Args:
            concept_name: Concept name
            pdf_id: Document ID
            metadata: Additional metadata
        """
        if not self.initialized or not concept_name:
            return
        
        try:
            async with self.driver.session(database=self.database) as session:
                # Create concept node and relationship to document
                await session.run("""
                    MATCH (d:Document {pdf_id: $pdf_id})
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET
                      c.created_at = datetime(),
                      c.metadata = $metadata
                    ON MATCH SET
                      c.updated_at = datetime(),
                      c.metadata = CASE WHEN c.metadata IS NULL THEN $metadata
                                       ELSE c.metadata END
                    MERGE (d)-[:HAS_CONCEPT]->(c)
                """, {
                    "pdf_id": pdf_id,
                    "name": concept_name,
                    "metadata": metadata or {}
                })
                
                logger.debug(f"Added concept {concept_name} for document {pdf_id}")
                
        except Exception as e:
            logger.error(f"Error adding concept: {str(e)}")
    
    async def add_concept_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        pdf_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relationship between concepts.
        
        Args:
            source: Source concept
            target: Target concept
            rel_type: Relationship type
            pdf_id: Document ID
            metadata: Additional metadata
        """
        if not self.initialized or not source or not target:
            return
        
        try:
            # Create dynamic relationship type based on rel_type
            # Clean relationship type for Neo4j (no spaces, special chars)
            clean_rel_type = rel_type.upper().replace(" ", "_").replace("-", "_")
            
            async with self.driver.session(database=self.database) as session:
                # Create relationship between concepts
                await session.run(f"""
                    MATCH (d:Document {{pdf_id: $pdf_id}})
                    MATCH (c1:Concept {{name: $source}})
                    MATCH (c2:Concept {{name: $target}})
                    MERGE (c1)-[r:{clean_rel_type}]->(c2)
                    ON CREATE SET
                      r.created_at = datetime(),
                      r.pdf_id = $pdf_id,
                      r.metadata = $metadata
                    ON MATCH SET
                      r.updated_at = datetime(),
                      r.metadata = CASE WHEN r.metadata IS NULL THEN $metadata
                                       ELSE r.metadata END,
                      r.count = CASE WHEN r.count IS NULL THEN 1 ELSE r.count + 1 END
                """, {
                    "source": source,
                    "target": target,
                    "pdf_id": pdf_id,
                    "metadata": metadata or {}
                })
                
                logger.debug(f"Added concept relationship {source} -> {target}")
                
        except Exception as e:
            logger.error(f"Error adding concept relationship: {str(e)}")
    
    async def add_section_concept_relation(
        self,
        section: str,
        concept: str,
        pdf_id: str
    ) -> None:
        """
        Add a relationship between a section and a concept.
        
        Args:
            section: Section path
            concept: Concept name
            pdf_id: Document ID
        """
        if not self.initialized or not section or not concept:
            return
        
        try:
            async with self.driver.session(database=self.database) as session:
                await session.run("""
                    MATCH (s:Section {path: $section, pdf_id: $pdf_id})
                    MATCH (c:Concept {name: $concept})
                    MERGE (s)-[r:DISCUSSES]->(c)
                    ON CREATE SET r.created_at = datetime()
                """, {
                    "section": section,
                    "concept": concept,
                    "pdf_id": pdf_id
                })
                
                logger.debug(f"Added section-concept relationship: {section} -> {concept}")
                
        except Exception as e:
            logger.error(f"Error adding section-concept relationship: {str(e)}")
    
    async def semantic_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None,
        content_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Query string
            k: Number of results
            pdf_id: Optional document ID filter
            content_types: Optional content type filter
            
        Returns:
            List of matching documents
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build Cypher query
            cypher_query = """
                CALL db.index.vector.queryNodes(
                  'content_vector',
                  $k,
                  $embedding
                ) YIELD node, score
            """
            
            # Add filters for PDF ID and content types
            filters = []
            if pdf_id:
                filters.append("node.pdf_id = $pdf_id")
            if content_types and len(content_types) > 0:
                filters.append("node.content_type IN $content_types")
            
            if filters:
                cypher_query += f" WHERE {' AND '.join(filters)}"
            
            cypher_query += """
                MATCH (d:Document)-[:CONTAINS]->(node)
                RETURN
                  node.element_id AS id,
                  node.content AS content,
                  node.content_type AS content_type,
                  d.pdf_id AS pdf_id,
                  node.page_number AS page_number,
                  node.section AS section,
                  node.metadata AS metadata,
                  score
                ORDER BY score DESC
                LIMIT $limit
            """
            
            params = {
                "embedding": query_embedding,
                "k": k * 2,  # Request more candidates to allow for filtering
                "limit": k,
                "pdf_id": pdf_id,
                "content_types": content_types
            }
            
            async with self.driver.session(database=self.database) as session:
                result = await session.run(cypher_query, params)
                
                documents = []
                async for record in result:
                    # Extract data from record
                    content = record["content"]
                    metadata = {
                        "id": record["id"],
                        "content_type": record["content_type"],
                        "pdf_id": record["pdf_id"],
                        "page_number": record["page_number"],
                        "section": record["section"],
                        "score": record["score"],
                        "node_metadata": record["metadata"]
                    }
                    
                    # Create LangChain document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            # Record metrics
            query_time = time.time() - start_time
            self.metrics.record_query_time(query_time)
            
            logger.info(f"Semantic search completed in {query_time:.2f}s, found {len(documents)} results")
            
            return documents
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            self.metrics.record_error(str(e))
            return []
    
    async def keyword_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None
    ) -> List[Document]:
        """
        Perform keyword-based search.
        
        Args:
            query: Query string
            k: Number of results
            pdf_id: Optional document ID filter
            
        Returns:
            List of matching documents
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Extract keywords from query
            from app.chat.utils.extraction import extract_technical_terms
            keywords = extract_technical_terms(query)
            
            if not keywords:
                # If no keywords extracted, use words from the query
                keywords = [w for w in query.split() if len(w) > 3]
            
            # Build fulltext search pattern
            search_terms = " OR ".join(keywords)
            
            async with self.driver.session(database=self.database) as session:
                # Build Cypher query for keyword search
                cypher_query = """
                    CALL db.index.fulltext.queryNodes(
                      'content_fulltext',
                      $search_terms
                    ) YIELD node, score
                """
                
                if pdf_id:
                    cypher_query += " WHERE node.pdf_id = $pdf_id"
                
                cypher_query += """
                    MATCH (d:Document)-[:CONTAINS]->(node)
                    RETURN
                      node.element_id AS id,
                      node.content AS content,
                      node.content_type AS content_type,
                      d.pdf_id AS pdf_id,
                      node.page_number AS page_number,
                      node.section AS section,
                      node.metadata AS metadata,
                      score
                    ORDER BY score DESC
                    LIMIT $limit
                """
                
                result = await session.run(cypher_query, {
                    "search_terms": search_terms,
                    "pdf_id": pdf_id,
                    "limit": k
                })
                
                documents = []
                async for record in result:
                    # Extract data from record
                    content = record["content"]
                    metadata = {
                        "id": record["id"],
                        "content_type": record["content_type"],
                        "pdf_id": record["pdf_id"],
                        "page_number": record["page_number"],
                        "section": record["section"],
                        "score": record["score"],
                        "node_metadata": record["metadata"]
                    }
                    
                    # Create LangChain document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            # Record metrics
            query_time = time.time() - start_time
            self.metrics.record_query_time(query_time)
            
            logger.info(f"Keyword search completed in {query_time:.2f}s, found {len(documents)} results")
            
            return documents
            
        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
            self.metrics.record_error(str(e))
            return []
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None,
        content_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Query string
            k: Number of results
            pdf_id: Optional document ID filter
            content_types: Optional content type filter
            
        Returns:
            List of matching documents
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Perform both semantic and keyword search
            semantic_results = await self.semantic_search(
                query=query,
                k=k,
                pdf_id=pdf_id,
                content_types=content_types
            )
            
            keyword_results = await self.keyword_search(
                query=query,
                k=k,
                pdf_id=pdf_id
            )
            
            # Combine results with scoring
            combined_results = {}
            
            # Add semantic results with their scores
            for doc in semantic_results:
                doc_id = doc.metadata["id"]
                combined_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": doc.metadata["score"],
                    "keyword_score": 0.0
                }
            
            # Add keyword results or update existing ones
            for doc in keyword_results:
                doc_id = doc.metadata["id"]
                if doc_id in combined_results:
                    combined_results[doc_id]["keyword_score"] = doc.metadata["score"]
                else:
                    combined_results[doc_id] = {
                        "doc": doc,
                        "semantic_score": 0.0,
                        "keyword_score": doc.metadata["score"]
                    }
            
            # Calculate combined score (weighted average)
            for doc_id, data in combined_results.items():
                # Give slightly more weight to semantic search (60/40)
                combined_score = (data["semantic_score"] * 0.6) + (data["keyword_score"] * 0.4)
                data["combined_score"] = combined_score
                
                # Update document metadata with combined score
                data["doc"].metadata["score"] = combined_score
            
            # Sort by combined score and limit to k results
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )
            
            final_results = [data["doc"] for data in sorted_results[:k]]
            
            # Record metrics
            query_time = time.time() - start_time
            self.metrics.record_query_time(query_time)
            
            logger.info(f"Hybrid search completed in {query_time:.2f}s, found {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search error: {str(e)}")
            self.metrics.record_error(str(e))
            return []
    
    async def concept_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None
    ) -> List[Document]:
        """
        Perform concept-based search using the graph structure.
        
        Args:
            query: Query string
            k: Number of results
            pdf_id: Optional document ID filter
            
        Returns:
            List of matching documents
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Extract concepts from query
            from app.chat.utils.extraction import extract_technical_terms
            concepts = extract_technical_terms(query)
            
            if not concepts:
                # If no concepts extracted, fall back to hybrid search
                logger.info("No concepts extracted from query, falling back to hybrid search")
                return await self.hybrid_search(query, k, pdf_id)
            
            # Use the graph structure to find content elements related to concepts
            async with self.driver.session(database=self.database) as session:
                cypher_query = """
                    MATCH (c:Concept)
                    WHERE c.name IN $concepts
                    WITH c
                    MATCH path = (c)<-[:HAS_CONCEPT|DISCUSSES*1..3]-(element:ContentElement)
                """
                
                if pdf_id:
                    cypher_query += """
                        MATCH (d:Document {pdf_id: $pdf_id})-[:CONTAINS]->(element)
                    """
                
                cypher_query += """
                    WITH
                      element,
                      c.name AS concept,
                      length(path) AS distance
                    WITH
                      element,
                      collect(concept) AS matched_concepts,
                      min(distance) AS min_distance
                    RETURN
                      element.element_id AS id,
                      element.content AS content,
                      element.content_type AS content_type,
                      element.pdf_id AS pdf_id,
                      element.page_number AS page_number,
                      element.section AS section,
                      element.metadata AS metadata,
                      matched_concepts,
                      min_distance,
                      size(matched_concepts) AS concept_count
                    ORDER BY concept_count DESC, min_distance ASC
                    LIMIT $limit
                """
                
                result = await session.run(cypher_query, {
                    "concepts": concepts,
                    "pdf_id": pdf_id,
                    "limit": k
                })
                
                documents = []
                async for record in result:
                    # Extract data from record
                    content = record["content"]
                    metadata = {
                        "id": record["id"],
                        "content_type": record["content_type"],
                        "pdf_id": record["pdf_id"],
                        "page_number": record["page_number"],
                        "section": record["section"],
                        "matched_concepts": record["matched_concepts"],
                        "min_distance": record["min_distance"],
                        "concept_count": record["concept_count"],
                        "score": 1.0 / (record["min_distance"] + 1) * record["concept_count"],
                        "node_metadata": record["metadata"]
                    }
                    
                    # Create LangChain document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            # Record metrics
            query_time = time.time() - start_time
            self.metrics.record_query_time(query_time)
            
            logger.info(f"Concept search completed in {query_time:.2f}s, found {len(documents)} results")
            
            return documents
            
        except Exception as e:
            logger.error(f"Concept search error: {str(e)}")
            self.metrics.record_error(str(e))
            return []
    
    async def combined_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None
    ) -> List[Document]:
        """
        Perform combined search using all available strategies.
        This is the most comprehensive search method.
        
        Args:
            query: Query string
            k: Number of results
            pdf_id: Optional document ID filter
            
        Returns:
            List of matching documents
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Determine if query has concepts
            from app.chat.utils.extraction import extract_technical_terms
            concepts = extract_technical_terms(query)
            has_concepts = len(concepts) > 0
            
            # Perform searches with appropriate strategies
            semantic_results = await self.semantic_search(query, k, pdf_id)
            
            concept_results = []
            if has_concepts:
                concept_results = await self.concept_search(query, k, pdf_id)
            
            keyword_results = await self.keyword_search(query, k, pdf_id)
            
            # Combine results with scoring
            combined_results = {}
            
            # Add semantic results
            for doc in semantic_results:
                doc_id = doc.metadata["id"]
                combined_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": doc.metadata["score"],
                    "keyword_score": 0.0,
                    "concept_score": 0.0
                }
            
            # Add keyword results
            for doc in keyword_results:
                doc_id = doc.metadata["id"]
                if doc_id in combined_results:
                    combined_results[doc_id]["keyword_score"] = doc.metadata["score"]
                else:
                    combined_results[doc_id] = {
                        "doc": doc,
                        "semantic_score": 0.0,
                        "keyword_score": doc.metadata["score"],
                        "concept_score": 0.0
                    }
            
            # Add concept results
            for doc in concept_results:
                doc_id = doc.metadata["id"]
                if doc_id in combined_results:
                    combined_results[doc_id]["concept_score"] = doc.metadata["score"]
                else:
                    combined_results[doc_id] = {
                        "doc": doc,
                        "semantic_score": 0.0,
                        "keyword_score": 0.0,
                        "concept_score": doc.metadata["score"]
                    }
            
            # Calculate combined score with weighted average
            for doc_id, data in combined_results.items():
                # Weights: semantic (40%), concept (40%), keyword (20%)
                weights = [0.4, 0.2, 0.4] if has_concepts else [0.6, 0.4, 0.0]
                combined_score = (
                    (data["semantic_score"] * weights[0]) + 
                    (data["keyword_score"] * weights[1]) + 
                    (data["concept_score"] * weights[2])
                )
                data["combined_score"] = combined_score
                
                # Update document metadata with combined score
                data["doc"].metadata["score"] = combined_score
            
            # Sort by combined score and limit to k results
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )
            
            final_results = [data["doc"] for data in sorted_results[:k]]
            
            # Record metrics
            query_time = time.time() - start_time
            self.metrics.record_query_time(query_time)
            
            logger.info(f"Combined search completed in {query_time:.2f}s, found {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Combined search error: {str(e)}")
            self.metrics.record_error(str(e))
            return []
    
    async def find_common_concepts(self, pdf_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Find concepts that appear across multiple documents.
        
        Args:
            pdf_ids: List of document IDs
            
        Returns:
            List of common concepts with document references
        """
        if not self.initialized or not pdf_ids or len(pdf_ids) < 2:
            return []
        
        try:
            async with self.driver.session(database=self.database) as session:
                cypher_query = """
                    MATCH (d:Document)-[:HAS_CONCEPT]->(c:Concept)
                    WHERE d.pdf_id IN $pdf_ids
                    WITH
                      c,
                      collect(d.pdf_id) AS documents,
                      count(d) AS doc_count
                    WHERE doc_count > 1
                    OPTIONAL MATCH (c)-[r]->(c2:Concept)
                    WITH
                      c.name AS name,
                      documents,
                      doc_count,
                      count(r) AS relationship_count
                    RETURN
                      name,
                      documents,
                      doc_count,
                      relationship_count,
                      1.0 * doc_count / size($pdf_ids) AS relevance
                    ORDER BY relevance DESC, relationship_count DESC
                    LIMIT 20
                """
                
                result = await session.run(cypher_query, {"pdf_ids": pdf_ids})
                
                common_concepts = []
                async for record in result:
                    common_concepts.append({
                        "name": record["name"],
                        "documents": record["documents"],
                        "document_count": record["doc_count"],
                        "relationship_count": record["relationship_count"],
                        "relevance": record["relevance"]
                    })
                
                logger.info(f"Found {len(common_concepts)} common concepts across {len(pdf_ids)} documents")
                
                return common_concepts
                
        except Exception as e:
            logger.error(f"Error finding common concepts: {str(e)}")
            return []
    
    async def find_concept_paths(
        self,
        start_concept: str,
        pdf_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find paths between concepts across documents.
        
        Args:
            start_concept: Starting concept
            pdf_ids: List of document IDs
            
        Returns:
            List of concept paths
        """
        if not self.initialized or not start_concept or not pdf_ids:
            return []
        
        try:
            async with self.driver.session(database=self.database) as session:
                cypher_query = """
                    MATCH (c1:Concept)
                    WHERE c1.name CONTAINS $concept
                    WITH c1 LIMIT 1
                    MATCH
                      path = (c1)-[r:EXTENDS|PART_OF|USES|IMPLEMENTS|RELATES_TO*1..5]-(c2:Concept)
                    WHERE
                      c1 <> c2 AND
                      ANY(doc IN $pdf_ids WHERE (c1)-[:HAS_CONCEPT]-(:Document {pdf_id: doc})) AND
                      ANY(doc IN $pdf_ids WHERE (c2)-[:HAS_CONCEPT]-(:Document {pdf_id: doc}))
                    WITH
                      nodes(path) AS path_nodes,
                      relationships(path) AS path_rels,
                      length(path) AS path_length
                    RETURN
                      [n IN path_nodes | n.name] AS path,
                      [r IN path_rels | type(r)] AS relationship_types,
                      path_length
                    ORDER BY path_length
                    LIMIT 10
                """
                
                result = await session.run(cypher_query, {
                    "concept": start_concept,
                    "pdf_ids": pdf_ids
                })
                
                paths = []
                async for record in result:
                    paths.append({
                        "path": record["path"],
                        "relationship_types": record["relationship_types"],
                        "length": record["path_length"]
                    })
                
                logger.info(f"Found {len(paths)} concept paths from {start_concept}")
                
                return paths
                
        except Exception as e:
            logger.error(f"Error finding concept paths: {str(e)}")
            return []
    
    async def ingest_processed_content(self, result: Any) -> bool:
        """
        Ingest processed document content.
        
        Args:
            result: ProcessingResult from document processor
            
        Returns:
            Success status
        """
        if not hasattr(result, 'pdf_id') or not result.pdf_id:
            logger.error("Invalid processing result (missing pdf_id)")
            return False
        
        logger.info(f"Ingesting processed content for {result.pdf_id}")
        
        try:
            # Create document node
            title = "Untitled Document"
            if hasattr(result, 'document_summary') and result.document_summary:
                if hasattr(result.document_summary, 'title'):
                    title = result.document_summary.title
            
            await self.create_document_node(
                pdf_id=result.pdf_id,
                title=title,
                metadata={
                    "processed_at": datetime.utcnow().isoformat(),
                    "element_count": len(result.elements) if hasattr(result, 'elements') else 0
                }
            )
            
            # Add content elements
            if hasattr(result, 'elements') and result.elements:
                for element in result.elements:
                    await self.add_content_element(element, result.pdf_id)
            
            # Add concepts and relationships
            if hasattr(result, 'concept_network') and result.concept_network:
                # Add concepts
                for concept in result.concept_network.concepts:
                    await self.add_concept(
                        concept_name=concept.name,
                        pdf_id=result.pdf_id,
                        metadata={
                            "importance": concept.importance_score,
                            "is_primary": concept.is_primary,
                            "category": concept.category
                        }
                    )
                
                # Add relationships
                for relationship in result.concept_network.relationships:
                    await self.add_concept_relationship(
                        source=relationship.source,
                        target=relationship.target,
                        rel_type=relationship.type.value if hasattr(relationship.type, 'value') else str(relationship.type),
                        pdf_id=result.pdf_id,
                        metadata={
                            "weight": relationship.weight,
                            "context": relationship.context
                        }
                    )
                
                # Add section-concept relationships
                if hasattr(result.concept_network, 'section_concepts'):
                    for section, concepts in result.concept_network.section_concepts.items():
                        for concept in concepts:
                            await self.add_section_concept_relation(
                                section=section,
                                concept=concept,
                                pdf_id=result.pdf_id
                            )
            
            logger.info(f"Successfully ingested content for {result.pdf_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting processed content: {str(e)}", exc_info=True)
            return False
    
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Legacy compatibility method for similarity search (LangChain compatible).
        
        Args:
            query: Query string
            k: Number of results
            filter: Filter criteria
            
        Returns:
            List of matching documents
        """
        # Extract PDF ID from filter if present
        pdf_id = None
        content_types = None
        
        if filter:
            if "pdf_id" in filter:
                pdf_id = filter["pdf_id"]
            if "content_types" in filter:
                content_types = filter["content_types"]
        
        # Use semantic search
        return await self.semantic_search(
            query=query,
            k=k,
            pdf_id=pdf_id,
            content_types=content_types
        )
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Legacy compatibility method for adding texts (LangChain compatible).
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs
            
        Returns:
            List of IDs for added texts
        """
        import asyncio
        result_ids = []
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def add_all_texts():
            nonlocal result_ids
            
            for i, text in enumerate(texts):
                # Get metadata if available
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                
                # Get ID if available, otherwise generate one
                element_id = ids[i] if ids and i < len(ids) else f"txt_{uuid.uuid4().hex}"
                result_ids.append(element_id)
                
                # Get PDF ID from metadata
                pdf_id = metadata.get("pdf_id", "unknown")
                
                # Create content element
                element = ContentElement(
                    element_id=element_id,
                    content=text,
                    content_type=metadata.get("content_type", "text"),
                    pdf_id=pdf_id,
                    metadata=ContentMetadata(
                        pdf_id=pdf_id,
                        page_number=metadata.get("page_number", 0),
                        section=metadata.get("section", ""),
                        content_type=metadata.get("content_type", "text")
                    )
                )
                
                # Add to Neo4j
                await self.add_content_element(element, pdf_id)
        
        loop.run_until_complete(add_all_texts())
        
        return result_ids
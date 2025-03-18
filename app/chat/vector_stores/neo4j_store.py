"""
Neo4j vector store implementation for LangGraph architecture.
Provides graph-based storage and retrieval for technical documents.
"""

import os
import logging
import json
import uuid
import time
import asyncio
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
    Neo4j vector store for document storage and retrieval.
    """

    def __init__(
        self,
        url: str = "neo4j://localhost:7687",  # Use neo4j:// for Neo4j 5.x
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        embedding_dimension: int = 1536,
        index_name: str = "content_vector",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.url = url
        self.username = username
        self.password = password
        self.database = database
        self.embedding_dimension = embedding_dimension
        self.index_name = index_name
        self.embedding_model = embedding_model

        # Feature flags for database and index availability.
        self.initialized = False
        self.has_vector_index = False
        self.has_fulltext_index = False

        # Create driver instance.
        self.driver = AsyncGraphDatabase.driver(
            self.url,
            auth=(self.username, self.password),
            database=self.database
        )
        logger.info(f"Initialized Neo4j Vector Store with {self.embedding_dimension} dimensions")

        # Initialize embeddings (using our LangChain-compatible client).
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize metrics.
        self.metrics = VectorStoreMetrics()

    async def initialize_database(self) -> bool:
        if self.initialized:
            return True
        try:
            async with self.driver.session(database=self.database) as session:
                # Test connection.
                result = await session.run("RETURN 1 AS test")
                record = await result.single()
                if not record or record.get("test") != 1:
                    logger.error("Database connection test failed")
                    return False

                # Check Neo4j version and set index flags accordingly.
                try:
                    version_result = await session.run(
                        "CALL dbms.components() YIELD versions RETURN versions[0] as version"
                    )
                    version_record = await version_result.single()
                    neo4j_version = version_record.get("version", "unknown") if version_record else "unknown"
                    logger.info(f"Connected to Neo4j version: {neo4j_version}")

                    if neo4j_version.startswith("5."):
                        indexes_result = await session.run("SHOW INDEXES")
                        try:
                            records = await indexes_result.values()
                        except TypeError:
                            records = []
                            while await indexes_result.fetch(1):
                                records.append(indexes_result.current())
                    else:
                        indexes_result = await session.run("CALL db.indexes()")
                        records = []
                        while await indexes_result.fetch(1):
                            records.append(indexes_result.current())

                    self.has_vector_index = any("content_vector" in str(record) for record in records)
                    self.has_fulltext_index = any("content_fulltext" in str(record) for record in records)
                    logger.info(f"Index check: vector={self.has_vector_index}, fulltext={self.has_fulltext_index}")
                except Exception as e:
                    logger.warning(f"Could not determine Neo4j version: {str(e)}")

                self.initialized = True
                return True
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            return False

    async def ensure_connection(self, max_retries=3, retry_delay=2.0) -> bool:
        """
        Ensure Neo4j connection is active with retry logic.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if connection successful, False otherwise
        """
        if self.initialized and self.driver:
            try:
                async with self.driver.session(database=self.database) as session:
                    result = await session.run("RETURN 1 as test")
                    record = await result.single()
                    if record and record["test"] == 1:
                        logger.debug("Neo4j connection is already active")
                        return True
            except Exception as e:
                logger.warning(f"Existing Neo4j connection failed: {str(e)}")
                self.initialized = False
                self.driver = None

        retries = 0
        last_error = None
        while retries < max_retries:
            try:
                logger.info(f"Attempting Neo4j connection (attempt {retries+1}/{max_retries})")
                self.driver = AsyncGraphDatabase.driver(
                    self.url,
                    auth=(self.username, self.password),
                    max_connection_lifetime=3600
                )
                self.embeddings = OpenAIEmbeddings(
                    model=self.embedding_model,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                async with self.driver.session(database=self.database) as session:
                    result = await session.run("RETURN 1 as test")
                    record = await result.single()
                    if record and record["test"] == 1:
                        self.initialized = True
                        logger.info(f"Neo4j connection established successfully on attempt {retries+1}")
                        await self._setup_database()
                        return True
            except Exception as e:
                last_error = e
                logger.warning(f"Neo4j connection attempt {retries+1} failed: {str(e)}")
                if self.driver:
                    await self.driver.close()
                    self.driver = None
                self.initialized = False
            retries += 1
            if retries < max_retries:
                delay = retry_delay * (1.5 ** retries)
                logger.info(f"Waiting {delay:.1f}s before next connection attempt")
                await asyncio.sleep(delay)
        if last_error:
            logger.error(f"All Neo4j connection attempts failed: {str(last_error)}")
        return False

    async def _setup_database(self) -> None:
        """Set up Neo4j database with necessary indexes and constraints."""
        try:
            async with self.driver.session(database=self.database) as session:
                # Create document uniqueness constraint.
                await session.run("""
                    CREATE CONSTRAINT document_id IF NOT EXISTS
                    FOR (d:Document) REQUIRE d.pdf_id IS UNIQUE
                """)
                # Create content element index.
                await session.run("""
                    CREATE INDEX content_element_id IF NOT EXISTS
                    FOR (e:ContentElement) ON (e.element_id)
                """)
                # Create concept index.
                await session.run("""
                    CREATE INDEX concept_name IF NOT EXISTS
                    FOR (c:Concept) ON (c.name)
                """)
                # Create section index.
                await session.run("""
                    CREATE INDEX section_path IF NOT EXISTS
                    FOR (s:Section) ON (s.path)
                """)
                # Create vector index if not exists.
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
        Create a document node in Neo4j with robust handling of complex metadata.
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return
        try:
            clean_metadata = {}
            if metadata:
                for key, value in metadata.items():
                    if key == "document_summary":
                        if isinstance(value, dict):
                            clean_metadata["document_summary_json"] = json.dumps(value)
                            if "title" in value:
                                clean_metadata["doc_title"] = str(value["title"])
                        elif isinstance(value, str):
                            clean_metadata["document_summary_json"] = value
                        else:
                            clean_metadata["document_summary_text"] = str(value)
                    elif isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = value
                    elif isinstance(value, list):
                        if all(isinstance(item, (str, int, float, bool)) for item in value):
                            clean_metadata[key] = value
                        else:
                            clean_metadata[f"{key}_json"] = json.dumps(value)
                    elif isinstance(value, dict):
                        clean_metadata[f"{key}_json"] = json.dumps(value)
                    else:
                        clean_metadata[key] = str(value)
            logger.debug(f"Cleaned metadata for Neo4j: {list(clean_metadata.keys())}")
            async with self.driver.session(database=self.database) as session:
                await session.run("""
                    MERGE (d:Document {pdf_id: $pdf_id})
                    ON CREATE SET
                      d.title = $title,
                      d.created_at = datetime()
                    ON MATCH SET
                      d.title = $title,
                      d.updated_at = datetime()
                """, {"pdf_id": pdf_id, "title": title})
                if clean_metadata:
                    for key, value in clean_metadata.items():
                        try:
                            cypher_query = f"""
                                MATCH (d:Document {{pdf_id: $pdf_id}})
                                SET d.{key} = $value
                            """
                            await session.run(cypher_query, {"pdf_id": pdf_id, "value": value})
                        except Exception as prop_error:
                            logger.warning(f"Failed to set property {key}: {str(prop_error)}")
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
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return
        try:
            if not element.content:
                logger.warning(f"Empty content for element {element.element_id}")
                return
            # Generate embedding for the content.
            embedding = self.embeddings.embed_query(element.content)
            content_type = element.content_type.value if hasattr(element.content_type, 'value') else str(element.content_type)
            page_number = element.metadata.page_number if hasattr(element.metadata, 'page_number') else 0
            section = element.metadata.section if hasattr(element.metadata, 'section') else ""
            async with self.driver.session(database=self.database) as session:
                await session.run("""
                    MATCH (d:Document {pdf_id: $pdf_id})
                    MERGE (e:ContentElement {element_id: $element_id})
                    ON CREATE SET
                      e.content = $content,
                      e.content_type = $content_type,
                      e.page_number = $page_number,
                      e.section = $section,
                      e.embedding = $embedding,
                      e.created_at = datetime()
                    ON MATCH SET
                      e.content = $content,
                      e.content_type = $content_type,
                      e.page_number = $page_number,
                      e.section = $section,
                      e.embedding = $embedding,
                      e.updated_at = datetime()
                    MERGE (d)-[:CONTAINS]->(e)
                """, {
                    "pdf_id": pdf_id,
                    "element_id": element.element_id,
                    "content": element.content,
                    "content_type": content_type,
                    "page_number": page_number,
                    "section": section,
                    "embedding": embedding
                })
                if hasattr(element, 'metadata'):
                    if hasattr(element.metadata, 'technical_terms') and element.metadata.technical_terms:
                        await session.run("""
                            MATCH (e:ContentElement {element_id: $element_id})
                            SET e.technical_terms = $terms
                        """, {"element_id": element.element_id, "terms": element.metadata.technical_terms})
                    if hasattr(element.metadata, 'hierarchy_level'):
                        await session.run("""
                            MATCH (e:ContentElement {element_id: $element_id})
                            SET e.hierarchy_level = $level
                        """, {"element_id": element.element_id, "level": element.metadata.hierarchy_level})
                    if hasattr(element.metadata, 'section_headers') and element.metadata.section_headers:
                        await session.run("""
                            MATCH (e:ContentElement {element_id: $element_id})
                            SET e.section_headers = $headers
                        """, {"element_id": element.element_id, "headers": element.metadata.section_headers})
                    if hasattr(element.metadata, 'surrounding_context') and element.metadata.surrounding_context:
                        await session.run("""
                            MATCH (e:ContentElement {element_id: $element_id})
                            SET e.surrounding_context = $context
                        """, {"element_id": element.element_id, "context": element.metadata.surrounding_context})
                    if hasattr(element.metadata, 'image_metadata') and element.metadata.image_metadata:
                        if hasattr(element.metadata.image_metadata, 'dict'):
                            image_data = element.metadata.image_metadata.dict()
                        else:
                            image_data = {"data": str(element.metadata.image_metadata)}
                        await session.run("""
                            MATCH (e:ContentElement {element_id: $element_id})
                            SET e.image_data = $data
                        """, {"element_id": element.element_id, "data": json.dumps(image_data)})
                    if hasattr(element.metadata, 'table_data') and element.metadata.table_data:
                        if hasattr(element.metadata.table_data, 'dict'):
                            table_data = element.metadata.table_data.dict()
                        else:
                            table_data = {"data": str(element.metadata.table_data)}
                        await session.run("""
                            MATCH (e:ContentElement {element_id: $element_id})
                            SET e.table_data = $data
                        """, {"element_id": element.element_id, "data": json.dumps(table_data)})
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
        Add a concept to Neo4j with proper handling of complex metadata.
        """
        if not self.initialized or not concept_name:
            return
        try:
            async with self.driver.session(database=self.database) as session:
                await session.run("""
                    MATCH (d:Document {pdf_id: $pdf_id})
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET c.created_at = datetime()
                    ON MATCH SET c.updated_at = datetime()
                    MERGE (d)-[:HAS_CONCEPT]->(c)
                """, {"pdf_id": pdf_id, "name": concept_name})
                if metadata:
                    if "importance" in metadata and isinstance(metadata["importance"], (int, float)):
                        await session.run("""
                            MATCH (c:Concept {name: $name})
                            SET c.importance_score = $importance
                        """, {"name": concept_name, "importance": metadata["importance"]})
                    if "is_primary" in metadata and isinstance(metadata["is_primary"], bool):
                        await session.run("""
                            MATCH (c:Concept {name: $name})
                            SET c.is_primary = $is_primary
                        """, {"name": concept_name, "is_primary": metadata["is_primary"]})
                    if "category" in metadata and metadata["category"] is not None:
                        await session.run("""
                            MATCH (c:Concept {name: $name})
                            SET c.category = $category
                        """, {"name": concept_name, "category": str(metadata["category"])})
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
        Add a relationship between concepts with proper handling of complex metadata.
        """
        if not self.initialized or not source or not target:
            return
        try:
            clean_rel_type = rel_type.upper().replace(" ", "_").replace("-", "_")
            async with self.driver.session(database=self.database) as session:
                await session.run(f"""
                    MATCH (d:Document {{pdf_id: $pdf_id}})
                    MATCH (c1:Concept {{name: $source}})
                    MATCH (c2:Concept {{name: $target}})
                    MERGE (c1)-[r:{clean_rel_type}]->(c2)
                    ON CREATE SET r.created_at = datetime(), r.pdf_id = $pdf_id
                    ON MATCH SET r.updated_at = datetime(),
                      r.count = CASE WHEN r.count IS NULL THEN 1 ELSE r.count + 1 END
                """, {"source": source, "target": target, "pdf_id": pdf_id})
                if metadata:
                    if "weight" in metadata and isinstance(metadata["weight"], (int, float)):
                        await session.run(f"""
                            MATCH (c1:Concept {{name: $source}})-[r:{clean_rel_type}]->(c2:Concept {{name: $target}})
                            WHERE r.pdf_id = $pdf_id
                            SET r.weight = $weight
                        """, {"source": source, "target": target, "pdf_id": pdf_id, "weight": metadata["weight"]})
                    if "context" in metadata and metadata["context"]:
                        await session.run(f"""
                            MATCH (c1:Concept {{name: $source}})-[r:{clean_rel_type}]->(c2:Concept {{name: $target}})
                            WHERE r.pdf_id = $pdf_id
                            SET r.context = $context
                        """, {"source": source, "target": target, "pdf_id": pdf_id, "context": str(metadata["context"])})
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
                """, {"section": section, "concept": concept, "pdf_id": pdf_id})
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
        """
        if not self.initialized:
            await self.initialize_database()
        if not self.has_vector_index:
            logger.warning("Vector index not available, falling back to keyword search")
            return await self.keyword_search(query, k, pdf_id, content_types)
        start_time = time.time()
        try:
            query_embedding = await self.generate_embedding(query)
            cypher_query = f"""
                CALL db.index.vector.queryNodes(
                  '{self.index_name}',
                  $k,
                  $embedding
                ) YIELD node, score
            """
            filters = []
            if pdf_id:
                filters.append("node.pdf_id = $pdf_id")
            if content_types and len(content_types) > 0:
                filters.append("node.content_type IN $content_types")
            if filters:
                cypher_query += " WHERE " + " AND ".join(filters)
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
                "k": k * 2,
                "limit": k
            }
            if pdf_id:
                params["pdf_id"] = pdf_id
            if content_types and len(content_types) > 0:
                params["content_types"] = content_types
            async with self.driver.session(database=self.database) as session:
                result = await session.run(cypher_query, params)
                documents = []
                try:
                    records = await result.values()
                except TypeError:
                    records = []
                    while await result.fetch(1):
                        records.append(result.current())
                for record in records:
                    if len(record) < 8:
                        logger.warning(f"Unexpected record format: {record}")
                        continue
                    metadata = {
                        "id": record[0],
                        "content_type": record[2] or "text",
                        "pdf_id": record[3] or "",
                        "page_number": record[4] or 0,
                        "section": record[5] or "",
                        "score": record[7] or 0.0,
                        "node_metadata": record[6]
                    }
                    doc = Document(page_content=record[1] or "", metadata=metadata)
                    documents.append(doc)
            query_time = time.time() - start_time
            self.metrics.record_query_time(query_time)
            logger.info(f"Semantic search completed in {query_time:.2f}s, found {len(documents)} results")
            return documents
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            self.metrics.record_error(str(e))
            return []

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for text using the OpenAI API.
        """
        try:
            # If the underlying embed_query is synchronous, this wrapper can be made async.
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * self.embedding_dimension

    async def keyword_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None,
        content_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Perform keyword search using standard index matching as a fallback.
        """
        if not query:
            return []
        try:
            keywords = [kw.strip().lower() for kw in query.split() if len(kw.strip()) > 2]
            if not keywords:
                return []
            search_terms = [f".*{keyword}.*" for keyword in keywords]
            regex_pattern = "|".join(search_terms)
            async with self.driver.session() as session:
                cypher_query = """
                    MATCH (c:ContentElement)
                    WHERE c.content =~ $regex_pattern
                """
                if pdf_id:
                    cypher_query += " AND c.pdf_id = $pdf_id"
                if content_types and len(content_types) > 0:
                    cypher_query += " AND c.content_type IN $content_types"
                cypher_query += """
                    MATCH (d:Document)-[:CONTAINS]->(c)
                    RETURN
                        c.element_id AS id,
                        c.content AS content,
                        c.content_type AS content_type,
                        d.pdf_id AS pdf_id,
                        c.page_number AS page_number,
                        c.section AS section,
                        toFloat(1.0) AS score
                    LIMIT $limit
                """
                params = {"regex_pattern": f"(?i)({regex_pattern})", "limit": k}
                if pdf_id:
                    params["pdf_id"] = pdf_id
                if content_types and len(content_types) > 0:
                    params["content_types"] = content_types
                result = await session.run(cypher_query, params)
                records = await result.values()
                documents = []
                for record in records:
                    metadata = {
                        "id": record[0],
                        "content_type": record[2] or "text",
                        "pdf_id": record[3] or "",
                        "page_number": record[4] or 0,
                        "section": record[5] or "",
                        "score": record[6] or 0.0
                    }
                    documents.append(Document(page_content=record[1] or "", metadata=metadata))
                return documents
        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
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
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        start_time = time.time()
        try:
            semantic_results = await self.semantic_search(query, k, pdf_id, content_types)
            keyword_results = await self.keyword_search(query, k, pdf_id)
            combined_results = {}
            for doc in semantic_results:
                doc_id = doc.metadata["id"]
                combined_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": doc.metadata["score"],
                    "keyword_score": 0.0
                }
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
            for doc_id, data in combined_results.items():
                combined_score = (data["semantic_score"] * 0.6) + (data["keyword_score"] * 0.4)
                data["combined_score"] = combined_score
                data["doc"].metadata["score"] = combined_score
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )
            final_results = [data["doc"] for data in sorted_results[:k]]
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
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        start_time = time.time()
        try:
            from app.chat.utils.extraction import extract_technical_terms
            concepts = extract_technical_terms(query)
            if not concepts:
                logger.info("No concepts extracted from query, falling back to hybrid search")
                return await self.hybrid_search(query, k, pdf_id)
            async with self.driver.session(database=self.database) as session:
                cypher_query = """
                    MATCH (c:Concept)
                    WHERE c.name IN $concepts
                    WITH c
                    MATCH path = (c)<-[:HAS_CONCEPT|DISCUSSES*1..3]-(element:ContentElement)
                """
                if pdf_id:
                    cypher_query += " MATCH (d:Document {pdf_id: $pdf_id})-[:CONTAINS]->(element)"
                cypher_query += """
                    WITH element, c.name AS concept, length(path) AS distance
                    WITH element, collect(concept) AS matched_concepts, min(distance) AS min_distance
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
                result = await session.run(cypher_query, {"concepts": concepts, "pdf_id": pdf_id, "limit": k})
                documents = []
                async for record in result:
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
                    doc = Document(page_content=record["content"], metadata=metadata)
                    documents.append(doc)
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
        """
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []
        start_time = time.time()
        try:
            from app.chat.utils.extraction import extract_technical_terms
            concepts = extract_technical_terms(query)
            has_concepts = len(concepts) > 0
            semantic_results = await self.semantic_search(query, k, pdf_id)
            concept_results = await self.concept_search(query, k, pdf_id) if has_concepts else []
            keyword_results = await self.keyword_search(query, k, pdf_id)
            combined_results = {}
            for doc in semantic_results:
                doc_id = doc.metadata["id"]
                combined_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": doc.metadata["score"],
                    "keyword_score": 0.0,
                    "concept_score": 0.0
                }
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
            for doc_id, data in combined_results.items():
                weights = [0.4, 0.2, 0.4] if has_concepts else [0.6, 0.4, 0.0]
                combined_score = (data["semantic_score"] * weights[0] +
                                  data["keyword_score"] * weights[1] +
                                  data["concept_score"] * weights[2])
                data["combined_score"] = combined_score
                data["doc"].metadata["score"] = combined_score
            sorted_results = sorted(combined_results.values(), key=lambda x: x["combined_score"], reverse=True)
            final_results = [data["doc"] for data in sorted_results[:k]]
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
        """
        if not self.initialized or not pdf_ids or len(pdf_ids) < 2:
            return []
        try:
            async with self.driver.session(database=self.database) as session:
                cypher_query = """
                    MATCH (d:Document)-[:HAS_CONCEPT]->(c:Concept)
                    WHERE d.pdf_id IN $pdf_ids
                    WITH c, collect(d.pdf_id) AS documents, count(d) AS doc_count
                    WHERE doc_count > 1
                    OPTIONAL MATCH (c)-[r]->(c2:Concept)
                    WITH c.name AS name, documents, doc_count, count(r) AS relationship_count
                    RETURN name, documents, doc_count, relationship_count, 1.0 * doc_count / size($pdf_ids) AS relevance
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
        """
        if not self.initialized or not start_concept or not pdf_ids:
            return []
        try:
            async with self.driver.session(database=self.database) as session:
                cypher_query = """
                    MATCH (c1:Concept)
                    WHERE c1.name CONTAINS $concept
                    WITH c1 LIMIT 1
                    MATCH path = (c1)-[r:EXTENDS|PART_OF|USES|IMPLEMENTS|RELATES_TO*1..5]-(c2:Concept)
                    WHERE c1 <> c2 AND
                      ANY(doc IN $pdf_ids WHERE (c1)-[:HAS_CONCEPT]-(:Document {pdf_id: doc})) AND
                      ANY(doc IN $pdf_ids WHERE (c2)-[:HAS_CONCEPT]-(:Document {pdf_id: doc}))
                    WITH nodes(path) AS path_nodes, relationships(path) AS path_rels, length(path) AS path_length
                    RETURN [n IN path_nodes | n.name] AS path, [r IN path_rels | type(r)] AS relationship_types, path_length
                    ORDER BY path_length
                    LIMIT 10
                """
                result = await session.run(cypher_query, {"concept": start_concept, "pdf_ids": pdf_ids})
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
        Ingest processed document content with enhanced error handling and metadata processing.
        """
        if not hasattr(result, 'pdf_id') or not result.pdf_id:
            logger.error("Invalid processing result (missing pdf_id)")
            return False
        pdf_id = result.pdf_id
        logger.info(f"Ingesting processed content for {pdf_id}")
        try:
            try:
                existing = await self._check_document_exists(pdf_id)
                if existing:
                    logger.info(f"Document {pdf_id} exists, clearing for re-ingestion")
                    await self.delete_document_content(pdf_id)
            except Exception as del_err:
                logger.warning(f"Error checking/clearing existing document: {str(del_err)}")
            title = "Untitled Document"
            doc_summary_dict = None
            if hasattr(result, 'document_summary') and result.document_summary:
                if hasattr(result.document_summary, 'title'):
                    title = result.document_summary.title
                try:
                    if hasattr(result.document_summary, 'dict'):
                        doc_summary_dict = result.document_summary.dict()
                    elif hasattr(result.document_summary, '__dict__'):
                        doc_summary_dict = result.document_summary.__dict__
                    else:
                        if isinstance(result.document_summary, str) and result.document_summary.startswith('{'):
                            try:
                                doc_summary_dict = json.loads(result.document_summary)
                            except json.JSONDecodeError:
                                doc_summary_dict = {"content": result.document_summary}
                        else:
                            doc_summary_dict = {"content": str(result.document_summary)}
                except Exception as dict_err:
                    logger.warning(f"Error converting document summary to dict: {str(dict_err)}")
                    doc_summary_dict = {"error": "Could not convert summary"}
            metadata = {
                "processed_at": datetime.utcnow().isoformat(),
                "element_count": len(result.elements) if hasattr(result, 'elements') else 0,
                "chunk_count": len(result.chunks) if hasattr(result, 'chunks') else 0,
                "domain_category": result._predict_document_category(
                    result._extract_all_technical_terms(result.elements),
                    result.markdown_content
                ) if hasattr(result, '_predict_document_category') else "general"
            }
            if doc_summary_dict:
                metadata["document_summary"] = doc_summary_dict
            await self.create_document_node(pdf_id=pdf_id, title=title, metadata=metadata)
            elements_by_type = {}
            for element in result.elements:
                element_type = element.content_type.value if hasattr(element.content_type, 'value') else str(element.content_type)
                if element_type not in elements_by_type:
                    elements_by_type[element_type] = []
                elements_by_type[element_type].append(element)
            for element_type, type_elements in elements_by_type.items():
                logger.info(f"Processing {len(type_elements)} elements of type {element_type}")
                batch_size = 50
                for i in range(0, len(type_elements), batch_size):
                    batch = type_elements[i:i+batch_size]
                    for element in batch:
                        try:
                            await self.add_content_element(element, pdf_id)
                        except Exception as element_err:
                            logger.warning(f"Error adding element: {str(element_err)}")
                    if i + batch_size < len(type_elements):
                        logger.info(f"Processed {i + batch_size}/{len(type_elements)} {element_type} elements")
            if hasattr(result, 'concept_network') and result.concept_network:
                await self._add_concept_network(result.concept_network, pdf_id)
            logger.info(f"Successfully ingested content for {pdf_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to ingest content: {str(e)}", exc_info=True)
            return False

    async def _add_concept_network(self, concept_network, pdf_id: str) -> None:
        """
        Helper method to add concept network to Neo4j with enhanced error handling.
        """
        try:
            concept_batch_size = 100
            concepts = concept_network.concepts if hasattr(concept_network, 'concepts') else []
            for i in range(0, len(concepts), concept_batch_size):
                batch = concepts[i:i+concept_batch_size]
                for concept in batch:
                    try:
                        concept_metadata = {
                            "importance": float(concept.importance_score) if hasattr(concept, 'importance_score') else 0.5,
                            "is_primary": bool(concept.is_primary) if hasattr(concept, 'is_primary') else False
                        }
                        if hasattr(concept, 'category') and concept.category is not None:
                            concept_metadata["category"] = str(concept.category)
                        await self.add_concept(concept_name=concept.name, pdf_id=pdf_id, metadata=concept_metadata)
                    except Exception as concept_err:
                        logger.warning(f"Error adding concept {concept.name}: {str(concept_err)}")
                if i + concept_batch_size < len(concepts):
                    logger.info(f"Processed {i + concept_batch_size}/{len(concepts)} concepts")
            rel_batch_size = 100
            relationships = concept_network.relationships if hasattr(concept_network, 'relationships') else []
            for i in range(0, len(relationships), rel_batch_size):
                batch = relationships[i:i+rel_batch_size]
                for rel in batch:
                    try:
                        rel_type = rel.type
                        if hasattr(rel_type, 'value'):
                            rel_type = rel_type.value
                        rel_metadata = {}
                        if hasattr(rel, 'weight'):
                            rel_metadata["weight"] = float(rel.weight)
                        if hasattr(rel, 'context') and rel.context:
                            rel_metadata["context"] = str(rel.context)
                        await self.add_concept_relationship(
                            source=rel.source,
                            target=rel.target,
                            rel_type=str(rel_type),
                            pdf_id=pdf_id,
                            metadata=rel_metadata
                        )
                    except Exception as rel_err:
                        logger.warning(f"Error adding relationship: {str(rel_err)}")
                if i + rel_batch_size < len(relationships):
                    logger.info(f"Processed {i + rel_batch_size}/{len(relationships)} relationships")
        except Exception as e:
            logger.error(f"Error adding concept network: {str(e)}")

    async def _check_document_exists(self, pdf_id: str) -> bool:
        """Check if document exists in Neo4j."""
        if not self.initialized:
            return False
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    "MATCH (d:Document {pdf_id: $pdf_id}) RETURN count(d) as count",
                    {"pdf_id": pdf_id}
                )
                record = await result.single()
                return record and record["count"] > 0
        except Exception as e:
            logger.warning(f"Error checking document existence: {str(e)}")
            return False

    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Legacy compatibility method for similarity search (LangChain compatible).
        """
        pdf_id = None
        content_types = None
        if filter:
            if "pdf_id" in filter:
                pdf_id = filter["pdf_id"]
            if "content_types" in filter:
                content_types = filter["content_types"]
        return await self.semantic_search(query=query, k=k, pdf_id=pdf_id, content_types=content_types)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Legacy compatibility method for adding texts (LangChain compatible).
        """
        import asyncio
        result_ids = []
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            async def add_all_texts():
                nonlocal result_ids
                for i, text in enumerate(texts):
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    element_id = ids[i] if ids and i < len(ids) else f"txt_{uuid.uuid4().hex}"
                    result_ids.append(element_id)
                    pdf_id = metadata.get("pdf_id", "unknown")
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
                    await self.add_content_element(element, pdf_id)
            loop.run_until_complete(add_all_texts())
            return result_ids
        except Exception as e:
            logger.error(f"Error in add_texts: {str(e)}")
            return []

    async def check_health(self) -> Dict[str, Any]:
        """Check the health status of the Neo4j connection and database."""
        health_info = {
            "status": "unknown",
            "connection": False,
            "database_ready": False,
            "indexes": [],
            "vector_indexes": [],
            "node_counts": {},
            "last_error": None,
            "timestamp": datetime.now().isoformat()
        }
        if not self.initialized:
            health_info["status"] = "not_initialized"
            return health_info
        try:
            connection_result = await self.ensure_connection(max_retries=1)
            health_info["connection"] = connection_result
            if not connection_result:
                health_info["status"] = "connection_failed"
                return health_info
            async with self.driver.session(database=self.database) as session:
                result = await session.run("SHOW INDEXES")
                indexes = []
                async for record in result:
                    index_info = {
                        "name": record.get("name", "unknown"),
                        "type": record.get("type", "unknown"),
                        "uniqueness": record.get("uniqueness", "unknown"),
                        "state": record.get("state", "unknown")
                    }
                    indexes.append(index_info)
                    if "vector" in record.get("type", "").lower():
                        health_info["vector_indexes"].append(index_info)
                health_info["indexes"] = indexes
                health_info["database_ready"] = any(idx.get("name") == "content_vector" for idx in indexes)
                try:
                    result = await session.run("""
                        MATCH (n)
                        WITH labels(n) AS labels, count(n) AS count
                        RETURN labels, count
                        ORDER BY count DESC
                    """)
                    counts = {}
                    async for record in result:
                        label = ":".join(record["labels"]) or "no_label"
                        counts[label] = record["count"]
                    health_info["node_counts"] = counts
                except Exception as count_err:
                    logger.error(f"Error counting nodes: {str(count_err)}")
                    health_info["node_counts"] = {"error": str(count_err)}
                health_info["status"] = "ready" if health_info["database_ready"] else "schema_incomplete"
                return health_info
        except Exception as e:
            health_info["status"] = "error"
            health_info["last_error"] = str(e)
            logger.error(f"Error checking Neo4j health: {str(e)}")
            return health_info

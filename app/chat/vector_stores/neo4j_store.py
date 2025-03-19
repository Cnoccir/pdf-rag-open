"""
Simplified Neo4j vector store implementation for LangGraph architecture.
Provides graph-based storage and retrieval with improved connection management.
"""

import os
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

class Neo4jVectorStore:
    """
    Neo4j vector store with simplified connection management and error handling.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for connection management"""
        if cls._instance is None:
            cls._instance = super(Neo4jVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        url: str = None,
        username: str = None,
        password: str = None,
        database: str = "neo4j",
        embedding_dimension: int = 1536,
        embedding_model: str = "text-embedding-3-small"
    ):
        # Skip initialization if already done (singleton pattern)
        if self._initialized:
            return

        # Use parameters or environment variables
        self.url = url or os.environ.get("NEO4J_URL", "bolt://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.database = database
        self.embedding_dimension = embedding_dimension
        self.embedding_model = embedding_model

        # Track initialization state
        self._initialized = False
        self.driver = None
        self.error = None

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Initialize metrics
        self.metrics = {
            "queries": 0,
            "errors": 0,
            "avg_query_time": 0,
            "total_query_time": 0
        }

        # Initialize connection
        self.initialize()

        logger.info(f"Neo4j Vector Store initialized with {self.embedding_dimension} dimensions")

    def initialize(self) -> bool:
        """
        Initialize the Neo4j connection and database schema.
        Returns: Success status
        """
        if self._initialized and self.driver:
            return True

        try:
            # Create driver with optimized connection settings
            self.driver = GraphDatabase.driver(
                self.url,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_timeout=30.0
            )

            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()

                if not record or record.get("test") != 1:
                    raise Exception("Failed to connect to Neo4j - database not responding correctly")

                # Set up constraints and indices
                self._setup_schema(session)

            self._initialized = True
            logger.info("Neo4j connection and schema initialized successfully")
            return True

        except Exception as e:
            self.error = str(e)
            logger.error(f"Failed to initialize Neo4j: {str(e)}")

            # Clean up failed connection
            if self.driver:
                self.driver.close()
                self.driver = None

            self._initialized = False
            return False

    def _setup_schema(self, session) -> None:
        """Set up Neo4j schema, constraints and indices"""
        try:
            # Create document uniqueness constraint
            session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.pdf_id IS UNIQUE")

            # Create content element index
            session.run("CREATE INDEX content_element_id IF NOT EXISTS FOR (e:ContentElement) ON (e.element_id)")

            # Create vector index if it doesn't exist
            try:
                session.run(
                    """
                    CALL db.index.vector.createNodeIndex(
                        'content_vector',
                        'ContentElement',
                        'embedding',
                        $dimension,
                        'cosine'
                    )
                    """,
                    {"dimension": self.embedding_dimension}
                )
                logger.info("Created vector index: content_vector")
            except Exception as e:
                # Check if this is because the index already exists
                if "already exists" in str(e):
                    logger.info("Vector index content_vector already exists")
                else:
                    logger.warning(f"Error creating vector index: {str(e)}")

        except Exception as e:
            logger.error(f"Error setting up schema: {str(e)}")
            raise

    def semantic_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None,
        content_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Perform semantic search using vector similarity.

        Args:
            query: Query text
            k: Number of results to return
            pdf_id: Optional PDF ID to filter by
            content_types: Optional content types to filter by

        Returns:
            List of document results
        """
        if not self._initialized:
            if not self.initialize():
                return []

        start_time = datetime.now()

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

            # Add filters if needed
            filters = []
            if pdf_id:
                filters.append("EXISTS((node)<-[:CONTAINS]-(:Document {pdf_id: $pdf_id}))")
            if content_types and len(content_types) > 0:
                filters.append("node.content_type IN $content_types")

            if filters:
                cypher_query += " WHERE " + " AND ".join(filters)

            # Add return clause
            cypher_query += """
                MATCH (d:Document)-[:CONTAINS]->(node)
                RETURN
                    node.element_id AS id,
                    node.content AS content,
                    node.content_type AS content_type,
                    d.pdf_id AS pdf_id,
                    node.page_number AS page_number,
                    node.section AS section,
                    score
                ORDER BY score DESC
                LIMIT $limit
            """

            # Set parameters
            params = {
                "embedding": query_embedding,
                "k": k * 2,  # Fetch more for filtering
                "limit": k
            }

            if pdf_id:
                params["pdf_id"] = pdf_id
            if content_types and len(content_types) > 0:
                params["content_types"] = content_types

            # Execute query
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query, params)

                # Process results
                documents = []
                for record in result:
                    metadata = {
                        "id": record["id"],
                        "content_type": record["content_type"] or "text",
                        "pdf_id": record["pdf_id"] or "",
                        "page_number": record["page_number"] or 0,
                        "section": record["section"] or "",
                        "score": record["score"] or 0.0
                    }

                    doc = Document(page_content=record["content"] or "", metadata=metadata)
                    documents.append(doc)

            # Update metrics
            query_time = (datetime.now() - start_time).total_seconds()
            self.metrics["queries"] += 1
            self.metrics["total_query_time"] += query_time
            self.metrics["avg_query_time"] = self.metrics["total_query_time"] / self.metrics["queries"]

            logger.info(f"Semantic search completed in {query_time:.2f}s, found {len(documents)} results")
            return documents

        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            self.metrics["errors"] += 1
            return []

    def keyword_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None
    ) -> List[Document]:
        """
        Perform keyword search using standard index matching.

        Args:
            query: Query text
            k: Number of results to return
            pdf_id: Optional PDF ID to filter by

        Returns:
            List of document results
        """
        if not self._initialized:
            if not self.initialize():
                return []

        try:
            # Extract keywords
            keywords = [kw.strip().lower() for kw in query.split() if len(kw.strip()) > 2]
            if not keywords:
                return []

            # Build regex pattern
            search_terms = [f"(?i).*{keyword}.*" for keyword in keywords]
            regex_pattern = "|".join(search_terms)

            # Build Cypher query
            cypher_query = """
                MATCH (c:ContentElement)
                WHERE c.content =~ $regex_pattern
            """

            if pdf_id:
                cypher_query += """
                    AND EXISTS((c)<-[:CONTAINS]-(:Document {pdf_id: $pdf_id}))
                """

            cypher_query += """
                MATCH (d:Document)-[:CONTAINS]->(c)
                RETURN
                    c.element_id AS id,
                    c.content AS content,
                    c.content_type AS content_type,
                    d.pdf_id AS pdf_id,
                    c.page_number AS page_number,
                    c.section AS section,
                    1.0 AS score
                LIMIT $limit
            """

            # Set parameters
            params = {"regex_pattern": regex_pattern, "limit": k}
            if pdf_id:
                params["pdf_id"] = pdf_id

            # Execute query
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query, params)

                # Process results
                documents = []
                for record in result:
                    metadata = {
                        "id": record["id"],
                        "content_type": record["content_type"] or "text",
                        "pdf_id": record["pdf_id"] or "",
                        "page_number": record["page_number"] or 0,
                        "section": record["section"] or "",
                        "score": record["score"] or 0.0
                    }

                    doc = Document(page_content=record["content"] or "", metadata=metadata)
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
            self.metrics["errors"] += 1
            return []

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        pdf_id: Optional[str] = None,
        content_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Query text
            k: Number of results to return
            pdf_id: Optional PDF ID to filter by
            content_types: Optional content types to filter by

        Returns:
            List of document results
        """
        if not self._initialized:
            if not self.initialize():
                return []

        start_time = datetime.now()

        try:
            # Perform semantic search
            semantic_results = self.semantic_search(query, k, pdf_id, content_types)

            # Perform keyword search
            keyword_results = self.keyword_search(query, k, pdf_id)

            # Combine results
            combined_results = {}

            # Add semantic results
            for doc in semantic_results:
                doc_id = doc.metadata["id"]
                combined_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": doc.metadata["score"],
                    "keyword_score": 0.0
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
                        "keyword_score": doc.metadata["score"]
                    }

            # Calculate combined scores
            for doc_id, data in combined_results.items():
                combined_score = (data["semantic_score"] * 0.7) + (data["keyword_score"] * 0.3)
                data["combined_score"] = combined_score
                data["doc"].metadata["score"] = combined_score

            # Sort by combined score
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )

            # Return top k results
            final_results = [data["doc"] for data in sorted_results[:k]]

            # Update metrics
            query_time = (datetime.now() - start_time).total_seconds()
            self.metrics["queries"] += 1
            self.metrics["total_query_time"] += query_time
            self.metrics["avg_query_time"] = self.metrics["total_query_time"] / self.metrics["queries"]

            logger.info(f"Hybrid search completed in {query_time:.2f}s, found {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Hybrid search error: {str(e)}")
            self.metrics["errors"] += 1
            return []

    def add_document(
        self,
        pdf_id: str,
        title: str = "Untitled Document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add document to Neo4j.

        Args:
            pdf_id: PDF ID
            title: Document title
            metadata: Optional metadata

        Returns:
            Success status
        """
        if not self._initialized:
            if not self.initialize():
                return False

        try:
            with self.driver.session(database=self.database) as session:
                # Create document node
                session.run(
                    """
                    MERGE (d:Document {pdf_id: $pdf_id})
                    ON CREATE SET
                        d.title = $title,
                        d.created_at = datetime()
                    ON MATCH SET
                        d.title = $title,
                        d.updated_at = datetime()
                    """,
                    {"pdf_id": pdf_id, "title": title}
                )

                # Add metadata if provided
                if metadata:
                    # Convert complex objects to JSON strings
                    for key, value in metadata.items():
                        if isinstance(value, (dict, list)):
                            metadata[key] = json.dumps(value)

                    # Add metadata properties
                    for key, value in metadata.items():
                        session.run(
                            f"""
                            MATCH (d:Document {{pdf_id: $pdf_id}})
                            SET d.{key} = $value
                            """,
                            {"pdf_id": pdf_id, "value": value}
                        )

            logger.info(f"Added document {pdf_id}: {title}")
            return True

        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False

    def add_content_element(
        self,
        element_id: str,
        content: str,
        content_type: str,
        pdf_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add content element to Neo4j.

        Args:
            element_id: Element ID
            content: Element content
            content_type: Content type
            pdf_id: PDF ID
            metadata: Optional metadata

        Returns:
            Success status
        """
        if not self._initialized:
            if not self.initialize():
                return False

        if not content or not element_id or not pdf_id:
            logger.warning(f"Missing required fields for content element: {element_id}")
            return False

        try:
            # Generate embedding
            embedding = self.embeddings.embed_query(content)

            # Prepare metadata
            page_number = metadata.get("page_number", 0) if metadata else 0
            section = metadata.get("section", "") if metadata else ""

            with self.driver.session(database=self.database) as session:
                # Create content element
                session.run(
                    """
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
                    """,
                    {
                        "pdf_id": pdf_id,
                        "element_id": element_id,
                        "content": content,
                        "content_type": content_type,
                        "page_number": page_number,
                        "section": section,
                        "embedding": embedding
                    }
                )

                # Add additional metadata if provided
                if metadata:
                    # Add technical terms if available
                    if "technical_terms" in metadata and isinstance(metadata["technical_terms"], list):
                        session.run(
                            """
                            MATCH (e:ContentElement {element_id: $element_id})
                            SET e.technical_terms = $terms
                            """,
                            {"element_id": element_id, "terms": metadata["technical_terms"]}
                        )

            logger.debug(f"Added content element {element_id} for document {pdf_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding content element: {str(e)}")
            return False

    def delete_document(self, pdf_id: str) -> bool:
        """
        Delete a document and its content from Neo4j.

        Args:
            pdf_id: PDF ID

        Returns:
            Success status
        """
        if not self._initialized:
            if not self.initialize():
                return False

        try:
            with self.driver.session(database=self.database) as session:
                # Delete content elements
                session.run(
                    """
                    MATCH (d:Document {pdf_id: $pdf_id})-[:CONTAINS]->(e:ContentElement)
                    DETACH DELETE e
                    """,
                    {"pdf_id": pdf_id}
                )

                # Delete document
                session.run(
                    """
                    MATCH (d:Document {pdf_id: $pdf_id})
                    DETACH DELETE d
                    """,
                    {"pdf_id": pdf_id}
                )

            logger.info(f"Deleted document {pdf_id} and its content")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            self._initialized = False
            logger.info("Neo4j connection closed")

    async def check_health(self) -> Dict[str, Any]:
        """
        Check Neo4j database health and connectivity.

        Returns:
            Dictionary with health status information
        """
        health_info = {
            "status": "error",
            "connection": "failed",
            "database_ready": False,
            "timestamp": datetime.utcnow().isoformat()
        }

        if not self.driver or not self._initialized:
            health_info["error"] = "Neo4j driver not initialized"
            return health_info

        try:
            # Test basic connectivity
            with self.driver.session() as session:
                # Basic connectivity check
                result = session.run("RETURN 1 AS n")
                record = result.single()

                if record and record["n"] == 1:
                    health_info["connection"] = "connected"
                    health_info["status"] = "ok"
                    health_info["database_ready"] = True
                else:
                    health_info["error"] = "Database returned unexpected result"
                    return health_info

            # Check for vector index
            with self.driver.session() as session:
                # Check if our embedding index exists
                try:
                    # Try to get a content element with embedding to confirm vectors are working
                    query = """
                    MATCH (e:ContentElement)
                    WHERE e.embedding IS NOT NULL
                    RETURN count(e) AS vector_count
                    LIMIT 1
                    """

                    result = session.run(query)
                    record = result.single()

                    if record and record["vector_count"] > 0:
                        health_info["has_vectors"] = True
                    else:
                        health_info["has_vectors"] = False
                except Exception as e:
                    health_info["vector_error"] = str(e)

            return health_info

        except Exception as e:
            health_info["error"] = str(e)
            return health_info

# Singleton instance getter
def get_vector_store() -> Neo4jVectorStore:
    """Get or create Neo4j vector store instance"""
    return Neo4jVectorStore()

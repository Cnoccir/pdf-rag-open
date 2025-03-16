# app/chat/vector_stores/__init__.py
"""
Vector store module with unified implementation for Neo4j integration.
"""

import os
import logging
from typing import Optional

from app.chat.vector_stores.neo4j_store import Neo4jVectorStore

logger = logging.getLogger(__name__)

_vector_store_instance = None

def get_vector_store() -> Neo4jVectorStore:
    """
    Get unified vector store instance.
    Returns Neo4j vector store by default.
    
    Returns:
        Vector store instance
    """
    global _vector_store_instance
    
    if _vector_store_instance is None:
        logger.info("Initializing Neo4j vector store")
        try:
            neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
            
            _vector_store_instance = Neo4jVectorStore(
                url=neo4j_url,
                username=neo4j_user,
                password=neo4j_password
            )
            
            logger.info("Neo4j vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j vector store: {str(e)}")
            raise
    
    return _vector_store_instance

__all__ = ["Neo4jVectorStore", "get_vector_store"]
# app/chat/vector_stores/__init__.py
"""
Vector store module for the PDF RAG system.
This module provides functionality for storing and retrieving document vectors.
"""

# Export only the new TechDocVectorStore and helper functions
from .vector_store import TechDocVectorStore, get_vector_store, VectorStoreMetrics, CachedEmbeddings

__all__ = ["TechDocVectorStore", "get_vector_store", "VectorStoreMetrics", "CachedEmbeddings"]

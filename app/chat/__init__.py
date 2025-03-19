"""
PDF RAG system with LangGraph architecture and Neo4j integration.
Provides document processing, querying, and multi-document research capabilities.
"""

import logging
logger = logging.getLogger(__name__)

# Core type definitions
from app.chat.types import (
    # Core Enums
    ContentType,
    ResearchMode,
    RelationType,

    # Concept Models
    Concept,
    ConceptRelationship,
    ConceptNetwork,

    # Image Models
    ImageFeatures,
    ImageAnalysis,
    ImagePaths,
    ImageMetadata,

    # Table Models
    TableData,

    # Core Models
    ContentElement,
    ContentMetadata,

    # Research Models
    ResearchManager,
    ResearchContext,

    # Search Models
    SearchQuery,

    # Processing Models
    ProcessingResult,
    ProcessingConfig,

    # Chat Models
    ChatArgs
)

# Chat manager for orchestrating the LangGraph workflow
from app.chat.chat_manager import ChatManager

# Document processor for PDF processing
from app.chat.document_fetcher import process_technical_document

# Vector store for Neo4j access
from app.chat.vector_stores import get_vector_store, Neo4jVectorStore

# Memory manager for conversation persistence
from app.chat.memories.memory_manager import MemoryManager

def initialize_chat(chat_args):
    """
    Initialize chat with LangGraph architecture.

    Args:
        chat_args: Chat configuration

    Returns:
        ChatManager instance
    """
    logger.info("Initializing chat with LangGraph architecture")

    # Ensure research mode is correctly set
    if not hasattr(chat_args, "research_mode") or chat_args.research_mode is None:
        chat_args.research_mode = ResearchMode.SINGLE

    # Create chat manager
    chat_manager = ChatManager(chat_args)

    # Initialize conversation history
    chat_manager.initialize()

    return chat_manager

__all__ = [
    # Core Enums
    "ContentType",
    "ResearchMode",
    "RelationType",

    # Concept Models
    "Concept",
    "ConceptRelationship",
    "ConceptNetwork",

    # Image Models
    "ImageFeatures",
    "ImageAnalysis",
    "ImagePaths",
    "ImageMetadata",

    # Table Models
    "TableData",

    # Core Models
    "ContentElement",
    "ContentMetadata",

    # Research Models
    "ResearchManager",
    "ResearchContext",

    # Search Models
    "SearchQuery",

    # Processing Models
    "ProcessingResult",
    "ProcessingConfig",

    # Chat Models
    "ChatArgs",

    # Chat Functions
    "initialize_chat",
    "ChatManager",

    # Document Processing
    "process_technical_document",

    # Vector Store
    "get_vector_store",
    "Neo4jVectorStore",

    # Memory Management
    "MemoryManager"
]

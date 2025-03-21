"""
PDF RAG system with LangGraph architecture and MongoDB/Qdrant integration.
Provides document processing, querying, and multi-document research capabilities.
"""

import logging
logger = logging.getLogger(__name__)

# Core type definitions
from app.chat.types import (
    # Core Enums
    ContentType,
    ChunkLevel,
    EmbeddingType,
    ResearchMode,
    RelationType,

    # Content Element Models
    ContentMetadata,
    ContentElement,

    # Chunking Models
    ChunkMetadata,
    DocumentChunk,

    # Concept Models
    Concept,
    ConceptRelationship,
    ConceptNetwork,

    # Processing Models
    ProcessingConfig,
    ProcessingResult,

    # Chat Management Models
    ChatArgs
)

# Chat manager for orchestrating the LangGraph workflow
from app.chat.chat_manager import ChatManager

from app.chat.research.research_manager import EnhancedResearchManager, ResearchContex

# Document processor for PDF processing
from app.chat.document_fetcher import process_technical_document

# Vector store for database access
from app.chat.vector_stores import (
    get_vector_store,
    get_mongo_store,
    get_qdrant_store,
    UnifiedVectorStore
)

# Memory manager for conversation persistence
from app.chat.memories.memory_manager import MemoryManager

# LangGraph components
from app.chat.langgraph import (
    GraphState,
    QueryState,
    RetrievalState,
    GenerationState,
    ResearchState,
    ConversationState,

    create_document_graph,
    create_query_graph,
    create_research_graph
)

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

def initialize_vector_stores():
    """Initialize MongoDB and Qdrant vector stores."""
    logger.info("Initializing vector stores...")

    mongo_store = get_mongo_store()
    qdrant_store = get_qdrant_store()
    vector_store = get_vector_store()

    # Initialize connections
    if hasattr(vector_store, "initialize"):
        vector_store.initialize()

    logger.info("Vector stores initialized")
    return vector_store

__all__ = [
    # Core Enums
    "ContentType",
    "ResearchMode",
    "RelationType",
    "ChunkLevel",
    "EmbeddingType",

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
    "SearchResult",

    # Processing Models
    "ProcessingResult",
    "ProcessingConfig",

    # Chat Models
    "ChatArgs",

    # LangGraph States
    "GraphState",
    "QueryState",
    "RetrievalState",
    "GenerationState",
    "ResearchState",
    "ConversationState",

    # LangGraph Graphs
    "create_document_graph",
    "create_query_graph",
    "create_research_graph",

    # Chat Functions
    "initialize_chat",
    "initialize_vector_stores",
    "ChatManager",

    # Document Processing
    "process_technical_document",

    # Vector Store
    "get_vector_store",
    "get_mongo_store",
    "get_qdrant_store",
    "UnifiedVectorStore",

    "EnhancedResearchManager",
    "ResearchManager",
    "ResearchContex",

    # Memory Management
    "MemoryManager"
]

#app/chat/__init__.py
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
    DocumentSummary,

    # Search Models
    SearchQuery,

    # Processing Models
    ProcessingResult,
    ProcessingConfig
)

# Chat and metadata models
from app.chat.models import (
    ChatArgs,
    Metadata,
    TechnicalMetadata,
    ConceptMetadata
)

# Import document processor
from app.chat.langgraph.nodes.document_processor import DocumentProcessor

# Import chat manager for LangGraph
from app.chat.chat_manager import ChatManager

def initialize_chat(chat_args):
    """
    Initialize chat with LangGraph architecture.
    
    Args:
        chat_args: Chat arguments
        
    Returns:
        ChatManager instance
    """
    logger.info("Initializing chat with LangGraph architecture")
    
    # Ensure research mode is correctly set
    if not hasattr(chat_args, "research_mode") or chat_args.research_mode is None:
        chat_args.research_mode = ResearchMode.SINGLE
        
    # Ensure stream_enabled is set
    if not hasattr(chat_args, "stream_enabled"):
        chat_args.stream_enabled = True
        
    # Default chunk size for streaming
    if not hasattr(chat_args, "stream_chunk_size"):
        chat_args.stream_chunk_size = 20
        
    # Return the chat manager
    return ChatManager(chat_args)

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
    "DocumentSummary",

    # Search Models
    "SearchQuery",

    # Processing Models
    "ProcessingResult",
    "ProcessingConfig",

    # Chat Models
    "ChatArgs",
    "Metadata",
    "TechnicalMetadata",
    "ConceptMetadata",

    # Document Processing
    "DocumentProcessor",

    # Chat Functions
    "initialize_chat",
    "ChatManager"
]

"""
State definitions for the PDF RAG LangGraph implementation.
Updated for MongoDB + Qdrant integration with multi-level chunking.
"""

from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

from app.chat.types import ChunkLevel, EmbeddingType, ContentType

# -----------------------
# Type Definitions
# -----------------------
class MessageType(str, Enum):
    """Message types in a conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class RetrievalStrategy(str, Enum):
    """Retrieval strategies for document search."""
    SEMANTIC = "semantic"     # Semantic search using embeddings
    KEYWORD = "keyword"       # Keyword-based search
    HYBRID = "hybrid"         # Combination of semantic and keyword
    CONCEPT = "concept"       # Concept-based navigation
    PROCEDURE = "procedure"   # Procedure-focused retrieval

# -----------------------
# Message Models
# -----------------------
class Message(BaseModel):
    """Message in a conversation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

# -----------------------
# Conversation State
# -----------------------
class ConversationState(BaseModel):
    """Conversation state for LangGraph architecture"""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Untitled Conversation"
    pdf_id: str = ""
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    technical_concepts: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, type: Union[MessageType, str], content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to the conversation"""
        if isinstance(type, str):
            type = MessageType(type)

        message = Message(
            type=type,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation state to dictionary"""
        return {
            "id": self.conversation_id,
            "title": self.title,
            "pdf_id": self.pdf_id,
            "messages": [msg.dict() for msg in self.messages],
            "metadata": self.metadata,
            "technical_concepts": self.technical_concepts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

# -----------------------
# Graph State Components
# -----------------------
class QueryState(BaseModel):
    """State for query understanding"""
    query: str
    pdf_ids: List[str] = Field(default_factory=list)
    query_type: Optional[str] = None
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    focused_elements: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Multi-level chunking preferences
    preferred_chunk_level: Optional[ChunkLevel] = None

    # Multi-embedding preferences
    preferred_embedding_type: Optional[EmbeddingType] = None

    # Procedure-specific fields
    procedure_focused: bool = False
    parameter_query: bool = False

class RetrievalState(BaseModel):
    """State for retrieval results"""
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    strategies_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # New fields for multi-level chunking
    chunk_levels_used: List[str] = Field(default_factory=list)

    # New fields for multi-embedding
    embedding_types_used: List[str] = Field(default_factory=list)

    # Procedure-specific fields
    procedures_retrieved: List[Dict[str, Any]] = Field(default_factory=list)
    parameters_retrieved: List[Dict[str, Any]] = Field(default_factory=list)

class GenerationState(BaseModel):
    """State for response generation"""
    response: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)

    # New field for procedure step breakdown
    procedure_steps: List[Dict[str, Any]] = Field(default_factory=list)

class ResearchState(BaseModel):
    """State for multi-document research"""
    query_state: Optional[QueryState] = None
    cross_references: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    research_context: Any = None  # Will hold ResearchContext from types.py

# -----------------------
# Combined Graph State
# -----------------------
class GraphState(BaseModel):
    """Combined state for the entire LangGraph"""
    # Core states
    query_state: Optional[QueryState] = None
    retrieval_state: Optional[RetrievalState] = None
    generation_state: Optional[GenerationState] = None
    research_state: Optional[ResearchState] = None
    conversation_state: Optional[ConversationState] = None

    # Optional document processing state
    document_state: Optional[Dict[str, Any]] = None

    class Config:
        """Pydantic config"""
        validate_assignment = True

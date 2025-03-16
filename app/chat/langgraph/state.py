"""
State definitions for the PDF RAG LangGraph implementation.
These state models define the data structures passed between LangGraph nodes.
"""

from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum, auto
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


# -----------------------
# Basic Type Definitions
# -----------------------
class MessageType(str, Enum):
    """Message types in a conversation"""
    SYSTEM = "system" 
    USER = "user"
    AI = "assistant"
    TOOL = "tool"

class RetrievalStrategy(str, Enum):
    """Retrieval strategies for document search."""
    SEMANTIC = "semantic"     # Semantic search using embeddings
    KEYWORD = "keyword"       # Keyword-based search  
    HYBRID = "hybrid"         # Combination of semantic and keyword
    CONCEPT = "concept"       # Concept-based navigation
    TABLE = "table"           # Table-specific search
    IMAGE = "image"           # Image-specific search
    COMBINED = "combined"     # Multi-strategy approach

class ContentType(str, Enum):
    """Content types in documents."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    EQUATION = "equation"
    DIAGRAM = "diagram"
    PAGE = "page"

# -----------------------
# Message Models
# -----------------------
class Message(BaseModel):
    """Message in a conversation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType  # "user", "assistant", "system", "tool"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

# -----------------------
# Conversation State
# -----------------------
class ConversationState(BaseModel):
    """Full conversation state for LangGraph architecture"""
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

# -----------------------
# LangGraph State Components
# -----------------------
class QueryState(BaseModel):
    """State for query understanding nodes."""
    query: str
    pdf_ids: List[str] = Field(default_factory=list)
    query_type: Optional[str] = None
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    focused_elements: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalState(BaseModel):
    """State for retrieval nodes."""
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    strategies_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GenerationState(BaseModel):
    """State for response generation nodes."""
    response: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)

class ResearchState(BaseModel):
    """State for multi-document research nodes."""
    query_state: Optional[QueryState] = None
    cross_references: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# -----------------------
# Combined Graph State
# -----------------------
class GraphState(BaseModel):
    """Combined state for the entire LangGraph."""
    document_state: Optional[Dict[str, Any]] = None
    query_state: Optional[QueryState] = None
    retrieval_state: Optional[RetrievalState] = None
    generation_state: Optional[GenerationState] = None
    research_state: Optional[ResearchState] = None
    conversation_state: Optional[ConversationState] = None
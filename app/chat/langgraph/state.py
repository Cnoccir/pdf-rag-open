"""
State definitions for the PDF RAG LangGraph implementation.
These state models define the data structures passed between LangGraph nodes.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from app.chat.types import (
    ContentElement,
    ContentType,
    ConceptNetwork,
    DocumentSummary,
    ResearchContext,
    SearchQuery
)
from app.chat.models.conversation import ConversationState, Message, MessageType

class RetrievalStrategy(str, Enum):
    """Retrieval strategies for document search."""
    SEMANTIC = "semantic"         # Semantic search using embeddings
    KEYWORD = "keyword"           # Keyword-based search
    HYBRID = "hybrid"             # Combination of semantic and keyword
    CONCEPT = "concept"           # Concept-based navigation
    TABLE = "table"               # Table-specific search
    IMAGE = "image"               # Image-specific search
    COMBINED = "combined"         # Multi-strategy approach

class DocumentState(BaseModel):
    """State for document processing nodes."""
    pdf_id: str
    processing_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "pending"
    elements: List[ContentElement] = Field(default_factory=list)
    concept_network: Optional[ConceptNetwork] = None
    document_summary: Optional[DocumentSummary] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

class QueryState(BaseModel):
    """State for query understanding nodes."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    pdf_ids: List[str] = Field(default_factory=list)
    query_type: Optional[str] = None
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    focused_elements: List[ContentType] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_search_query(self) -> SearchQuery:
        """Convert to SearchQuery for vector store."""
        return SearchQuery(
            query=self.query,
            pdf_ids=self.pdf_ids,
            content_types=self.focused_elements,
            metadata_filters=self.metadata,
            strategy=self.retrieval_strategy.value,
            concepts=self.concepts,
            keywords=self.keywords
        )

class RetrievalState(BaseModel):
    """State for retrieval nodes."""
    query_state: QueryState
    elements: List[ContentElement] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    strategies_used: List[RetrievalStrategy] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GenerationState(BaseModel):
    """State for response generation nodes."""
    retrieval_state: RetrievalState
    response: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)

class ResearchState(BaseModel):
    """State for multi-document research nodes."""
    query_state: QueryState
    research_context: Optional[ResearchContext] = None
    cross_references: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GraphState(BaseModel):
    """Combined state for the entire graph."""
    document_state: Optional[DocumentState] = None
    query_state: Optional[QueryState] = None
    retrieval_state: Optional[RetrievalState] = None
    generation_state: Optional[GenerationState] = None
    research_state: Optional[ResearchState] = None
    conversation_state: Optional[ConversationState] = None
    
    def get_response(self) -> Optional[str]:
        """Get the final response if available."""
        if self.generation_state and self.generation_state.response:
            return self.generation_state.response
        return None
    
    def add_message(self, type: MessageType, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation history."""
        if not self.conversation_state:
            self.conversation_state = ConversationState()
            
        self.conversation_state.add_message(type, content, metadata or {})
    
    def get_conversation_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get formatted conversation context for LLM."""
        if not self.conversation_state:
            return []
            
        # Get messages in chronological order, most recent first
        messages = self.conversation_state.messages.copy()
        
        # Only include user and assistant messages, not system
        context_messages = [
            {"role": "user" if msg.type == "user" else "assistant", "content": msg.content}
            for msg in messages 
            if msg.type in ["user", "assistant"]
        ]
        
        # Return the most recent messages, limited by max_messages
        return context_messages[-max_messages:] if max_messages > 0 else context_messages

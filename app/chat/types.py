"""
Core type definitions for the PDF RAG system.
Focused on LangGraph architecture without legacy compatibility layers.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, model_validator, constr
import uuid
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------------
class ContentType(str, Enum):
    """Types of content elements in a document."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    EQUATION = "equation"
    DIAGRAM = "diagram"
    PAGE = "page"

class ResearchMode(str, Enum):
    """Research mode for document analysis."""
    SINGLE = "single"  # Single document analysis
    MULTI = "multi"    # Multi-document analysis

class MessageRole(str, Enum):
    """Message role in a chat conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class RelationType(str, Enum):
    """Types of relationships between concepts."""
    IS_A = "is_a"  # Hierarchical relationship
    PART_OF = "part_of"  # Composition relationship
    HAS_PROPERTY = "has_property"  # Property relationship
    RELATES_TO = "relates_to"  # General relationship
    DEFINES = "defines"  # Definition relationship
    LEADS_TO = "leads_to"  # Causal relationship
    OPPOSES = "opposes"  # Opposing relationship
    SUPPORTS = "supports"  # Supporting relationship
    USED_FOR = "used_for"  # Usage relationship
    
    @classmethod
    def map_type(cls, type_str: str) -> 'RelationType':
        """Map a string to a RelationType enum value."""
        try:
            return cls(type_str.lower())
        except ValueError:
            return cls.RELATES_TO

# ------------------------------------------------------------------------
# Chat Models
# ------------------------------------------------------------------------
class TokenUsage(BaseModel):
    """Model for tracking token usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add token usage counts from another TokenUsage instance."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )
    
    def update(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Update token counts."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens

class ChatMessage(BaseModel):
    """Model for chat message with role, content, and metadata."""
    role: MessageRole
    content: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    
    @classmethod
    def system(cls, content: str) -> 'ChatMessage':
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> 'ChatMessage':
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> 'ChatMessage':
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)
    
    @classmethod
    def function(cls, content: str) -> 'ChatMessage':
        """Create a function message."""
        return cls(role=MessageRole.FUNCTION, content=content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "citations": self.citations
        }

class ChatState(BaseModel):
    """Model for chat state with history, context, and settings."""
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pdf_id: Optional[str] = None
    pdf_ids: List[str] = Field(default_factory=list)
    messages: List[ChatMessage] = Field(default_factory=list)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    active_pdf_id: Optional[str] = None
    research_mode: bool = False
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    stream_id: Optional[str] = None
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the chat history."""
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
    
    def add_context(self, context_item: Dict[str, Any]) -> None:
        """Add context item for the current conversation."""
        self.context.append(context_item)
    
    def set_active_pdf(self, pdf_id: str) -> None:
        """Set the active PDF for the conversation."""
        self.active_pdf_id = pdf_id
        if pdf_id not in self.pdf_ids:
            self.pdf_ids.append(pdf_id)
    
    def toggle_research_mode(self, enable: bool = True) -> None:
        """Toggle research mode on/off."""
        self.research_mode = enable

class ChatArgs:
    """Arguments for a chat interaction"""
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        pdf_id: Optional[str] = None,
        research_mode: ResearchMode = ResearchMode.SINGLE,
        stream_enabled: bool = False,
        stream_chunk_size: int = 10
    ):
        self.conversation_id = conversation_id
        self.pdf_id = pdf_id
        self.research_mode = research_mode
        self.stream_enabled = stream_enabled
        self.stream_chunk_size = stream_chunk_size

# ------------------------------------------------------------------------
# Document Content Models
# ------------------------------------------------------------------------
class ContentMetadata(BaseModel):
    """Metadata for a content element."""
    page: int = 0
    source: str = ""
    relevance_score: float = 0.0

class ContentElement(BaseModel):
    """A content element from a document."""
    type: ContentType = ContentType.TEXT
    content: str
    metadata: ContentMetadata = Field(default_factory=ContentMetadata)

class TableData(BaseModel):
    """Model for structured table data extracted from documents."""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    caption: Optional[str] = None
    markdown: Optional[str] = None
    summary: Optional[str] = None
    row_count: int = 0
    column_count: int = 0
    technical_concepts: List[str] = Field(default_factory=list)

class DocumentSummary(BaseModel):
    """Model for document summary with key insights and concepts."""
    title: str
    author: Optional[str] = None
    document_type: Optional[str] = None
    primary_concepts: List[str] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    technical_terms: List[str] = Field(default_factory=list)
    summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ------------------------------------------------------------------------
# Processing Models
# ------------------------------------------------------------------------
class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    pdf_id: constr(min_length=1)
    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    embedding_model: str = Field(default="text-embedding-3-small")
    process_images: bool = True
    process_tables: bool = True
    search_limit: Optional[int] = 5

class ProcessingResult(BaseModel):
    """Result of document processing."""
    pdf_id: str
    elements: List[ContentElement] = Field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ------------------------------------------------------------------------
# Search Models
# ------------------------------------------------------------------------
class SearchQuery(BaseModel):
    """Query for searching documents."""
    query: str
    pdf_id: Optional[str] = None
    limit: int = Field(default=5, gt=0)

class SearchResult(BaseModel):
    """Result of a document search."""
    query: SearchQuery
    elements: List[ContentElement] = Field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ------------------------------------------------------------------------
# Concept Network Models
# ------------------------------------------------------------------------
class Concept(BaseModel):
    """Model for a concept in the concept network."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    source_document: Optional[str] = None
    source_page: Optional[int] = None
    importance: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConceptRelationship(BaseModel):
    """Model for a relationship between concepts."""
    source_id: str
    target_id: str
    type: RelationType = RelationType.RELATES_TO
    strength: float = 0.5
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConceptNetwork(BaseModel):
    """Model for a network of concepts and their relationships."""
    concepts: List[Concept] = Field(default_factory=list)
    relationships: List[ConceptRelationship] = Field(default_factory=list)
    
    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the network."""
        self.concepts.append(concept)
    
    def add_relationship(self, relationship: ConceptRelationship) -> None:
        """Add a relationship to the network."""
        self.relationships.append(relationship)

# ------------------------------------------------------------------------
# Image Models
# ------------------------------------------------------------------------
class ImagePaths(BaseModel):
    """Model for image file paths."""
    original: str
    processed: Optional[str] = None
    thumbnail: Optional[str] = None

class ImageMetadata(BaseModel):
    """Model for image metadata."""
    width: int
    height: int
    format: str
    page: int
    position: Dict[str, float] = Field(default_factory=dict)

class ImageFeatures(BaseModel):
    """Model for image features extracted from analysis."""
    embedding: List[float] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    objects: List[Dict[str, Any]] = Field(default_factory=list)
    text_blocks: List[Dict[str, Any]] = Field(default_factory=list)

class ImageAnalysis(BaseModel):
    """Model for image analysis results."""
    image_id: str
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    objects: List[Dict[str, Any]] = Field(default_factory=list)
    features: Optional[ImageFeatures] = None
    metadata: ImageMetadata
    paths: ImagePaths
    error: Optional[str] = None

# ------------------------------------------------------------------------
# Research Models
# ------------------------------------------------------------------------
class DocumentReference(BaseModel):
    """Reference to a document in the research corpus."""
    pdf_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None
    primary_concepts: Optional[List[str]] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    active: bool = True

class CrossDocumentReference(BaseModel):
    """Evidence connecting concepts across documents."""
    concept: str
    documents: List[str]
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    context: Optional[str] = None

class ResearchContext(BaseModel):
    """Research context for multi-document analysis."""
    documents: List[DocumentReference] = Field(default_factory=list)
    active_documents: List[str] = Field(default_factory=list)
    cross_document_evidence: List[CrossDocumentReference] = Field(default_factory=list)
    concept_network: Optional[ConceptNetwork] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_document(self, document: DocumentReference) -> None:
        """Add a document to the research context."""
        self.documents.append(document)
        
    def set_active_documents(self, pdf_ids: List[str]) -> None:
        """Set active documents for research."""
        self.active_documents = pdf_ids
        
    def add_cross_document_evidence(self, evidence: CrossDocumentReference) -> None:
        """Add cross-document evidence."""
        self.cross_document_evidence.append(evidence)

class ResearchResult(BaseModel):
    """Results from a research operation across multiple documents."""
    query: str
    documents: List[DocumentReference] = Field(default_factory=list)
    concepts: List[Concept] = Field(default_factory=list)
    cross_references: List[CrossDocumentReference] = Field(default_factory=list)
    summary: Optional[str] = None
    elements: List[ContentElement] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class ResearchManager:
    """
    Forward reference for the ResearchManager class to avoid circular imports.
    
    The actual implementation is in app.chat.research.research_manager.
    """
    pass

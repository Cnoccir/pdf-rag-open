# app/chat/models/__init__.py

import sys
from pydantic import BaseModel, Field, validator
from typing import (
    Optional, Dict, Any, List, Set, TYPE_CHECKING
)
from datetime import datetime

# If your code uses ChatOpenAI:
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = Any  # fallback if not installed

if TYPE_CHECKING:
    from app.chat.research.research_manager import ResearchManager
    RetrievalManager = Any
else:
    ResearchManager = Any
    RetrievalManager = Any

# Import enumerations and partial types from app.chat.types
from app.chat.types import (
    ResearchMode,
    ProcessingConfig,
    ContentType,
    ConceptNetwork
)

# ---------------------------------------------------------------------
# CANONICAL RESEARCH CONTEXT
# ---------------------------------------------------------------------
class ResearchContext(BaseModel):
    """
    Canonical Research Context for cross-document analysis.
    Consolidates the context used in multi-document "research" mode.
    """
    primary_pdf_id: Optional[str] = None
    active_pdf_ids: Set[str] = Field(default_factory=set)
    document_titles: Dict[str, str] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure that primary_pdf_id is in active_pdf_ids
        if self.primary_pdf_id:
            self.active_pdf_ids.add(self.primary_pdf_id)

    def add_document(self, pdf_id: str, title: Optional[str] = None) -> None:
        """Add a document to the research context."""
        self.active_pdf_ids.add(pdf_id)
        if title:
            self.document_titles[pdf_id] = title
        self.last_updated = datetime.utcnow()

    def remove_document(self, pdf_id: str) -> None:
        """Remove a document from the research context."""
        if pdf_id != self.primary_pdf_id and pdf_id in self.active_pdf_ids:
            self.active_pdf_ids.remove(pdf_id)
            self.document_titles.pop(pdf_id, None)
        self.last_updated = datetime.utcnow()

# ---------------------------------------------------------------------
# METADATA AND SUPPORT MODELS
# ---------------------------------------------------------------------
class ConceptMetadata(BaseModel):
    concepts: List[str] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 1.0
    doc_coverage: Set[str] = Field(default_factory=set)

class TechnicalMetadata(BaseModel):
    doc_type: Optional[str] = None
    component_types: Set[str] = Field(default_factory=set)
    patterns: Set[str] = Field(default_factory=set)
    current_section: Optional[str] = None
    hierarchy: List[str] = Field(default_factory=list)
    concept_data: Optional[ConceptMetadata] = None
    
class Metadata(BaseModel):
    conversation_id: str
    user_id: str
    pdf_id: str

    technical_context: TechnicalMetadata = Field(default_factory=TechnicalMetadata)
    research_context: Dict[str, Any] = Field(default_factory=dict)
    concept_network: Optional[ConceptNetwork] = None
    document_names: Dict[str, str] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# ---------------------------------------------------------------------
# CANONICAL CHATARGS
# ---------------------------------------------------------------------
class ChatArgs(BaseModel):
    """
    Unified ChatArgs definition, merging fields from both existing versions.
    """
    conversation_id: Optional[str] = None
    pdf_id: Optional[str] = None

    # Consolidated streaming fields
    stream_enabled: bool = False
    stream_chunk_size: int = 20

    # If you need memory/LLM references
    memory_type: str = "sql_buffer_memory"
    llm: Optional[ChatOpenAI] = None

    # Use ResearchMode enum
    research_mode: ResearchMode = ResearchMode.SINGLE

    # Canonical ResearchContext
    research_context: Optional[ResearchContext] = None
    research_manager: Optional[Any] = None
    retrieval_manager: Optional[Any] = None

    # Additional fields from older versions
    metadata: Optional[Metadata] = None
    technical_context: TechnicalMetadata = Field(default_factory=TechnicalMetadata)
    processing_config: Optional[ProcessingConfig] = None
    content_filters: Dict[str, Any] = Field(default_factory=lambda: {
        "content_types": None,
        "doc_type": None,
        "component_type": None,
        "min_confidence": 0.6
    })

    class Config:
        arbitrary_types_allowed = True

    # ----------------------------------
    # Validators to auto-create context
    # ----------------------------------
    @validator('research_context', pre=True, always=True)
    def ensure_research_context(cls, v, values):
        if v is None and values.get('pdf_id'):
            return ResearchContext(primary_pdf_id=values['pdf_id'])
        return v

    @validator('research_manager', pre=True, always=True)
    def ensure_research_manager(cls, v, values):
        if v is None and values.get('pdf_id'):
            from app.chat.research.research_manager import ResearchManager
            return ResearchManager(primary_pdf_id=values['pdf_id'])
        return v

    @validator('metadata', pre=True, always=True)
    def ensure_metadata(cls, v, values):
        if v is None and values.get('pdf_id') and values.get('conversation_id'):
            return Metadata(
                conversation_id=str(values['conversation_id']),
                user_id="",
                pdf_id=str(values['pdf_id'])
            )
        return v

    # ----------------------------------
    # Example utility methods
    # ----------------------------------
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration with safe metadata handling."""
        metadata_dict = {}
        if self.metadata:
            try:
                if hasattr(self.metadata, 'dict'):
                    metadata_dict = self.metadata.dict()
                elif hasattr(self.metadata, '__dict__'):
                    metadata_dict = self.metadata.__dict__
            except Exception:
                metadata_dict = {"pdf_id": self.pdf_id}

        return {
            "memory_key": "chat_history",
            "return_messages": True,
            "output_key": "answer",
            "input_key": "question",
            "llm": self.llm,
            "metadata": {
                "pdf_id": self.pdf_id,
                "research_mode": self.research_mode.value,
                "technical_context": (
                    self.technical_context.dict()
                    if hasattr(self.technical_context, 'dict') else {}
                )
            }
        }

    def is_research_active(self) -> bool:
        """
        Checks if multi-document research is active.
        """
        if self.research_mode != ResearchMode.RESEARCH:
            return False
        if not self.research_context or not self.research_context.active_pdf_ids:
            return False
        return len(self.research_context.active_pdf_ids) > 1


__all__ = [
    "ChatArgs",
    "ResearchContext",
    "Metadata",
    "ConceptMetadata",
    "TechnicalMetadata"
]

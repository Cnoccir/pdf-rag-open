import sys
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Set, Callable, TYPE_CHECKING
from datetime import datetime
from langchain_openai import ChatOpenAI
from app.chat.types import (
    ResearchContext,
    ResearchMode,
    ProcessingConfig,
    ContentType,
    ConceptNetwork
)

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from app.chat.chains.retrieval import RetrievalManager
    from app.chat.research.research_manager import ResearchManager
else:
    # Runtime aliasing
    RetrievalManager = Any
    # For runtime we'll import later when needed

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

class ChatArgs(BaseModel):
    conversation_id: str
    pdf_id: str
    streaming: bool = False
    metadata: Optional[Metadata] = None
    memory_type: str = "sql_buffer_memory"
    llm: Optional[ChatOpenAI] = None

    research_mode: ResearchMode = ResearchMode.SINGLE
    research_context: Optional[ResearchContext] = None
    research_manager: Optional[Any] = None  # Using Any for type here to avoid import
    retrieval_manager: Optional[Any] = None

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

    # Validators
    @validator('research_context', pre=True, always=True)
    def ensure_research_context(cls, v, values):
        if v is None and 'pdf_id' in values:
            pdf_id = str(values['pdf_id'])
            return ResearchContext(primary_pdf_id=pdf_id)
        return v

    @validator('research_manager', pre=True, always=True)
    def ensure_research_manager(cls, v, values):
        if v is None and 'pdf_id' in values:
            # Import here to avoid circular imports
            from app.chat.research.research_manager import ResearchManager
            pdf_id = str(values['pdf_id'])
            return ResearchManager(primary_pdf_id=pdf_id)
        return v

    @validator('metadata', pre=True, always=True)
    def ensure_metadata(cls, v, values):
        if v is None and 'pdf_id' in values and 'conversation_id' in values:
            pdf_id = str(values['pdf_id'])
            conversation_id = values['conversation_id']
            return Metadata(
                conversation_id=conversation_id,
                user_id="",
                pdf_id=pdf_id
            )
        return v

    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration with safe metadata handling"""
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
                "technical_context": self.technical_context.dict() if hasattr(self.technical_context, 'dict') else {}
            }
        }

    def with_concept_network(self, network: ConceptNetwork) -> 'ChatArgs':
        """Add concept network to metadata with proper initialization"""
        if not self.metadata:
            self.metadata = Metadata(
                conversation_id=self.conversation_id,
                user_id="",
                pdf_id=self.pdf_id
            )
        self.metadata.concept_network = network
        return self

    def get_technical_state(self) -> Dict[str, Any]:
        """Get technical state with safe handling of attributes"""
        result = {
            "technical_context": {},
            "concept_network": None,
            "processing_config": {},
            "last_updated": datetime.utcnow().isoformat()
        }

        # Safely add technical context
        if hasattr(self, 'technical_context') and self.technical_context:
            if hasattr(self.technical_context, 'dict'):
                result["technical_context"] = self.technical_context.dict()

        # Safely add concept network
        if self.metadata and hasattr(self.metadata, 'concept_network') and self.metadata.concept_network:
            if hasattr(self.metadata.concept_network, 'dict'):
                result["concept_network"] = self.metadata.concept_network.dict()

        # Safely add processing config
        if self.processing_config:
            if hasattr(self.processing_config, 'dict'):
                result["processing_config"] = self.processing_config.dict()

        return result

    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research context for API responses with enhanced error handling"""
        # First check metadata.research_mode
        if hasattr(self, 'metadata') and self.metadata:
            if hasattr(self.metadata, 'research_mode'):
                rm = self.metadata.research_mode
                if isinstance(rm, dict):
                    return {
                        "active": rm.get("active", False),
                        "pdf_ids": rm.get("pdf_ids", [self.pdf_id]),
                        "primary_pdf_id": self.pdf_id,
                        "last_updated": datetime.utcnow().isoformat()
                    }

        # Fall back to research_context check
        if not hasattr(self, 'research_context') or not self.research_context:
            return {
                "active": False,
                "pdf_ids": [self.pdf_id]
            }

        # Get active PDF IDs safely
        active_pdf_ids = []
        if hasattr(self.research_context, 'active_pdf_ids'):
            try:
                # Handle both set and list formats
                if isinstance(self.research_context.active_pdf_ids, set):
                    active_pdf_ids = list(self.research_context.active_pdf_ids)
                elif isinstance(self.research_context.active_pdf_ids, list):
                    active_pdf_ids = self.research_context.active_pdf_ids
                else:
                    # Try to convert to list as a fallback
                    active_pdf_ids = list(self.research_context.active_pdf_ids)
            except Exception:
                active_pdf_ids = [self.pdf_id]

        # Get primary PDF ID safely
        primary_pdf_id = self.pdf_id  # Default to current PDF ID
        if hasattr(self.research_context, 'primary_pdf_id'):
            primary_pdf_id = self.research_context.primary_pdf_id

        # Get last updated timestamp safely
        last_updated = datetime.utcnow().isoformat()
        if hasattr(self.research_context, 'last_updated'):
            try:
                if isinstance(self.research_context.last_updated, datetime):
                    last_updated = self.research_context.last_updated.isoformat()
                elif isinstance(self.research_context.last_updated, str):
                    last_updated = self.research_context.last_updated
            except Exception:
                pass  # Keep default value

        return {
            "active": self.research_mode == ResearchMode.MULTI,
            "pdf_ids": active_pdf_ids,
            "primary_pdf_id": primary_pdf_id,
            "last_updated": last_updated
        }

    @property
    def is_research_active(self) -> bool:
        """Safely check if research mode is active with proper error handling"""
        try:
            # First check the research_mode enum
            if self.research_mode != ResearchMode.MULTI:
                return False

            # Then check the research context
            if not hasattr(self, 'research_context') or self.research_context is None:
                return False

            # Check for active_pdf_ids attribute and content
            if not hasattr(self.research_context, 'active_pdf_ids'):
                return False

            # Check the length of active PDFs
            return len(self.research_context.active_pdf_ids) > 1
        except Exception:
            # Any error means research is not active
            return False

__all__ = ["ChatArgs", "Metadata", "TechnicalMetadata", "ConceptMetadata"]

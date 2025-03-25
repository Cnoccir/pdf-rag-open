"""
Type definitions for the Docling n8n API integration.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    EQUATION = "equation"
    DIAGRAM = "diagram"
    PAGE = "page"
    PROCEDURE = "procedure"
    PARAMETER = "parameter"


class ChunkLevel(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PROCEDURE = "procedure"
    STEP = "step"


class EmbeddingType(str, Enum):
    CONCEPTUAL = "conceptual"
    TASK = "task"
    TECHNICAL = "technical"
    GENERAL = "general"


class ProcessingConfig(BaseModel):
    """Configuration options for document processing."""
    pdf_id: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    process_images: bool = True
    process_tables: bool = True
    extract_technical_terms: bool = True
    extract_relationships: bool = True
    extract_procedures: bool = True
    merge_list_items: bool = True
    max_concepts_per_document: int = 200


class ProcessingResponse(BaseModel):
    """Response model for document processing."""
    pdf_id: str
    markdown: str
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    technical_terms: List[str] = Field(default_factory=list)
    procedures: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    concept_relationships: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float = 0.0

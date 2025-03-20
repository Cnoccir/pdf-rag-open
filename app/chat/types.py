"""
Streamlined type definitions for the PDF RAG system.
Optimized for MongoDB document structure and Qdrant vector storage.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime
from pydantic import BaseModel, Field

# -----------------------
# Core Enums
# -----------------------
class ContentType(str, Enum):
    """Content types in documents."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    EQUATION = "equation"
    DIAGRAM = "diagram"
    PAGE = "page"
    PROCEDURE = "procedure"  # Added for procedure extraction
    PARAMETER = "parameter"  # Added for parameter extraction

class ChunkLevel(str, Enum):
    """Hierarchical chunk levels for multi-level chunking."""
    DOCUMENT = "document"   # L1: Document-level (2000-4000 tokens)
    SECTION = "section"     # L2: Section-level (1000-2000 tokens)
    PROCEDURE = "procedure" # L3: Procedure-level (500-1000 tokens)
    STEP = "step"           # L4: Step-level (100-300 tokens)

class EmbeddingType(str, Enum):
    """Embedding types for multi-embedding strategy."""
    CONCEPTUAL = "conceptual"      # Document-level conceptual understanding
    TASK = "task"                  # Procedure-level task understanding
    TECHNICAL = "technical"        # Parameter/technical details
    GENERAL = "general"            # Default embedding type

class ResearchMode(str, Enum):
    """Research modes for the PDF RAG system."""
    SINGLE = "single"     # Single document mode
    RESEARCH = "research"  # Multi-document research mode

class RelationType(str, Enum):
    """Relationship types between technical concepts."""
    PART_OF = "part_of"           # Component is part of a larger system
    USES = "uses"                 # One component uses/depends on another
    IMPLEMENTS = "implements"     # Component implements an interface/abstract concept
    EXTENDS = "extends"           # Component extends/inherits from another
    RELATES_TO = "relates_to"     # General relationship between components
    CONFIGURES = "configures"     # One component configures another
    PREREQUISITE = "prerequisite" # One step/procedure requires another first
    REFERENCES = "references"     # One document refers to another

    @classmethod
    def map_type(cls, type_str: str) -> 'RelationType':
        """Map a string to a RelationType."""
        mapping = {
            "part_of": cls.PART_OF,
            "uses": cls.USES,
            "implements": cls.IMPLEMENTS,
            "extends": cls.EXTENDS,
            "relates_to": cls.RELATES_TO,
            "configures": cls.CONFIGURES,
            "prerequisite": cls.PREREQUISITE,
            "references": cls.REFERENCES,
            "is_a": cls.EXTENDS,
            "has_a": cls.PART_OF,
            "contains": cls.PART_OF
        }
        return mapping.get(type_str.lower(), cls.RELATES_TO)

# -----------------------
# Content Element Models
# -----------------------
class ContentMetadata(BaseModel):
    """Metadata for content elements."""
    pdf_id: str
    page_number: Optional[int] = 0
    content_type: Optional[Union[ContentType, str]] = ContentType.TEXT
    parent_element_id: Optional[str] = None  # Changed from parent_element to parent_element_id
    hierarchy_level: Optional[int] = 0
    section_headers: List[str] = Field(default_factory=list)
    technical_terms: List[str] = Field(default_factory=list)
    chunk_level: Optional[ChunkLevel] = None  # Added for multi-level chunking
    embedding_type: Optional[EmbeddingType] = EmbeddingType.GENERAL  # Added for multi-embedding
    confidence: float = 1.0
    context: Optional[str] = None  # Changed from surrounding_context to context
    image_path: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None
    image_metadata: Optional[Dict[str, Any]] = None
    procedure_metadata: Optional[Dict[str, Any]] = None  # Added for procedure extraction
    parameter_metadata: Optional[Dict[str, Any]] = None  # Added for parameter extraction
    element_id: Optional[str] = None  # Added to help with MongoDB queries
    mongo_id: Optional[str] = None  # Added to reference MongoDB _id
    qdrant_id: Optional[str] = None  # Added to reference Qdrant point ID
    score: float = 0.0
    doc_hash: Optional[str] = None  # Added document hash for versioning

class ContentElement(BaseModel):
    """Content element from a document."""
    element_id: str
    content: str
    content_type: Union[ContentType, str]
    pdf_id: str
    page: Optional[int] = 0
    metadata: ContentMetadata
    embedding: Optional[List[float]] = None  # Added to optionally store embedding
    vector_id: Optional[str] = None  # Added to reference vector ID in Qdrant
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        # Create a dictionary representation
        element_dict = self.dict(exclude={"embedding"})  # Exclude embedding from MongoDB

        # Ensure content_type is a string
        element_dict["content_type"] = str(self.content_type)

        # Add timestamps
        if not element_dict.get("created_at"):
            element_dict["created_at"] = datetime.utcnow()
        element_dict["updated_at"] = datetime.utcnow()

        return element_dict

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to metadata payload for Qdrant."""
        # Create a minimal metadata payload for Qdrant
        payload = {
            "element_id": self.element_id,
            "pdf_id": self.pdf_id,
            "content_type": str(self.content_type),
            "page_number": self.metadata.page_number if hasattr(self.metadata, "page_number") else 0,
            "chunk_level": str(self.metadata.chunk_level) if hasattr(self.metadata, "chunk_level") and self.metadata.chunk_level else None,
            "section": " > ".join(self.metadata.section_headers) if hasattr(self.metadata, "section_headers") and self.metadata.section_headers else None,
            "technical_terms": self.metadata.technical_terms[:10] if hasattr(self.metadata, "technical_terms") and self.metadata.technical_terms else [],
            "hierarchy_level": self.metadata.hierarchy_level if hasattr(self.metadata, "hierarchy_level") else 0
        }
        return payload

# -----------------------
# Chunking Models
# -----------------------
class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    pdf_id: str
    content_type: str
    chunk_level: ChunkLevel
    chunk_index: int
    page_numbers: List[int] = Field(default_factory=list)
    section_headers: List[str] = Field(default_factory=list)
    parent_chunk_id: Optional[str] = None
    technical_terms: List[str] = Field(default_factory=list)
    embedding_type: EmbeddingType = EmbeddingType.GENERAL
    element_ids: List[str] = Field(default_factory=list)  # IDs of elements in this chunk
    token_count: int = 0

class DocumentChunk(BaseModel):
    """Document chunk for vector storage."""
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    vector_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        chunk_dict = self.dict(exclude={"embedding"})
        chunk_dict["content_type"] = str(self.metadata.content_type)
        chunk_dict["chunk_level"] = str(self.metadata.chunk_level)
        chunk_dict["embedding_type"] = str(self.metadata.embedding_type)
        chunk_dict["updated_at"] = datetime.utcnow()
        return chunk_dict

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to metadata payload for Qdrant."""
        payload = {
            "chunk_id": self.chunk_id,
            "pdf_id": self.metadata.pdf_id,
            "content_type": str(self.metadata.content_type),
            "chunk_level": str(self.metadata.chunk_level),
            "embedding_type": str(self.metadata.embedding_type),
            "page_numbers": self.metadata.page_numbers,
            "section": " > ".join(self.metadata.section_headers) if self.metadata.section_headers else None,
            "technical_terms": self.metadata.technical_terms[:10],
            "token_count": self.metadata.token_count
        }
        return payload

# -----------------------
# Concept Models
# -----------------------
class Concept(BaseModel):
    """Technical concept extracted from document."""
    name: str
    occurrences: int = 1
    in_headers: bool = False
    sections: List[str] = Field(default_factory=list)
    first_occurrence_page: Optional[int] = None
    importance_score: float = 0.5
    is_primary: bool = False
    category: Optional[str] = None
    pdf_id: str  # Added to help with MongoDB queries

class ConceptRelationship(BaseModel):
    """Relationship between technical concepts."""
    source: str
    target: str
    type: RelationType = RelationType.RELATES_TO
    weight: float = 0.5
    context: str = ""
    extraction_method: str = "rule-based"
    pdf_id: str  # Added to help with MongoDB queries

class ConceptNetwork(BaseModel):
    """Network of concepts and their relationships."""
    concepts: List[Concept] = Field(default_factory=list)
    relationships: List[ConceptRelationship] = Field(default_factory=list)
    section_concepts: Dict[str, List[str]] = Field(default_factory=dict)
    primary_concepts: List[str] = Field(default_factory=list)
    pdf_id: Optional[str] = None  # Added to help with MongoDB queries

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the network."""
        for existing in self.concepts:
            if existing.name.lower() == concept.name.lower():
                # Update existing concept
                existing.occurrences += concept.occurrences
                existing.importance_score = max(existing.importance_score, concept.importance_score)
                existing.is_primary = existing.is_primary or concept.is_primary
                return
        self.concepts.append(concept)

    def add_relationship(self, relationship: ConceptRelationship) -> None:
        """Add a relationship to the network."""
        for existing in self.relationships:
            if (existing.source.lower() == relationship.source.lower() and
                existing.target.lower() == relationship.target.lower() and
                existing.type == relationship.type):
                # Update existing relationship
                existing.weight = max(existing.weight, relationship.weight)
                return
        self.relationships.append(relationship)

    def add_section_concepts(self, section: str, concepts: List[str]) -> None:
        """Add concepts to a section."""
        if section not in self.section_concepts:
            self.section_concepts[section] = []
        for concept in concepts:
            if concept not in self.section_concepts[section]:
                self.section_concepts[section].append(concept)

    def calculate_importance_scores(self) -> None:
        """Calculate importance scores for concepts based on network properties."""
        # Implementation remains the same
        pass

# -----------------------
# Processing Models
# -----------------------
class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    pdf_id: str
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    process_images: bool = True
    process_tables: bool = True
    extract_technical_terms: bool = True
    extract_relationships: bool = True
    extract_procedures: bool = True  # Added for procedure extraction
    merge_list_items: bool = True
    max_concepts_per_document: int = 200
    # Multi-level chunking configuration
    chunk_levels: Dict[ChunkLevel, int] = Field(
        default_factory=lambda: {
            ChunkLevel.DOCUMENT: 3000,  # 3000 tokens (~2250 words)
            ChunkLevel.SECTION: 1500,   # 1500 tokens (~1125 words)
            ChunkLevel.PROCEDURE: 800,  # 800 tokens (~600 words)
            ChunkLevel.STEP: 200        # 200 tokens (~150 words)
        }
    )
    # Multi-embedding strategy configuration
    embedding_types: Dict[EmbeddingType, str] = Field(
        default_factory=lambda: {
            EmbeddingType.CONCEPTUAL: "text-embedding-3-small",
            EmbeddingType.TASK: "text-embedding-3-small",
            EmbeddingType.TECHNICAL: "text-embedding-3-small",
            EmbeddingType.GENERAL: "text-embedding-3-small"
        }
    )

class ProcessingResult(BaseModel):
    """Result of document processing."""
    pdf_id: str
    elements: List[ContentElement] = Field(default_factory=list)
    chunks: List[DocumentChunk] = Field(default_factory=list)
    processing_metrics: Dict[str, Any] = Field(default_factory=dict)
    markdown_content: str = ""
    markdown_path: str = ""
    concept_network: Optional[ConceptNetwork] = None
    visual_elements: List[ContentElement] = Field(default_factory=list)
    document_summary: Optional[Dict[str, Any]] = None
    procedures: List[Dict[str, Any]] = Field(default_factory=list)  # Added for procedures
    parameters: List[Dict[str, Any]] = Field(default_factory=list)  # Added for parameters
    raw_data: Dict[str, Any] = Field(default_factory=dict)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processing result."""
        element_types = {}
        for element in self.elements:
            element_type = str(element.content_type)
            element_types[element_type] = element_types.get(element_type, 0) + 1

        top_technical_terms = {}
        for element in self.elements:
            if hasattr(element.metadata, 'technical_terms'):
                for term in element.metadata.technical_terms:
                    top_technical_terms[term] = top_technical_terms.get(term, 0) + 1

        return {
            "element_count": len(self.elements),
            "chunk_count": len(self.chunks),
            "element_types": element_types,
            "procedure_count": len(self.procedures),
            "parameter_count": len(self.parameters),
            "top_technical_terms": dict(sorted(top_technical_terms.items(), key=lambda x: x[1], reverse=True)[:20])
        }

# -----------------------
# Chat Management Models
# -----------------------
class ChatArgs(BaseModel):
    """Arguments for chat initialization."""
    conversation_id: Optional[str] = None
    pdf_id: Optional[str] = None
    research_mode: ResearchMode = ResearchMode.SINGLE
    stream_enabled: bool = False
    stream_chunk_size: int = 20
    memory_type: str = "sql"
    metadata: Optional[Dict[str, Any]] = None

    # Additional settings for retrieval
    chunk_level_preference: Optional[ChunkLevel] = None  # Preferred chunk level for retrieval
    embedding_type_preference: Optional[EmbeddingType] = None  # Preferred embedding type

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize research context if not provided
        if not hasattr(self, 'research_context') and hasattr(self, 'pdf_id') and self.pdf_id:
            from app.chat.types import ResearchContext
            self.research_context = ResearchContext(primary_pdf_id=self.pdf_id)

# -----------------------
# Research and Cross-Document Models
# -----------------------
class ResearchContext(BaseModel):
    """Research context for cross-document analysis."""
    primary_pdf_id: str
    active_pdf_ids: Set[str] = Field(default_factory=set)
    document_titles: Dict[str, str] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure primary PDF ID is in active PDF IDs
        if self.primary_pdf_id:
            self.active_pdf_ids.add(self.primary_pdf_id)

    def add_document(self, pdf_id: str, title: Optional[str] = None) -> None:
        """Add a document to research."""
        self.active_pdf_ids.add(pdf_id)
        if title:
            self.document_titles[pdf_id] = title

    def remove_document(self, pdf_id: str) -> None:
        """Remove a document from research."""
        if pdf_id != self.primary_pdf_id and pdf_id in self.active_pdf_ids:
            self.active_pdf_ids.remove(pdf_id)
            if pdf_id in self.document_titles:
                del self.document_titles[pdf_id]

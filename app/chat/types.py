"""
Core type definitions for the PDF RAG system.
Focused on LangGraph architecture and Neo4j integration.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Set, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, constr
import uuid
import logging

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
    RESEARCH = "research"  # Alias for MULTI for backward compatibility

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
# Document Content Models
# ------------------------------------------------------------------------
class ContentMetadata(BaseModel):
    """Metadata for a content element."""
    pdf_id: str = ""
    page_number: int = 0
    section: str = ""
    content_type: str = "text"
    section_headers: List[str] = Field(default_factory=list)
    hierarchy_level: int = 0
    technical_terms: List[str] = Field(default_factory=list)
    surrounding_context: Optional[str] = None
    parent_element: Optional[str] = None
    docling_ref: Optional[str] = None
    confidence: float = 1.0
    image_path: Optional[str] = None
    image_metadata: Optional[Any] = None
    table_data: Optional[Any] = None

class ContentElement(BaseModel):
    """A content element from a document."""
    element_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    content_type: ContentType = ContentType.TEXT
    pdf_id: str
    metadata: ContentMetadata = Field(default_factory=ContentMetadata)

    class Config:
        arbitrary_types_allowed = True

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

# ------------------------------------------------------------------------
# Image Models
# ------------------------------------------------------------------------
class ImagePaths(BaseModel):
    """Model for image file paths."""
    original: str
    format: str = "PNG"
    size: int = 0
    processed: Optional[str] = None
    thumbnail: Optional[str] = None

class ImageFeatures(BaseModel):
    """Model for image features extracted from analysis."""
    dimensions: tuple = (0, 0)
    aspect_ratio: float = 1.0
    color_mode: str = "RGB"
    is_grayscale: bool = False
    embedding: List[float] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class ImageAnalysis(BaseModel):
    """Model for image analysis results."""
    description: str = ""
    detected_objects: List[str] = Field(default_factory=list)
    technical_details: Dict[str, Any] = Field(default_factory=dict)
    technical_concepts: List[str] = Field(default_factory=list)

class ImageMetadata(BaseModel):
    """Model for image metadata."""
    image_id: str
    paths: ImagePaths
    features: ImageFeatures
    analysis: ImageAnalysis
    page_number: int = 0

# ------------------------------------------------------------------------
# Concept Network Models
# ------------------------------------------------------------------------
class Concept(BaseModel):
    """Model for a concept in the concept network."""
    name: str
    occurrences: int = 0
    in_headers: bool = False
    sections: List[str] = Field(default_factory=list)
    first_occurrence_page: Optional[int] = None
    importance_score: float = 0.5
    is_primary: bool = False
    category: Optional[str] = None
    description: Optional[str] = None

class DocumentSummary(BaseModel):
    """Summary of a document with key information."""
    title: str = "Untitled Document"
    description: str = ""
    main_topics: List[str] = Field(default_factory=list)
    key_takeaways: List[str] = Field(default_factory=list)
    summary_text: str = ""
    category: Optional[str] = None
    estimated_reading_time: Optional[int] = None

class ConceptRelationship(BaseModel):
    """Model for a relationship between concepts."""
    source: str
    target: str
    type: RelationType = RelationType.RELATES_TO
    weight: float = 0.5
    context: str = ""
    extraction_method: str = "document-based"

class ConceptNetwork(BaseModel):
    """Model for a network of concepts and their relationships."""
    concepts: List[Concept] = Field(default_factory=list)
    relationships: List[ConceptRelationship] = Field(default_factory=list)
    primary_concepts: List[str] = Field(default_factory=list)
    section_concepts: Dict[str, List[str]] = Field(default_factory=dict)
    
    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the network."""
        self.concepts.append(concept)
    
    def add_relationship(self, relationship: ConceptRelationship) -> None:
        """Add a relationship to the network."""
        self.relationships.append(relationship)
        
    def calculate_importance_scores(self) -> None:
        """Calculate importance scores and identify primary concepts."""
        # Implement network centrality calculation here
        # For now, use a simple approach based on relationship count
        concept_connections = {}
        
        # Count connections for each concept
        for rel in self.relationships:
            concept_connections[rel.source] = concept_connections.get(rel.source, 0) + 1
            concept_connections[rel.target] = concept_connections.get(rel.target, 0) + 1
        
        # Update importance scores
        for concept in self.concepts:
            # Base score from occurrences
            base_score = min(0.7, (concept.occurrences or 0) * 0.1)
            # Connection score
            conn_score = min(0.3, (concept_connections.get(concept.name, 0) * 0.05))
            # Header score
            header_score = 0.2 if concept.in_headers else 0
            
            # Combined score
            concept.importance_score = base_score + conn_score + header_score
            concept.is_primary = concept.importance_score > 0.7
        
        # Sort concepts by importance score and update primary_concepts list
        sorted_concepts = sorted(
            self.concepts, 
            key=lambda c: c.importance_score, 
            reverse=True
        )
        
        # Take top N concepts as primary
        self.primary_concepts = [c.name for c in sorted_concepts[:10] if c.importance_score > 0.6]
    
    def add_section_concepts(self, section: str, concepts: List[str]) -> None:
        """Add concepts to a section."""
        if section not in self.section_concepts:
            self.section_concepts[section] = []
        
        for concept in concepts:
            if concept not in self.section_concepts[section]:
                self.section_concepts[section].append(concept)
    
    def build_section_concept_map(self) -> None:
        """Build section to concept mapping."""
        # This is a placeholder for more sophisticated section mapping
        # For now, we're using the simpler add_section_concepts method
        pass

# ------------------------------------------------------------------------
# Processing Models
# ------------------------------------------------------------------------
class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    pdf_id: constr(min_length=1)
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    process_images: bool = True
    process_tables: bool = True
    extract_technical_terms: bool = True
    extract_relationships: bool = True
    merge_list_items: bool = True
    max_concepts_per_document: int = 200
    search_limit: int = 5

class ProcessingResult(BaseModel):
    """Result of document processing."""
    pdf_id: str
    elements: List[ContentElement] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    markdown_content: str = ""
    markdown_path: str = ""
    processing_metrics: Dict[str, Any] = Field(default_factory=dict)
    concept_network: ConceptNetwork = Field(default_factory=ConceptNetwork)
    visual_elements: List[ContentElement] = Field(default_factory=list)
    document_summary: DocumentSummary = Field(default_factory=DocumentSummary)
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processing result."""
        stats = {
            "element_count": len(self.elements),
            "chunk_count": len(self.chunks),
            "concept_count": len(self.concept_network.concepts),
            "relationship_count": len(self.concept_network.relationships),
            "element_types": {},
            "top_technical_terms": {},
        }
        
        # Count element types
        for element in self.elements:
            element_type = element.content_type.value if hasattr(element.content_type, 'value') else str(element.content_type)
            stats["element_types"][element_type] = stats["element_types"].get(element_type, 0) + 1
        
        # Extract top technical terms
        term_counts = {}
        for element in self.elements:
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'technical_terms'):
                for term in element.metadata.technical_terms:
                    term_counts[term] = term_counts.get(term, 0) + 1
        
        # Sort by count
        stats["top_technical_terms"] = {
            term: count for term, count in 
            sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        }
        
        return stats

# ------------------------------------------------------------------------
# Search Models
# ------------------------------------------------------------------------
class SearchQuery(BaseModel):
    """Query for searching documents."""
    query: str
    pdf_ids: List[str] = Field(default_factory=list)
    active_pdf_ids: Optional[List[str]] = None
    content_types: List[str] = Field(default_factory=list)
    technical_terms: List[str] = Field(default_factory=list)
    strategy: str = "hybrid"
    max_results: int = 10
    research_mode: bool = False
    favor_visual: bool = False
    favor_tables: bool = False
    favor_code: bool = False
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    concepts: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

# ------------------------------------------------------------------------
# Research Context
# ------------------------------------------------------------------------
class ResearchContext(BaseModel):
    """Research context for the conversation."""
    primary_pdf_id: str
    active_pdf_ids: Set[str] = Field(default_factory=set)
    
    def add_pdf_id(self, pdf_id: str) -> None:
        """Add a PDF ID to the active set."""
        self.active_pdf_ids.add(pdf_id)
    
    def remove_pdf_id(self, pdf_id: str) -> None:
        """Remove a PDF ID from the active set."""
        if pdf_id in self.active_pdf_ids:
            self.active_pdf_ids.remove(pdf_id)
    
    @property
    def is_multi_document(self) -> bool:
        """Whether multiple documents are active."""
        return len(self.active_pdf_ids) > 1

# ------------------------------------------------------------------------
# Chat Args (for ChatManager configuration)
# ------------------------------------------------------------------------
class ChatArgs:
    """Arguments for chat configuration"""
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        pdf_id: Optional[str] = None,
        research_mode: ResearchMode = ResearchMode.SINGLE,
        stream_enabled: bool = False,
        stream_chunk_size: int = 10,
    ):
        self.conversation_id = conversation_id
        self.pdf_id = pdf_id
        self.research_mode = research_mode
        self.stream_enabled = stream_enabled
        self.stream_chunk_size = stream_chunk_size


# ------------------------------------------------------------------------
# Research Manager (forward declaration only)
# ------------------------------------------------------------------------
class ResearchManager:
    """Forward declaration for the Research Manager class."""
    def __init__(self, primary_pdf_id: str):
        self.primary_pdf_id = primary_pdf_id

    def register_concept_network(self, pdf_id: str, concept_network: ConceptNetwork) -> None:
        """Register a concept network for a document."""
        pass
        
    def register_shared_concept(self, concept: str, pdf_ids: Set[str], confidence: float) -> None:
        """Register a shared concept across documents."""
        pass
        
    def add_document_metadata(self, pdf_id: str, metadata: Dict[str, Any]) -> None:
        """Add metadata for a document."""
        pass
"""
Core type definitions for the PDF RAG system.
This module contains all shared type definitions used across the application.
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
    parent_element: Optional[str] = None
    hierarchy_level: Optional[int] = 0
    section_headers: List[str] = Field(default_factory=list)
    technical_terms: List[str] = Field(default_factory=list)
    docling_ref: Optional[str] = None
    confidence: float = 1.0
    surrounding_context: Optional[str] = None
    image_path: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None
    image_metadata: Optional[Dict[str, Any]] = None
    score: float = 0.0

class ContentElement(BaseModel):
    """Content element from a document."""
    element_id: str
    content: str
    content_type: Union[ContentType, str]
    pdf_id: str
    page: Optional[int] = 0
    metadata: ContentMetadata

# -----------------------
# Image Analysis Models
# -----------------------
class ImageFeatures(BaseModel):
    """Features extracted from an image."""
    dimensions: tuple = (0, 0)
    aspect_ratio: float = 1.0
    color_mode: str = "RGB"
    is_grayscale: bool = False

class ImageAnalysis(BaseModel):
    """Analysis results for an image."""
    description: str = ""
    detected_objects: List[str] = Field(default_factory=list)
    technical_details: Dict[str, Any] = Field(default_factory=dict)
    technical_concepts: List[str] = Field(default_factory=list)

class ImagePaths(BaseModel):
    """Paths to image files."""
    original: str
    format: str = "PNG"
    size: int = 0

class ImageMetadata(BaseModel):
    """Full metadata for an image."""
    image_id: str
    paths: ImagePaths
    features: ImageFeatures
    analysis: ImageAnalysis
    page_number: int = 0

# -----------------------
# Table Data Model
# -----------------------
class TableData(BaseModel):
    """Data extracted from a table."""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    caption: str = ""
    markdown: str = ""
    summary: str = ""
    row_count: int = 0
    column_count: int = 0
    technical_concepts: List[str] = Field(default_factory=list)

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

class ConceptRelationship(BaseModel):
    """Relationship between technical concepts."""
    source: str
    target: str
    type: RelationType = RelationType.RELATES_TO
    weight: float = 0.5
    context: str = ""
    extraction_method: str = "rule-based"

class ConceptNetwork(BaseModel):
    """Network of concepts and their relationships."""
    concepts: List[Concept] = Field(default_factory=list)
    relationships: List[ConceptRelationship] = Field(default_factory=list)
    section_concepts: Dict[str, List[str]] = Field(default_factory=dict)
    primary_concepts: List[str] = Field(default_factory=list)

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
        # Count relationships for each concept
        relationship_counts = {}
        for rel in self.relationships:
            relationship_counts[rel.source] = relationship_counts.get(rel.source, 0) + 1
            relationship_counts[rel.target] = relationship_counts.get(rel.target, 0) + 1

        # Update importance scores and identify primary concepts
        primary_concepts = []
        for concept in self.concepts:
            # Base score from occurrences with diminishing returns
            occurrence_score = min(0.5, 0.1 * concept.occurrences)

            # Bonus for being in headers
            header_bonus = 0.2 if concept.in_headers else 0

            # Bonus for relationship count
            relationship_bonus = min(0.3, 0.05 * relationship_counts.get(concept.name, 0))

            # Total score
            concept.importance_score = occurrence_score + header_bonus + relationship_bonus

            # Mark as primary if score is high enough
            concept.is_primary = concept.importance_score > 0.7

            if concept.is_primary:
                primary_concepts.append(concept.name)

        # Update primary concepts list
        self.primary_concepts = primary_concepts

    def build_section_concept_map(self) -> None:
        """Build a mapping of sections to concepts."""
        # This method can be implemented if needed
        pass

# -----------------------
# Research Models
# -----------------------
class ResearchContext(BaseModel):
    """Research context for multi-document analysis."""
    query: str
    summary: str = ""
    facts: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    cross_references: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class ResearchManager(BaseModel):
    """Manager for research-mode document interactions."""
    primary_pdf_id: str
    active_pdf_ids: Set[str] = Field(default_factory=set)
    document_names: Dict[str, str] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure primary PDF ID is in active PDF IDs
        if self.primary_pdf_id:
            self.active_pdf_ids.add(self.primary_pdf_id)

    def add_document(self, pdf_id: str, title: Optional[str] = None) -> None:
        """Add a document to research."""
        self.active_pdf_ids.add(pdf_id)
        if title:
            self.document_names[pdf_id] = title

    def remove_document(self, pdf_id: str) -> None:
        """Remove a document from research."""
        if pdf_id != self.primary_pdf_id and pdf_id in self.active_pdf_ids:
            self.active_pdf_ids.remove(pdf_id)
            if pdf_id in self.document_names:
                del self.document_names[pdf_id]

    def register_concept_network(self, pdf_id: str, network: ConceptNetwork) -> None:
        """Register a concept network with a document."""
        # Implementation can be added if needed
        pass

    def register_shared_concept(self, concept: str, pdf_ids: Set[str], confidence: float = 0.5) -> None:
        """Register a concept shared across multiple documents."""
        # Implementation can be added if needed
        pass

# -----------------------
# Search Models
# -----------------------
class SearchQuery(BaseModel):
    """Query for searching documents."""
    query: str
    active_pdf_ids: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    technical_terms: Optional[List[str]] = None
    max_results: int = 10
    research_mode: bool = False
    favor_visual: bool = False
    favor_tables: bool = False
    favor_code: bool = False

# -----------------------
# Processing Models
# -----------------------
class ProcessingResult(BaseModel):
    """Result of document processing."""
    pdf_id: str
    elements: List[ContentElement] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    processing_metrics: Dict[str, Any] = Field(default_factory=dict)
    markdown_content: str = ""
    markdown_path: str = ""
    concept_network: Optional[ConceptNetwork] = None
    visual_elements: List[ContentElement] = Field(default_factory=list)
    document_summary: Optional[Dict[str, Any]] = None
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
            "top_technical_terms": dict(sorted(top_technical_terms.items(), key=lambda x: x[1], reverse=True)[:20])
        }

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
    merge_list_items: bool = True
    max_concepts_per_document: int = 200

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

    # Fields to be initialized later
    research_context: Optional[ResearchContext] = None
    research_manager: Optional[ResearchManager] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize research manager if not provided
        if not self.research_manager and self.pdf_id:
            self.research_manager = ResearchManager(primary_pdf_id=self.pdf_id)

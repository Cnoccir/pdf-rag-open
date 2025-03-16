"""
Document manager for research mode.
Handles cross-document relationships and insights.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentReference(BaseModel):
    """Reference to a document with metadata."""
    document_id: str
    title: str
    relevance_score: float = 0.0
    access_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentRelationship(BaseModel):
    """Relationship between documents."""
    source_id: str
    target_id: str
    relationship_type: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentInsight(BaseModel):
    """Insight derived from cross-document analysis."""
    insight_id: str
    description: str
    documents: List[str]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentManager:
    """
    Manager for cross-document relationships and insights.
    
    Features:
    - Track relationships between documents
    - Generate cross-document insights
    - Recommend related documents
    - Support research mode functionality
    """
    
    def __init__(self, primary_pdf_id: Optional[str] = None):
        """
        Initialize document manager.
        
        Args:
            primary_pdf_id: Optional primary PDF ID
        """
        self.primary_pdf_id = primary_pdf_id
        self.documents = {}  # document_id -> DocumentReference
        self.relationships = []  # List of DocumentRelationship
        self.insights = []  # List of DocumentInsight
        
        logger.info(f"Initialized DocumentManager with primary_pdf_id={primary_pdf_id}")
    
    def add_document(
        self,
        document_id: str,
        title: str,
        relevance_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document to the manager.
        
        Args:
            document_id: Document ID
            title: Document title
            relevance_score: Relevance score
            metadata: Optional metadata
        """
        self.documents[document_id] = DocumentReference(
            document_id=document_id,
            title=title,
            relevance_score=relevance_score,
            metadata=metadata or {}
        )
        
        logger.debug(f"Added document to manager: {document_id}")
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relationship between documents.
        
        Args:
            source_id: Source document ID
            target_id: Target document ID
            relationship_type: Type of relationship
            confidence: Confidence in relationship
            metadata: Optional metadata
        """
        # Ensure documents exist
        for doc_id in [source_id, target_id]:
            if doc_id not in self.documents:
                logger.warning(f"Adding relationship for unknown document: {doc_id}")
                self.add_document(doc_id, f"Document {doc_id}", 0.0, {})
        
        # Check if relationship already exists
        for rel in self.relationships:
            if (rel.source_id == source_id and 
                rel.target_id == target_id and 
                rel.relationship_type == relationship_type):
                # Update existing relationship
                rel.confidence = max(rel.confidence, confidence)
                if metadata:
                    rel.metadata.update(metadata)
                rel.created_at = datetime.utcnow()
                logger.debug(f"Updated relationship: {source_id} -> {target_id} ({relationship_type})")
                return
        
        # Add new relationship
        self.relationships.append(DocumentRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            confidence=confidence,
            metadata=metadata or {}
        ))
        
        logger.debug(f"Added relationship: {source_id} -> {target_id} ({relationship_type})")
    
    def add_insight(
        self,
        description: str,
        documents: List[str],
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add an insight from cross-document analysis.
        
        Args:
            description: Insight description
            documents: List of document IDs
            confidence: Confidence in insight
            metadata: Optional metadata
            
        Returns:
            Insight ID
        """
        # Generate insight ID
        insight_id = f"insight_{len(self.insights) + 1}_{int(datetime.utcnow().timestamp())}"
        
        # Ensure documents exist
        for doc_id in documents:
            if doc_id not in self.documents:
                logger.warning(f"Adding insight for unknown document: {doc_id}")
                self.add_document(doc_id, f"Document {doc_id}", 0.0, {})
        
        # Add insight
        self.insights.append(DocumentInsight(
            insight_id=insight_id,
            description=description,
            documents=documents,
            confidence=confidence,
            metadata=metadata or {}
        ))
        
        logger.debug(f"Added insight {insight_id}: {description[:50]}...")
        
        return insight_id
    
    def get_document(self, document_id: str) -> Optional[DocumentReference]:
        """
        Get document reference.
        
        Args:
            document_id: Document ID
            
        Returns:
            DocumentReference or None if not found
        """
        if document_id in self.documents:
            # Update access count
            self.documents[document_id].access_count += 1
            return self.documents[document_id]
        return None
    
    def get_relationships(
        self,
        document_id: str,
        relationship_type: Optional[str] = None,
        min_confidence: float = 0.0,
        as_source: bool = True,
        as_target: bool = False
    ) -> List[DocumentRelationship]:
        """
        Get relationships for a document.
        
        Args:
            document_id: Document ID
            relationship_type: Optional relationship type filter
            min_confidence: Minimum confidence threshold
            as_source: Include relationships where document is source
            as_target: Include relationships where document is target
            
        Returns:
            List of DocumentRelationship objects
        """
        results = []
        
        for rel in self.relationships:
            # Check source/target conditions
            source_match = as_source and rel.source_id == document_id
            target_match = as_target and rel.target_id == document_id
            
            # Check if either condition is met
            if not (source_match or target_match):
                continue
            
            # Check relationship type
            if relationship_type and rel.relationship_type != relationship_type:
                continue
            
            # Check confidence
            if rel.confidence < min_confidence:
                continue
            
            results.append(rel)
        
        return results
    
    def get_related_documents(
        self,
        document_id: str,
        relationship_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get related documents.
        
        Args:
            document_id: Document ID
            relationship_types: Optional list of relationship types
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results
            
        Returns:
            List of related document information
        """
        # Get all relationships for this document as source
        relationships = self.get_relationships(
            document_id,
            relationship_type=None,
            min_confidence=min_confidence,
            as_source=True,
            as_target=False
        )
        
        # Filter by relationship types if specified
        if relationship_types:
            relationships = [r for r in relationships if r.relationship_type in relationship_types]
        
        # Sort by confidence (descending)
        relationships.sort(key=lambda r: r.confidence, reverse=True)
        
        # Build result list
        results = []
        for rel in relationships[:max_results]:
            target_doc = self.get_document(rel.target_id)
            if target_doc:
                results.append({
                    "document_id": target_doc.document_id,
                    "title": target_doc.title,
                    "relationship_type": rel.relationship_type,
                    "confidence": rel.confidence,
                    "relevance_score": target_doc.relevance_score,
                    "metadata": target_doc.metadata
                })
        
        return results
    
    def get_insights(
        self,
        document_id: Optional[str] = None,
        min_confidence: float = 0.0,
        max_results: int = 5
    ) -> List[DocumentInsight]:
        """
        Get insights, optionally filtered by document.
        
        Args:
            document_id: Optional document ID filter
            min_confidence: Minimum confidence threshold
            max_results: Maximum number of results
            
        Returns:
            List of DocumentInsight objects
        """
        # Filter insights
        filtered_insights = self.insights
        
        # Filter by document if specified
        if document_id:
            filtered_insights = [i for i in filtered_insights if document_id in i.documents]
        
        # Filter by confidence
        filtered_insights = [i for i in filtered_insights if i.confidence >= min_confidence]
        
        # Sort by confidence (descending) and then by recency (descending)
        filtered_insights.sort(key=lambda i: (i.confidence, i.created_at), reverse=True)
        
        return filtered_insights[:max_results]
    
    def generate_document_graph(self) -> Dict[str, Any]:
        """
        Generate a graph representation of document relationships.
        
        Returns:
            Dictionary with graph data
        """
        # Build nodes list
        nodes = []
        for doc_id, doc in self.documents.items():
            nodes.append({
                "id": doc_id,
                "label": doc.title,
                "relevance": doc.relevance_score,
                "access_count": doc.access_count,
                "metadata": doc.metadata
            })
        
        # Build edges list
        edges = []
        for rel in self.relationships:
            edges.append({
                "source": rel.source_id,
                "target": rel.target_id,
                "type": rel.relationship_type,
                "confidence": rel.confidence,
                "metadata": rel.metadata
            })
        
        # Build insights list
        insights_list = []
        for insight in self.insights:
            insights_list.append({
                "id": insight.insight_id,
                "description": insight.description,
                "documents": insight.documents,
                "confidence": insight.confidence,
                "created_at": insight.created_at.isoformat()
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "insights": insights_list,
            "primary_document": self.primary_pdf_id,
            "total_documents": len(self.documents),
            "total_relationships": len(self.relationships),
            "total_insights": len(self.insights)
        }

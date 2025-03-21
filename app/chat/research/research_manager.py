"""
Research manager for handling multi-document knowledge retrieval.
Provides capabilities for cross-document search and knowledge synthesis.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union, TYPE_CHECKING
from datetime import datetime

from langchain_core.documents import Document
from pydantic import BaseModel, Field

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from app.chat.models import ChatArgs, Metadata
else:
    # Create placeholders for runtime
    ChatArgs = Any
    Metadata = Any

from app.chat.types import (
    ContentType,
    SearchQuery,
    SearchResult,
    ResearchMode,
    ProcessingConfig,
    ConceptNetwork
)
from app.chat.utils.extraction import extract_technical_terms

logger = logging.getLogger(__name__)


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


class CrossDocumentEvidence(BaseModel):
    """Evidence connecting concepts across documents."""
    concept: str
    documents: List[str]
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    context: Optional[str] = None


class ResearchContext(BaseModel):
    """
    Research context for cross-document insights.

    Maintains:
    - Active document references
    - Cross-document concept relationships
    - Research metadata and statistics
    """
    documents: Dict[str, DocumentReference] = Field(default_factory=dict)
    active_pdf_ids: List[str] = Field(default_factory=list)
    cross_document_evidence: List[CrossDocumentEvidence] = Field(default_factory=list)
    primary_concepts: List[str] = Field(default_factory=list)
    domain_capabilities: List[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    query_count: int = 0
    document_count: int = 0


class ResearchManager:
    """
    Research manager for handling multi-document knowledge retrieval.

    Features:
    - Cross-document search and retrieval
    - Document relationship identification
    - Concept-based knowledge synthesis
    - Research context persistence
    """

    def __init__(self, chat_args: Optional[Any] = None, primary_pdf_id: Optional[str] = None):
        """
        Initialize the research manager.

        Args:
            chat_args: Chat arguments (optional)
            primary_pdf_id: Primary PDF ID (optional)
        """
        self.chat_args = chat_args
        self.primary_pdf_id = primary_pdf_id
        if chat_args and not primary_pdf_id:
            self.primary_pdf_id = getattr(chat_args, "pdf_id", None)

        self.context = ResearchContext()
        self.stores = {}  # PDF ID -> TechnicalDocumentStore
        self.llm = None  # Will be set by the chat manager

        # Concept network
        self.concept_network = None

        logger.info("Initialized ResearchManager")

    def add_document(
        self,
        pdf_id: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        document_type: Optional[str] = None,
        summary: Optional[str] = None,
        primary_concepts: Optional[List[str]] = None,
        source: Optional[str] = None
    ) -> None:
        """
        Add a document to the research context.

        Args:
            pdf_id: Document ID
            title: Document title
            author: Document author
            document_type: Document type
            summary: Document summary
            primary_concepts: Primary concepts in the document
            source: Document source
        """
        # Create document reference
        document_ref = DocumentReference(
            pdf_id=pdf_id,
            title=title,
            author=author,
            document_type=document_type,
            summary=summary,
            primary_concepts=primary_concepts,
            source=source,
            active=True
        )

        # Add to context
        self.context.documents[pdf_id] = document_ref
        if pdf_id not in self.context.active_pdf_ids:
            self.context.active_pdf_ids.append(pdf_id)

        # Update document count
        self.context.document_count = len(self.context.documents)

        # Update primary concepts
        if primary_concepts:
            for concept in primary_concepts:
                if concept not in self.context.primary_concepts:
                    self.context.primary_concepts.append(concept)

        # Update timestamp
        self.context.updated_at = datetime.utcnow()

        logger.info(f"Added document '{title}' (ID: {pdf_id}) to research context")

    def remove_document(self, pdf_id: str) -> None:
        """
        Remove a document from the research context.

        Args:
            pdf_id: Document ID
        """
        # Remove from active PDF IDs
        if pdf_id in self.context.active_pdf_ids:
            self.context.active_pdf_ids.remove(pdf_id)

        # Set document as inactive
        if pdf_id in self.context.documents:
            self.context.documents[pdf_id].active = False

        # Update document count
        self.context.document_count = len([doc for doc in self.context.documents.values() if doc.active])

        # Update timestamp
        self.context.updated_at = datetime.utcnow()

        logger.info(f"Removed document (ID: {pdf_id}) from research context")

    def add_document_store(self, pdf_id: str, store) -> None:
        """
        Add a document store for a document.

        Args:
            pdf_id: Document ID
            store: TechnicalDocumentStore instance
        """
        self.stores[pdf_id] = store
        logger.info(f"Added document store for PDF ID: {pdf_id}")

    def set_active_documents(self, pdf_ids: List[str]) -> None:
        """
        Set active documents for research.

        Args:
            pdf_ids: List of document IDs to activate
        """
        # Validate PDF IDs
        valid_pdf_ids = [pdf_id for pdf_id in pdf_ids if pdf_id in self.context.documents]

        # Update active PDF IDs
        self.context.active_pdf_ids = valid_pdf_ids

        # Update document references
        for pdf_id, doc_ref in self.context.documents.items():
            doc_ref.active = pdf_id in valid_pdf_ids

        # Update timestamp
        self.context.updated_at = datetime.utcnow()

        logger.info(f"Set active documents: {valid_pdf_ids}")

    def get_document_ids(self) -> List[str]:
        """
        Get all document IDs in the research context.

        Returns:
            List of document IDs
        """
        return list(self.context.documents.keys())

    def get_active_document_ids(self) -> List[str]:
        """
        Get active document IDs in the research context.

        Returns:
            List of active document IDs
        """
        return self.context.active_pdf_ids

    def get_document_categories(self) -> List[str]:
        """
        Get document categories in the research context.

        Returns:
            List of document categories
        """
        categories = set()
        for doc_ref in self.context.documents.values():
            if doc_ref.document_type:
                categories.add(doc_ref.document_type)
        return list(categories)

    def get_primary_concepts(self) -> List[str]:
        """
        Get primary concepts across all documents.

        Returns:
            List of primary concepts
        """
        return self.context.primary_concepts

    def get_domain_capabilities(self) -> List[str]:
        """
        Get domain capabilities across all documents.

        Returns:
            List of domain capabilities
        """
        return self.context.domain_capabilities

    def get_section_metadata(self) -> Dict[str, Any]:
        """
        Get section metadata across all documents.

        Returns:
            Dictionary with section metadata
        """
        section_metadata = {
            "hierarchy": [],
            "common_sections": [],
            "unique_sections": {}
        }

        # TODO: Implement section metadata extraction

        return section_metadata

    async def search_across_documents(self, search_query: SearchQuery) -> List[SearchResult]:
        """
        Search across multiple documents.

        Args:
            search_query: Search query

        Returns:
            List of search results
        """
        # Increment query count
        self.context.query_count += 1

        # Update timestamp
        self.context.updated_at = datetime.utcnow()

        # Get active document IDs
        active_pdf_ids = search_query.active_pdf_ids or self.context.active_pdf_ids

        # If no active documents, return empty results
        if not active_pdf_ids:
            logger.warning("No active documents for search")
            return []

        try:
            # Prepare search tasks
            search_tasks = []
            for pdf_id in active_pdf_ids:
                if pdf_id in self.stores:
                    # Clone search query for this document
                    doc_query = SearchQuery(
                        query=search_query.query,
                        content_types=search_query.content_types,
                        technical_terms=search_query.technical_terms,
                        max_results=search_query.max_results,
                        research_mode=True,
                        favor_visual=search_query.favor_visual,
                        favor_tables=search_query.favor_tables,
                        favor_code=search_query.favor_code
                    )

                    # Create search task
                    search_tasks.append(self.stores[pdf_id].search_async(doc_query))

            # Execute searches in parallel
            all_results = await asyncio.gather(*search_tasks)

            # Flatten results
            flattened_results = []
            for i, results in enumerate(all_results):
                pdf_id = active_pdf_ids[i]

                # Add document metadata to each result
                for result in results:
                    # Add document reference
                    result.pdf_id = pdf_id

                    # Add document title if available
                    if pdf_id in self.context.documents:
                        result.document_title = self.context.documents[pdf_id].title

                    flattened_results.append(result)

            # Sort by score
            flattened_results.sort(key=lambda x: x.score if hasattr(x, "score") else 0.0, reverse=True)

            # Limit results
            max_results = search_query.max_results or 10
            flattened_results = flattened_results[:max_results]

            return flattened_results

        except Exception as e:
            logger.error(f"Error in search_across_documents: {str(e)}", exc_info=True)
            return []

    def add_cross_document_evidence(
        self,
        concept: str,
        documents: List[str],
        strength: float = 0.5,
        context: Optional[str] = None
    ) -> None:
        """
        Add cross-document evidence for a concept.

        Args:
            concept: Concept
            documents: List of document IDs
            strength: Evidence strength
            context: Evidence context
        """
        # Create evidence
        evidence = CrossDocumentEvidence(
            concept=concept,
            documents=documents,
            strength=strength,
            context=context
        )

        # Add to context
        self.context.cross_document_evidence.append(evidence)

        # Update timestamp
        self.context.updated_at = datetime.utcnow()

        logger.info(f"Added cross-document evidence for concept: {concept}")

    def get_cross_document_evidence(self, concept: str) -> List[CrossDocumentEvidence]:
        """
        Get cross-document evidence for a concept.

        Args:
            concept: Concept

        Returns:
            List of cross-document evidence
        """
        return [
            evidence for evidence in self.context.cross_document_evidence
            if evidence.concept.lower() == concept.lower()
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert research context to dictionary.

        Returns:
            Dictionary representation of research context
        """
        return {
            "documents": {
                pdf_id: doc_ref.dict()
                for pdf_id, doc_ref in self.context.documents.items()
            },
            "active_pdf_ids": self.context.active_pdf_ids,
            "cross_document_evidence": [
                evidence.dict()
                for evidence in self.context.cross_document_evidence
            ],
            "primary_concepts": self.context.primary_concepts,
            "domain_capabilities": self.context.domain_capabilities,
            "created_at": self.context.created_at.isoformat(),
            "updated_at": self.context.updated_at.isoformat(),
            "query_count": self.context.query_count,
            "document_count": self.context.document_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], chat_args: Any) -> 'ResearchManager':
        """
        Create research manager from dictionary.

        Args:
            data: Dictionary representation of research context
            chat_args: Chat arguments

        Returns:
            ResearchManager instance
        """
        manager = cls(chat_args)

        # Create context
        context = ResearchContext(
            active_pdf_ids=data.get("active_pdf_ids", []),
            primary_concepts=data.get("primary_concepts", []),
            domain_capabilities=data.get("domain_capabilities", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
            query_count=data.get("query_count", 0),
            document_count=data.get("document_count", 0)
        )

        # Add documents
        for pdf_id, doc_data in data.get("documents", {}).items():
            doc_ref = DocumentReference(**doc_data)
            context.documents[pdf_id] = doc_ref

        # Add cross-document evidence
        for evidence_data in data.get("cross_document_evidence", []):
            evidence = CrossDocumentEvidence(**evidence_data)
            context.cross_document_evidence.append(evidence)

        manager.context = context

        return manager

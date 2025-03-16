"""
Enhanced document processor with Neo4j vector store integration.
Handles document extraction, processing, and ingestion for LangGraph RAG system.
"""

import asyncio
import aiofiles
import base64
import io
import json
import logging
import os
import re
import time
import uuid
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Iterator

from openai import AsyncOpenAI

# Docling imports
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    TextItem,
    TableItem,
    PictureItem,
    SectionHeaderItem,
    DocItemLabel,
    ImageRefMode,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.chunking import HybridChunker, BaseChunk

# OCR options imports
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    TesseractOcrOptions,
    TableStructureOptions
)

from langchain_core.documents import Document

# Import from modular utils
from app.chat.utils.extraction import (
    extract_technical_terms,
    extract_document_relationships,
    DOMAIN_SPECIFIC_TERMS,
    ALL_DOMAIN_TERMS
)
from app.chat.utils.tokenization import get_tokenizer
from app.chat.utils.document import (
    generate_document_summary,
    create_directory_if_not_exists
)
from app.chat.utils.processing import (
    normalize_metadata_for_vectorstore,
    normalize_pdf_id
)

from app.chat.types import (
    ContentType,
    ProcessingResult,
    ContentElement,
    ContentMetadata,
    ProcessingConfig,
    ImageMetadata,
    ImageFeatures,
    ImageAnalysis,
    ImagePaths,
    TableData,
    ConceptNetwork,
    ConceptRelationship,
    Concept,
    RelationType
)
from app.chat.errors import ProcessingError

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Enhanced document processor with Neo4j integration.
    Focuses on preserving document structure, domain knowledge integration,
    and hierarchical relationships for technical documentation.
    """

    def __init__(self, pdf_id: str, config: ProcessingConfig, openai_client: AsyncOpenAI, research_manager=None):
        self.pdf_id = pdf_id
        self.config = config
        self.openai_client = openai_client
        self.research_manager = research_manager
        self.processing_start = datetime.utcnow()
        self.metrics = {"timings": defaultdict(float), "counts": defaultdict(int)}
        self.output_dir = Path("output") / self.pdf_id
        self._setup_directories()
        self.markdown_path = self.output_dir / "content" / "document.md"
        self.docling_doc = None
        self.conversion_result = None
        self.concept_network = ConceptNetwork()

        # Track section hierarchy during processing
        self.section_hierarchy = []
        self.section_map = {}  # Maps section titles to their level
        self.element_section_map = {}  # Maps element IDs to their section context

        # Domain-specific counters to detect document type and primary concepts
        self.domain_term_counters = defaultdict(int)

        logger.info(f"Initialized DocumentProcessor for PDF {pdf_id} with config: {config.dict(exclude={'embedding_dimensions'})}")

    async def process_document(self) -> ProcessingResult:
        """
        Process document with Neo4j integration for relationship preservation.
        Preserves document structure and extracts rich metadata.
        Enhanced with document summarization and category detection.
        """
        logger.info(f"Starting enhanced document processing for {self.pdf_id}")
        start_time = time.time()

        try:
            # 1. Download and convert the document
            content = self._download_content()
            if not content:
                raise ProcessingError(f"No content found for PDF {self.pdf_id}")

            # 2. Convert document to Docling format
            logger.info(f"Converting document {self.pdf_id}")
            self.docling_doc = await self._convert_document(content)

            # 3. Extract and save markdown content
            logger.info(f"Exporting document {self.pdf_id} to markdown")
            md_content = await self._extract_markdown(self.docling_doc)
            await self._save_markdown(md_content)

            # 4. Extract content elements with hierarchy preservation
            logger.info(f"Extracting content elements from {self.pdf_id}")
            elements = await self._extract_content_elements(self.docling_doc)

            # 5. Generate optimized chunks using Docling's HybridChunker
            logger.info(f"Generating optimized chunks for {self.pdf_id}")
            chunks = self._generate_chunks(self.docling_doc)

            # 6. Build concept network from document content
            logger.info(f"Building concept network for {self.pdf_id}")
            await self._build_concept_network(elements, chunks)

            # 7. Create processing result with comprehensive metadata
            visual_elements = [e for e in elements if e.content_type == ContentType.IMAGE]

            # 8. Extract all technical terms for document summary
            all_technical_terms = self._extract_all_technical_terms(elements)

            # 9. Generate an enhanced document summary using LLM where possible
            logger.info(f"Generating enhanced document summary for {self.pdf_id}")
            document_summary = await generate_document_summary(
                text=md_content,
                technical_terms=all_technical_terms,
                relationships=self.concept_network.relationships,
                openai_client=self.openai_client
            )

            # 10. Predict document category
            predicted_category = self._predict_document_category(all_technical_terms, md_content)
            logger.info(f"Predicted document category: {predicted_category}")

            # 11. Create processing result
            result = ProcessingResult(
                pdf_id=self.pdf_id,
                elements=elements,
                chunks=chunks,
                processing_metrics=self.metrics,
                markdown_content=md_content,
                markdown_path=str(self.markdown_path),
                concept_network=self.concept_network,
                visual_elements=visual_elements,
                document_summary=document_summary  # Added document summary
            )

            # 12. Store processed content in Neo4j vector store
            logger.info(f"Ingesting content to Neo4j for {self.pdf_id}")
            await self._ingest_to_neo4j(result)

            # 13. Update PDF metadata in database with enhanced summary and description
            await self._update_pdf_metadata(document_summary, predicted_category)

            # 14. Register document metadata with research manager if available
            if self.research_manager:
                document_stats = result.get_statistics()

                # Get description for research context
                description = ""
                if document_summary.get('description'):
                    description = document_summary['description']
                elif document_summary.get('key_insights') and document_summary['key_insights']:
                    description = document_summary['key_insights'][0]

                self.research_manager.add_document_metadata(self.pdf_id, {
                    "total_elements": len(elements),
                    "content_types": document_stats["element_types"],
                    "technical_terms": set(document_stats["top_technical_terms"].keys()),
                    "hierarchies": [" > ".join(s) for s in self.section_hierarchy if s],
                    "concept_network": {
                        "total_concepts": len(self.concept_network.concepts),
                        "total_relationships": len(self.concept_network.relationships),
                        "primary_concepts": self.concept_network.primary_concepts
                    },
                    "top_technical_terms": list(document_stats["top_technical_terms"].keys())[:20],
                    "domain_category": predicted_category,
                    "description": description
                })

            # 15. Save results with optimized storage
            await self._save_results(result)

            # Record total processing time
            self.metrics["timings"]["total"] = time.time() - start_time
            logger.info(f"Completed enhanced processing for {self.pdf_id}")

            return result

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}", exc_info=True)
            # Update PDF record with error
            try:
                await self._update_pdf_error(str(e))
            except Exception as db_error:
                logger.error(f"Failed to update PDF error status: {str(db_error)}")
            raise

    def _download_content(self) -> bytes:
        """Download document content with error handling."""
        try:
            from app.web.files import download_file_content
            content = download_file_content(self.pdf_id)
            if not content:
                raise ProcessingError(f"Failed to download content for {self.pdf_id}")
            return content
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise ProcessingError(f"Content download failed: {str(e)}")

    async def _convert_document(self, content: bytes) -> DoclingDocument:
        """Convert raw PDF to DoclingDocument with image preservation."""
        conversion_start = time.time()

        # Save content to temporary file
        temp_path = self.output_dir / "temp" / f"{self.pdf_id}.pdf"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(content)

        try:
            # Configure with image preservation
            pipeline_options = PdfPipelineOptions()

            # Core options
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            # CRITICAL: These settings preserve images
            pipeline_options.images_scale = 2.0  # Higher resolution
            pipeline_options.generate_page_images = True  # Must be True
            pipeline_options.generate_picture_images = True  # Must be True

            # OCR and table settings
            ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
            pipeline_options.ocr_options = ocr_options
            if pipeline_options.table_structure_options:
                pipeline_options.table_structure_options.do_cell_matching = True

            # Initialize converter
            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )

            # Convert the document
            logger.info(f"Converting document {self.pdf_id} with Docling")
            conversion_result = converter.convert(str(temp_path))
            docling_doc = conversion_result.document

            # Save the conversion result for access to all properties
            self.conversion_result = conversion_result

            # Record metrics
            self.metrics["timings"]["conversion"] = time.time() - conversion_start
            self.metrics["counts"]["pages"] = len(docling_doc.pages) if docling_doc.pages else 0

            logger.info(f"Successfully converted document with {len(docling_doc.pages)} pages")
            return docling_doc

        except Exception as e:
            logger.error(f"Document conversion failed: {e}", exc_info=True)
            raise ProcessingError(f"Document conversion failed: {e}")

    async def _extract_markdown(self, doc: DoclingDocument) -> str:
        """Extract markdown using Docling's export method."""
        try:
            # Use the basic export
            md_content = doc.export_to_markdown()
            return md_content
        except Exception as e:
            logger.warning(f"Basic markdown export failed: {e}")

            # Try alternative approach using generate_multimodal_pages
            try:
                from docling.utils.export import generate_multimodal_pages

                md_parts = []
                for (_, content_md, _, _, _, _) in generate_multimodal_pages(self.conversion_result):
                    if content_md:
                        md_parts.append(content_md)

                return "\n\n".join(md_parts)
            except Exception as e2:
                logger.warning(f"Multimodal export failed: {e2}")

                # Last resort - extract plain text
                try:
                    return doc.export_to_text()
                except Exception as e3:
                    logger.error(f"All export methods failed: {e3}")
                    return ""

    async def _save_markdown(self, md_content: str) -> None:
        """Save markdown content to file with error handling."""
        try:
            self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self.markdown_path, "w", encoding="utf-8") as f:
                await f.write(md_content)
            logger.info(f"Saved markdown content to {self.markdown_path}")
        except Exception as e:
            logger.error(f"Failed to save markdown: {e}")

    async def _extract_content_elements(self, doc: DoclingDocument) -> List[ContentElement]:
        """
        Extract content elements (text, tables, pictures, headings) with enhanced
        hierarchical structure tracking and domain awareness.
        """
        extraction_start = time.time()
        elements = []

        # 1) Create a "page container" element for each page
        page_map = {}
        if doc.pages:
            for page_no, page_obj in doc.pages.items():
                page_id = f"page_{self.pdf_id}_{page_no}"
                page_element = self._create_content_element(
                    element_id=page_id,
                    content=f"Page {page_no}",
                    content_type=ContentType.PAGE,
                    metadata=ContentMetadata(
                        pdf_id=self.pdf_id,
                        page_number=page_no,
                        content_type=ContentType.PAGE,
                        hierarchy_level=0
                    )
                )
                elements.append(page_element)
                page_map[page_no] = page_id

        # Track section hierarchy
        current_section_path = []
        section_levels = {}

        # 2) Process all items with improved hierarchy tracking
        for item, level in doc.iterate_items():
            # Determine page_number from prov
            page_number = 0
            if item.prov and hasattr(item.prov[0], "page_no"):
                page_number = item.prov[0].page_no

            # Enhanced hierarchical section tracking
            if isinstance(item, SectionHeaderItem):
                # Adjust section path based on header level
                while current_section_path and section_levels.get(current_section_path[-1], 0) >= level:
                    current_section_path.pop()

                # Add current section to path
                section_title = item.text.strip()
                current_section_path.append(section_title)
                section_levels[section_title] = level

                # Save to section hierarchy for export
                if current_section_path not in self.section_hierarchy:
                    self.section_hierarchy.append(list(current_section_path))

                # Extract technical terms from section header
                technical_terms = []
                if self.config.extract_technical_terms:
                    technical_terms = extract_technical_terms(section_title)

                # Create header element with proper section context
                hdr_id = f"hdr_{self.pdf_id}_{uuid.uuid4().hex[:8]}"
                hdr_element = self._create_content_element(
                    element_id=hdr_id,
                    content=item.text,
                    content_type=ContentType.TEXT,
                    metadata=ContentMetadata(
                        pdf_id=self.pdf_id,
                        page_number=page_number,
                        content_type=ContentType.TEXT,
                        hierarchy_level=level,
                        technical_terms=technical_terms,
                        section_headers=list(current_section_path),
                    )
                )

                # Store section mapping for this element
                self.element_section_map[hdr_id] = list(current_section_path)
                self.section_map[section_title] = level

                # Link to page parent if applicable
                if page_number in page_map:
                    hdr_element.metadata.parent_element = page_map[page_number]

                elements.append(hdr_element)

                # Add to concept network
                if technical_terms:
                    section_path_str = " > ".join(current_section_path)
                    self._add_concepts_to_section(section_path_str, technical_terms)

                # Track domain-specific terms for document categorization
                self._track_domain_terms(item.text)

            elif isinstance(item, TableItem):
                table_element = self._process_table_item(
                    item,
                    doc,
                    section_headers=list(current_section_path),
                    hierarchy_level=level
                )
                if table_element:
                    # Parent the table to its page
                    if page_number in page_map:
                        table_element.metadata.parent_element = page_map[page_number]
                    elements.append(table_element)

                    # Store section mapping
                    self.element_section_map[table_element.element_id] = list(current_section_path)

                    # Track domain-specific terms
                    if table_element.metadata.table_data and table_element.metadata.table_data.caption:
                        self._track_domain_terms(table_element.metadata.table_data.caption)

            elif isinstance(item, PictureItem):
                pic_element = self._process_picture_item(
                    item,
                    doc,
                    section_headers=list(current_section_path),
                    hierarchy_level=level
                )
                if pic_element:
                    # Parent the picture to its page
                    if page_number in page_map:
                        pic_element.metadata.parent_element = page_map[page_number]
                    elements.append(pic_element)

                    # Store section mapping
                    self.element_section_map[pic_element.element_id] = list(current_section_path)

                    # Track domain-specific terms in image descriptions
                    if pic_element.metadata.image_metadata and pic_element.metadata.image_metadata.analysis.description:
                        self._track_domain_terms(pic_element.metadata.image_metadata.analysis.description)

            elif isinstance(item, TextItem):
                # skip empty text
                if not item.text.strip():
                    continue

                text_id = f"txt_{self.pdf_id}_{uuid.uuid4().hex[:8]}"

                # Extract technical terms
                technical_terms = []
                if self.config.extract_technical_terms:
                    technical_terms = extract_technical_terms(item.text)

                text_element = self._create_content_element(
                    element_id=text_id,
                    content=item.text,
                    content_type=ContentType.TEXT,
                    metadata=ContentMetadata(
                        pdf_id=self.pdf_id,
                        page_number=page_number,
                        content_type=ContentType.TEXT,
                        hierarchy_level=level,
                        technical_terms=technical_terms,
                        section_headers=list(current_section_path)
                    )
                )

                # Parent the text to its page
                if page_number in page_map:
                    text_element.metadata.parent_element = page_map[page_number]

                elements.append(text_element)

                # Store section mapping
                self.element_section_map[text_id] = list(current_section_path)

                # Add technical terms to section concepts
                if technical_terms and current_section_path:
                    section_path_str = " > ".join(current_section_path)
                    self._add_concepts_to_section(section_path_str, technical_terms)

                # Track domain-specific terms for document categorization
                self._track_domain_terms(item.text)

        # 3. Calculate metrics
        self.metrics["timings"]["extraction"] = time.time() - extraction_start
        self.metrics["counts"]["total_elements"] = len(elements)
        self.metrics["counts"]["text_elements"] = sum(1 for e in elements if e.content_type == ContentType.TEXT)
        self.metrics["counts"]["table_elements"] = sum(1 for e in elements if e.content_type == ContentType.TABLE)
        self.metrics["counts"]["image_elements"] = sum(1 for e in elements if e.content_type == ContentType.IMAGE)

        logger.info(f"Extracted {len(elements)} content elements with hierarchical context")
        return elements

    def _track_domain_terms(self, text: str) -> None:
        """
        Track occurrences of domain-specific terms to help categorize the document.

        Args:
            text: Text to analyze for domain terms
        """
        text_lower = text.lower()

        # Check each category of domain terms
        for category, terms in DOMAIN_SPECIFIC_TERMS.items():
            for term in terms:
                if term.lower() in text_lower:
                    self.domain_term_counters[category] += 1
                    # Also track the specific term
                    self.domain_term_counters[f"term:{term}"] += 1
                    break  # Only count one match per category per text segment

    def _create_metadata(
        self,
        item: Any,
        doc: DoclingDocument,
        content_type: ContentType,
        section_headers: List[str] = None,
        hierarchy_level: int = 0
    ) -> ContentMetadata:
        """Create uniform metadata for document elements."""
        page_number = 0
        if hasattr(item, "prov") and item.prov:
            if hasattr(item.prov[0], "page_no"):
                page_number = item.prov[0].page_no

        technical_terms = []
        if self.config.extract_technical_terms and hasattr(item, "text"):
            technical_terms = extract_technical_terms(item.text)

        metadata = ContentMetadata(
            pdf_id=self.pdf_id,
            page_number=page_number,
            content_type=content_type,
            hierarchy_level=hierarchy_level,
            technical_terms=technical_terms,
            section_headers=section_headers or [],
            docling_ref=getattr(item, "self_ref", None),
            confidence=1.0
        )

        return metadata

    def _extract_all_technical_terms(self, elements: List[ContentElement]) -> List[str]:
        """
        Extract all technical terms from document elements with deduplication.

        Args:
            elements: List of content elements

        Returns:
            List of unique technical terms
        """
        all_terms = set()

        for element in elements:
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'technical_terms'):
                all_terms.update(element.metadata.technical_terms)

        return list(all_terms)

    async def _update_pdf_metadata(self, document_summary: Dict[str, Any], predicted_category: str) -> None:
        """
        Update PDF metadata in the database with document summary and category.
        Also sets a meaningful description for the PDF.

        Args:
            document_summary: Document summary with title, concepts, insights, etc.
            predicted_category: Predicted document category
        """
        try:
            from app.web.db.models import Pdf
            from app.web.db import db
            import logging

            logger = logging.getLogger(__name__)

            # Fetch the PDF record
            pdf = db.session.execute(
                db.select(Pdf).filter_by(id=self.pdf_id)
            ).scalar_one()

            # Prepare metadata to add
            new_metadata = {
                'document_summary': document_summary,
                'predicted_category': predicted_category,
                'domain_term_counts': dict(self.domain_term_counters),
                'processing_date': datetime.utcnow().isoformat()
            }

            # Update metadata using our method
            pdf.update_metadata(new_metadata)

            # Set category if it's still "general" or not set
            if not pdf.category or pdf.category == 'general':
                pdf.category = predicted_category

            # IMPORTANT: Create a meaningful description from the summary
            description = ""

            # First try using the explicit description field if available
            if document_summary.get('description'):
                description = document_summary['description']
            # Next, try to find a descriptive insight to use
            elif document_summary.get('key_insights'):
                # Sort insights by length (descending) to find the most detailed one
                sorted_insights = sorted(document_summary['key_insights'], key=len, reverse=True)
                if sorted_insights:
                    description = sorted_insights[0]
            # Otherwise, build from title and concepts
            elif document_summary.get('title'):
                title = document_summary['title']
                # Skip generic titles
                if title != "Technical Document":
                    description = title

                # Add primary concepts if available
                if document_summary.get('primary_concepts'):
                    concepts = document_summary['primary_concepts'][:5]
                    if concepts:
                        if description:
                            description += ". "
                        description += f"Key concepts: {', '.join(concepts)}"

            # If still no description, create one from category
            if not description and predicted_category:
                description = f"Technical documentation related to {predicted_category.capitalize()}. "

                # Add primary concepts if available
                if document_summary.get('primary_concepts'):
                    concepts = document_summary['primary_concepts'][:5]
                    if concepts:
                        description += f"Key concepts: {', '.join(concepts)}"

            # Set a default if still empty
            if not description:
                description = f"Technical document with {len(document_summary.get('section_structure', []))} sections"

            # Trim if too long (database constraint might be 500-1000 chars)
            max_length = 500
            if len(description) > max_length:
                description = description[:max_length-3] + "..."

            # Update the description
            pdf.description = description
            logger.info(f"Updated PDF description: {description[:50]}...")

            # Mark as processed
            pdf.processed = True
            pdf.error = None
            pdf.processed_at = datetime.utcnow()

            # Commit changes
            db.session.commit()

            logger.info(f"Updated PDF metadata with summary, category: {predicted_category}, and description")
        except Exception as e:
            logger.error(f"Failed to update PDF metadata: {str(e)}")
            raise

    async def _update_pdf_error(self, error_message: str) -> None:
        """
        Update PDF record with error status.

        Args:
            error_message: Error message to store
        """
        try:
            from app.web.db.models import Pdf
            from app.web.db import db

            # Fetch the PDF record
            pdf = db.session.execute(
                db.select(Pdf).filter_by(id=self.pdf_id)
            ).scalar_one()

            # Update error information
            pdf.processed = False
            pdf.error = error_message
            pdf.processed_at = datetime.utcnow()

            # Commit changes
            db.session.commit()
            logger.info(f"Updated PDF with error status: {self.pdf_id}")
        except Exception as e:
            logger.error(f"Failed to update PDF error status: {str(e)}")
            raise

    def _create_content_element(
        self,
        element_id: str,
        content: str,
        content_type: ContentType,
        metadata: ContentMetadata
    ) -> ContentElement:
        """Create a content element with specified properties."""
        return ContentElement(
            element_id=element_id,
            content=content,
            content_type=content_type,
            pdf_id=self.pdf_id,
            metadata=metadata
        )

    def _process_table_item(
        self,
        item: TableItem,
        doc: DoclingDocument,
        section_headers: List[str] = None,
        hierarchy_level: int = 0
    ) -> Optional[ContentElement]:
        """Process a table item with optimized data extraction."""
        try:
            if not hasattr(item, "data") or not item.data:
                return None

            # Initialize headers and rows first
            headers = []
            rows = []
            caption = ""
            markdown = ""

            # Try advanced DataFrame export
            try:
                df = item.export_to_dataframe()
                headers = df.columns.tolist()
                rows = df.values.tolist()
            except Exception as df_error:
                logger.warning(f"DataFrame export failed: {df_error}")
                # Fallback
                try:
                    if hasattr(item.data, "grid"):
                        grid = item.data.grid
                        if grid and len(grid) > 0:
                            headers = [getattr(cell, "text", "") for cell in grid[0]]
                            rows = []
                            for i in range(1, len(grid)):
                                rows.append([getattr(cell, "text", "") for cell in grid[i]])
                except Exception as grid_error:
                    logger.warning(f"Grid extraction failed: {grid_error}")

            # Now filter headers after they're initialized
            headers = [str(h) for h in headers if h is not None]

            # Try caption
            try:
                if hasattr(item, "caption_text") and callable(getattr(item, "caption_text")):
                    caption = item.caption_text(doc) or ""
            except Exception as e:
                logger.warning(f"Caption extraction failed: {e}")

            # Try to get markdown
            try:
                markdown = item.export_to_markdown()
            except Exception as md_error:
                logger.warning(f"Markdown export failed: {md_error}")
                # Manual fallback
                md_lines = []
                if headers:
                    md_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in rows[:10]:
                        md_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
                markdown = "\n".join(md_lines)

            # Page number
            page_number = 0
            if item.prov and hasattr(item.prov[0], "page_no"):
                page_number = item.prov[0].page_no

            row_count = len(rows)
            col_count = len(headers) if headers else (len(rows[0]) if rows else 0)
            summary = f"Table with {row_count} rows and {col_count} columns"
            if caption:
                summary = f"{caption} - {summary}"

            # Extract technical terms from table content
            technical_terms = []
            if self.config.extract_technical_terms:
                text_parts = [caption] if caption else []
                text_parts.extend(str(h) for h in headers if h)
                if rows and rows[0]:
                    text_parts.extend(str(cell) for cell in rows[0] if cell)
                technical_terms = extract_technical_terms(" ".join(text_parts))

            table_data = TableData(
                headers=headers,
                rows=rows[:10],
                caption=caption,
                markdown=markdown,
                summary=summary,
                row_count=row_count,
                column_count=col_count,
                technical_concepts=technical_terms
            )

            metadata = ContentMetadata(
                pdf_id=self.pdf_id,
                page_number=page_number,
                content_type=ContentType.TABLE,
                technical_terms=technical_terms,
                table_data=table_data,
                section_headers=section_headers or [],
                hierarchy_level=hierarchy_level,
                docling_ref=getattr(item, "self_ref", None)
            )

            element_id = f"tbl_{self.pdf_id}_{uuid.uuid4().hex[:8]}"
            return ContentElement(
                element_id=element_id,
                content=markdown,
                content_type=ContentType.TABLE,
                pdf_id=self.pdf_id,
                metadata=metadata
            )

        except Exception as e:
            logger.warning(f"Failed to process table item: {e}")
            return None

    def _process_picture_item(
        self,
        item: PictureItem,
        doc: DoclingDocument,
        section_headers: List[str] = None,
        hierarchy_level: int = 0
    ) -> Optional[ContentElement]:
        """Process an image item with comprehensive metadata extraction."""
        try:
            if not hasattr(item, "get_image") or not callable(getattr(item, "get_image")):
                return None

            pil_image = None
            try:
                pil_image = item.get_image(doc)
            except Exception as img_error:
                logger.warning(f"Failed to get image: {img_error}")
                return None

            if not pil_image:
                return None

            image_id = f"img_{self.pdf_id}_{uuid.uuid4().hex[:8]}"

            images_dir = self.output_dir / "assets" / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            image_path = images_dir / f"{image_id}.png"

            try:
                pil_image.save(image_path, format="PNG")
            except Exception as save_error:
                logger.warning(f"Failed to save image: {save_error}")
                return None

            caption = "Image"
            try:
                if hasattr(item, "caption_text") and callable(getattr(item, "caption_text")):
                    maybe_cap = item.caption_text(doc)
                    if maybe_cap:
                        caption = maybe_cap
            except Exception as caption_error:
                logger.warning(f"Caption extraction failed: {caption_error}")

            page_number = 0
            if item.prov and hasattr(item.prov[0], "page_no"):
                page_number = item.prov[0].page_no

            # Surrounding context
            surrounding_context = self._extract_surrounding_context(item, doc)

            # Markdown
            rel_path = str(image_path.relative_to(self.output_dir)) if str(image_path).startswith(str(self.output_dir)) else str(image_path)
            md_content = f"![{caption}]({rel_path})"

            image_features = ImageFeatures(
                dimensions=(pil_image.width, pil_image.height),
                aspect_ratio=pil_image.width / pil_image.height if pil_image.height > 0 else 1.0,
                color_mode=pil_image.mode,
                is_grayscale=pil_image.mode in ("L", "LA")
            )

            detected_objects = self._detect_objects_in_image(pil_image, caption)

            # Enhanced technical term extraction from image caption and context
            technical_terms = extract_technical_terms(caption + " " + (surrounding_context or ""))

            image_analysis = ImageAnalysis(
                description=caption,
                detected_objects=detected_objects,
                technical_details={"width": pil_image.width, "height": pil_image.height},
                technical_concepts=technical_terms
            )

            image_paths = ImagePaths(
                original=str(image_path),
                format="PNG",
                size=os.path.getsize(image_path) if os.path.exists(image_path) else 0
            )

            image_metadata = ImageMetadata(
                image_id=image_id,
                paths=image_paths,
                features=image_features,
                analysis=image_analysis,
                page_number=page_number
            )

            metadata = ContentMetadata(
                pdf_id=self.pdf_id,
                page_number=page_number,
                content_type=ContentType.IMAGE,
                technical_terms=technical_terms,
                image_metadata=image_metadata,
                section_headers=section_headers or [],
                hierarchy_level=hierarchy_level,
                surrounding_context=surrounding_context,
                image_path=str(image_path),
                docling_ref=getattr(item, "self_ref", None)
            )

            return ContentElement(
                element_id=image_id,
                content=md_content,
                content_type=ContentType.IMAGE,
                pdf_id=self.pdf_id,
                metadata=metadata
            )

        except Exception as e:
            logger.warning(f"Failed to process picture item: {e}")
            return None

    def _extract_surrounding_context(self, item: Any, doc: DoclingDocument) -> Optional[str]:
        """Extract text surrounding an item to provide context."""
        try:
            context_parts = []
            if hasattr(item, "caption_text") and callable(getattr(item, "caption_text")):
                caption = item.caption_text(doc)
                if caption:
                    context_parts.append(caption)

            page_number = 0
            if hasattr(item, "prov") and item.prov:
                if hasattr(item.prov[0], "page_no"):
                    page_number = item.prov[0].page_no

            # If doc.texts is present, gather some text from same page
            if hasattr(doc, "texts"):
                page_texts = []
                for text_item in doc.texts:
                    text_page = 0
                    if text_item.prov and hasattr(text_item.prov[0], "page_no"):
                        text_page = text_item.prov[0].page_no
                    if text_page == page_number and hasattr(text_item, "text"):
                        page_texts.append(text_item.text)
                # Limit to the first few
                if page_texts:
                    context_parts.append(" ".join(page_texts[:3]))

            return " ".join(context_parts) if context_parts else None
        except Exception as e:
            logger.warning(f"Failed to extract context: {e}")
            return None

    def _detect_objects_in_image(self, image, caption: str) -> List[str]:
        """
        Detect objects in image based on caption and image properties.
        Enhanced with domain-specific detection for technical visualizations.
        """
        objects = []
        caption_lower = caption.lower()

        # Common technical visualization types
        common_visualization_types = {
            "chart", "graph", "diagram", "figure", "table", "schematic",
            "screenshot", "interface", "ui", "drawing", "illustration",
            "architecture", "component", "system", "network", "flow",
            "trend", "history", "visualization", "plot", "panel"
        }

        # Check caption for visualization types
        for obj_type in common_visualization_types:
            if obj_type in caption_lower:
                objects.append(obj_type)

        # Domain-specific detection for Niagara framework
        niagara_specific = {
            "hierarchy": ["hierarchy", "nav tree", "navigation", "structure"],
            "wire_sheet": ["wire sheet", "logic", "function block", "diagram"],
            "trend_chart": ["trend", "chart", "plot", "history", "graph"],
            "px_view": ["px view", "px page", "dashboard", "user interface"]
        }

        for category, terms in niagara_specific.items():
            if any(term in caption_lower for term in terms):
                objects.append(category)

        # If no objects detected yet, check image dimensions for clues
        if not objects:
            width, height = image.size
            if width > height * 1.5:
                objects.append("wide image")
            elif height > width * 1.5:
                objects.append("tall image")

        return objects

    def _generate_chunks(self, docling_doc: DoclingDocument) -> List[Dict[str, Any]]:
        """
        Generate optimized chunks using Docling's HybridChunker.
        Ensures chunks retain document structure and context.
        """
        chunking_start = time.time()
        try:
            tokenizer = get_tokenizer(self.config.embedding_model)
            chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                merge_peers=self.config.merge_list_items,
                merge_list_items=self.config.merge_list_items
            )
            chunks = []
            raw_chunks = list(chunker.chunk(dl_doc=docling_doc))

            for i, chunk in enumerate(raw_chunks):
                page_numbers = self._extract_page_numbers(chunk)
                headings = self._extract_headings(chunk)
                text = chunker.serialize(chunk=chunk)

                technical_terms = extract_technical_terms(text)

                # Determine section context for this chunk
                chunk_section_headers = []
                if headings:
                    chunk_section_headers = self._determine_section_context(headings)

                metadata = {
                    "chunk_id": f"{self.pdf_id}-chunk-{i}",
                    "pdf_id": self.pdf_id,
                    "chunk_index": i,
                    "page_numbers": page_numbers,
                    "headings": headings,
                    "section_headers": chunk_section_headers,  # Include section context
                    "technical_terms": technical_terms,
                    "position": i / max(1, len(raw_chunks) - 1),  # normalized
                    "source_type": "docling_chunk",
                    "content_types": self._detect_content_types(text)
                }

                chunks.append({
                    "chunk_id": metadata["chunk_id"],
                    "content": text,
                    "metadata": metadata
                })

            self.metrics["timings"]["chunking"] = time.time() - chunking_start
            self.metrics["counts"]["chunks"] = len(chunks)

            logger.info(f"Generated {len(chunks)} optimized chunks with HybridChunker")
            return chunks

        except Exception as e:
            logger.error(f"Chunk generation failed: {e}", exc_info=True)
            return []

    def _determine_section_context(self, headings: List[str]) -> List[str]:
        """Determine section context based on headings in a chunk."""
        if not headings:
            return []

        # Use our tracked section hierarchy to build proper context
        # Find the deepest heading that matches our known sections
        for heading in headings:
            if heading in self.section_map:
                level = self.section_map[heading]

                # Find the section path for this heading
                result = []
                for section_path in self.element_section_map.values():
                    if heading in section_path:
                        # Found the section path, return it
                        return section_path

        # If no match found, just return the headings
        return headings

    def _extract_page_numbers(self, chunk: BaseChunk) -> List[int]:
        """Extract page numbers from a chunk."""
        try:
            page_numbers = set()
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if hasattr(item, "prov") and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, "page_no"):
                                page_numbers.add(prov.page_no)
            return sorted(list(page_numbers)) if page_numbers else [0]
        except Exception as e:
            logger.warning(f"Failed to extract page numbers: {e}")
            return [0]

    def _extract_headings(self, chunk: BaseChunk) -> List[str]:
        """Extract headings from a chunk with improved detection."""
        try:
            headings = []
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                for item in chunk.meta.doc_items:
                    if isinstance(item, SectionHeaderItem) or getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
                        if hasattr(item, "text") and item.text:
                            headings.append(item.text.strip())
            return headings
        except Exception as e:
            logger.warning(f"Failed to extract headings: {e}")
            return []

    def _detect_content_types(self, text: str) -> List[str]:
        """
        Detect content types present in text with enhanced patterns.
        Especially focused on domain-specific content like trend data visualizations.
        """
        from app.chat.utils.processing import detect_content_types
        return detect_content_types(text)

    def _predict_document_category(self, technical_terms: List[str], content: str) -> str:
        """
        Predict document category based on detected domain terms and content.
        Uses the counters built up during processing.

        Args:
            technical_terms: List of technical terms extracted from the document
            content: Full document content

        Returns:
            Predicted category based on domain patterns
        """
        # Use domain term counters for more reliable categorization
        if not self.domain_term_counters:
            # Fall back to analyzing term frequency in the technical_terms list
            return self._legacy_predict_category(technical_terms, content)

        # Get the most frequent category based on term counts
        category_counts = {
            category: count for category, count in self.domain_term_counters.items()
            if not category.startswith("term:")  # Filter out individual term counts
        }

        if not category_counts:
            return "general"

        # Get the top category
        top_category = max(category_counts.items(), key=lambda x: x[1])[0]

        # Map to output category name (using the original mapping but with our enhanced counters)
        doc_type_map = {
            "niagara": "tridium",
            "station": "tridium",
            "hierarchy": "tridium",
            "component": "tridium",
            "hvac": "building_automation",
            "control": "building_automation",
            "alarm": "building_automation",
            "trend": "data_visualization",
            "interval": "data_visualization",
            "visualization": "data_visualization",
            "module": "development",
            "node": "development",
            "programming": "development",
            "security": "security",
            "database": "integration",
            "network": "networking"
        }

        return doc_type_map.get(top_category, "general")

    def _legacy_predict_category(self, technical_terms: List[str], content: str) -> str:
        """
        Legacy method to predict document category based on technical terms.
        Used as a fallback when domain term counters are not available.

        Args:
            technical_terms: List of technical terms extracted from the document
            content: Full document content

        Returns:
            Predicted category string
        """
        # Define vendor-specific terminology
        vendor_terms = {
            "tridium": {
                "niagara", "jace", "vykon", "workbench", "baja", "fox", "ax", "n4",
                "iojas", "nrio", "ntec", "niagaraax", "niagara4", "station", "tridium",
                "hierarchy", "nav", "navtree", "hierarchy definition", "hierarchyservice"
            },
            "honeywell": {
                "honeywell", "webs", "websx", "c-bus", "economizer", "spyder", "sylk",
                "excel", "eaglehawk", "jade", "lynx", "centraline", "wcps", "cbs",
                "excel web", "symmetre", "honeyweb", "analytics", "wpa", "ebi"
            },
            "johnson_controls": {
                "johnson", "metasys", "fec", "fas", "fms", "cctp", "vma", "vav", "nae",
                "ncm", "adc", "ddc", "vfd", "bacnet", "n2", "fpm", "n1", "bacpack",
                "jci", "field controller", "facility explorer", "jc companion"
            }
        }

        # Count matches for each vendor
        match_counts = {vendor: 0 for vendor in vendor_terms}

        # Check each technical term against vendor-specific terminology
        for term in technical_terms:
            term_lower = term.lower()
            for vendor, terms in vendor_terms.items():
                # Check for exact or partial match in vendor terms
                for v_term in terms:
                    if v_term in term_lower or term_lower in v_term:
                        match_counts[vendor] += 1
                        break

        # Check content for more vendor mentions (in case terms aren't in the extracted technical terms)
        content_lower = content.lower()
        for vendor, terms in vendor_terms.items():
            for term in terms:
                if term in content_lower:
                    match_counts[vendor] += 0.5  # Lower weight for raw content matches

        # Find the vendor with the highest match count
        if max(match_counts.values()) > 2:  # Require at least 2 matches for confident categorization
            best_vendor = max(match_counts.items(), key=lambda x: x[1])[0]
            return best_vendor

        return "general"  # Default category if no strong matches

    async def _build_concept_network(self, elements: List[ContentElement], chunks: List[Dict[str, Any]]) -> None:
        """
        Build concept network from document content using optimized extraction methods.
        Enhanced to focus on key technical concepts and domain-specific relationships.
        """
        try:
            # 1. Set enhanced configuration for concept extraction
            MIN_CONCEPT_OCCURRENCES = 1  # Capture ALL technical terms, even rare ones
            MIN_RELATIONSHIP_CONFIDENCE = 0.6  # Balanced value for relationships
            MAX_CONCEPTS = self.config.max_concepts_per_document

            # 2. Extract concepts with improved domain awareness
            concepts_info = defaultdict(lambda: {
                "count": 0,
                "in_headers": False,
                "sections": set(),
                "pages": set(),
                "domain_category": None  # Track domain category for each concept
            })

            # Process elements with deep extraction of all technical terms
            for element in elements:
                if element.metadata.technical_terms:
                    for term in element.metadata.technical_terms:
                        # Include ALL terms - even short ones might be important in technical docs
                        concepts_info[term]["count"] += 1

                        # Track header, section, and page information
                        if element.metadata.hierarchy_level <= 2:
                            concepts_info[term]["in_headers"] = True
                        if element.metadata.section_headers:
                            concepts_info[term]["sections"].update(element.metadata.section_headers)
                        if element.metadata.page_number:
                            concepts_info[term]["pages"].add(element.metadata.page_number)

                        # Check if this is a domain-specific term
                        for category, domain_terms in DOMAIN_SPECIFIC_TERMS.items():
                            term_lower = term.lower()
                            if any(dt.lower() in term_lower or term_lower in dt.lower() for dt in domain_terms):
                                concepts_info[term]["domain_category"] = category
                                break

            # 3. Score concepts for importance with enhanced logic
            concept_scores = {}
            for term, info in concepts_info.items():
                # Base score from occurrences with logarithmic scaling to avoid over-penalizing rare terms
                occurrences = info["count"]
                base_score = 0.5 + min(0.5, math.log(occurrences + 1) / 5.0)

                # Bonus for appearing in headers (critical for technical docs)
                header_bonus = 0.3 if info["in_headers"] else 0

                # Bonus for appearing in multiple sections
                section_bonus = min(0.3, len(info["sections"]) * 0.1)

                # Bonus for multi-word terms (likely more specific)
                specificity_bonus = 0.1 if " " in term else 0

                # Domain-specific bonus - prioritize terms from our known domain
                domain_bonus = 0.3 if info["domain_category"] else 0

                # Page coverage bonus - terms that appear across many pages are important
                page_bonus = min(0.2, len(info["pages"]) * 0.02)

                # Combined score
                concept_scores[term] = base_score + header_bonus + section_bonus + specificity_bonus + domain_bonus + page_bonus

            # 4. Select top concepts by importance score
            top_concepts = sorted(
                concept_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:MAX_CONCEPTS]

            # 5. Create concept objects for only the top concepts
            concept_objects = []
            top_concept_terms = set(term for term, _ in top_concepts)

            for term, score in top_concepts:
                info = concepts_info[term]
                concept = Concept(
                    name=term,
                    occurrences=info["count"],
                    in_headers=info["in_headers"],
                    sections=list(info["sections"]),
                    first_occurrence_page=min(info["pages"]) if info["pages"] else None,
                    importance_score=score,
                    is_primary=score > 0.8,  # Top concepts are primary
                    category=info["domain_category"]  # Include domain category
                )
                concept_objects.append(concept)
                self.concept_network.add_concept(concept)

            # 6. Extract all text content for relationship analysis
            full_text = ""
            for element in elements:
                if element.content_type == ContentType.TEXT and element.content:
                    full_text += element.content + "\n\n"

            # 7. Use the extract_document_relationships function with domain awareness
            relationships = extract_document_relationships(
                text=full_text,
                technical_terms=list(top_concept_terms),
                min_confidence=MIN_RELATIONSHIP_CONFIDENCE
            )

            # 8. Add extracted relationships to the concept network
            for rel in relationships:
                relationship = ConceptRelationship(
                    source=rel["source"],
                    target=rel["target"],
                    type=RelationType.map_type(rel["type"]),  # Convert to enum
                    weight=rel.get("weight", 0.75),
                    context=rel.get("context", ""),
                    extraction_method=rel.get("extraction_method", "document-based")
                )
                self.concept_network.add_relationship(relationship)

            # 9. Calculate importance scores and identify primary concepts
            self.concept_network.calculate_importance_scores()

            # 10. Build section to concept mapping
            self.concept_network.build_section_concept_map()

            # 11. Register with research manager for reuse
            if self.research_manager:
                self.research_manager.register_concept_network(self.pdf_id, self.concept_network)

                # Also register primary concepts for cross-document analysis
                if self.concept_network.primary_concepts:
                    for concept in self.concept_network.primary_concepts:
                        self.research_manager.register_shared_concept(
                            concept=concept,
                            pdf_ids={self.pdf_id},
                            confidence=0.95  # High confidence for primary concepts
                        )

            logger.info(
                f"Built optimized concept network with {len(concept_objects)} concepts "
                f"and {len(relationships)} relationships"
            )

        except Exception as e:
            logger.error(f"Concept network building failed: {e}", exc_info=True)
            # Don't fail completely, create an empty network
            self.concept_network = ConceptNetwork()

    def _add_concepts_to_section(self, section_path: str, concepts: List[str]) -> None:
        """Add concepts to a section in the concept network."""
        # Skip if no section path or concepts
        if not section_path or not concepts:
            return

        # Add to concept network's section mapping
        self.concept_network.add_section_concepts(section_path, concepts)

    def _setup_directories(self) -> None:
        """Setup directory structure for processing outputs."""
        try:
            dirs = [
                self.output_dir / "content",
                self.output_dir / "assets" / "images",
                self.output_dir / "assets" / "raw",
                self.output_dir / "metadata",
                self.output_dir / "temp",
            ]
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Directory setup error: {e}")
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (self.output_dir / "content").mkdir(exist_ok=True)
            except Exception:
                pass

    async def _ingest_to_neo4j(self, result: ProcessingResult) -> bool:
        """
        Ingest processed content into Neo4j vector store.
        This is the key connection point between document processing and Neo4j.
        
        Args:
            result: Processing result with elements and concept network
        
        Returns:
            Success status
        """
        logger.info(f"Ingesting processed content to Neo4j for {self.pdf_id}")
        
        try:
            # Get Neo4j vector store
            from app.chat.vector_stores import get_vector_store
            vector_store = get_vector_store()
            
            # Verify it's initialized
            if not vector_store.initialized:
                logger.error(f"Neo4j vector store not initialized for {self.pdf_id}")
                return False
            
            # 1. Create document node first
            document_title = "Untitled Document"
            if hasattr(result, 'document_summary') and result.document_summary:
                if 'title' in result.document_summary:
                    document_title = result.document_summary['title']
            
            metadata = {
                "processed_at": datetime.utcnow().isoformat(),
                "element_count": len(result.elements),
                "domain_category": self._predict_document_category(
                    self._extract_all_technical_terms(result.elements), 
                    result.markdown_content
                )
            }
            
            # Add document summary to metadata if available
            if hasattr(result, 'document_summary') and result.document_summary:
                metadata["document_summary"] = result.document_summary
            
            # Create document node
            await vector_store.create_document_node(
                pdf_id=self.pdf_id,
                title=document_title,
                metadata=metadata
            )
            
            # 2. Add content elements with appropriate relationships
            for element in result.elements:
                await vector_store.add_content_element(element, self.pdf_id)
            
            # 3. Add concepts and their relationships
            if hasattr(result, 'concept_network') and result.concept_network:
                # Add concepts
                for concept in result.concept_network.concepts:
                    await vector_store.add_concept(
                        concept_name=concept.name,
                        pdf_id=self.pdf_id,
                        metadata={
                            "importance": concept.importance_score,
                            "is_primary": concept.is_primary,
                            "category": concept.category
                        }
                    )
                
                # Add relationships between concepts
                for relationship in result.concept_network.relationships:
                    rel_type = relationship.type
                    if hasattr(rel_type, 'value'):
                        rel_type = rel_type.value
                    
                    await vector_store.add_concept_relationship(
                        source=relationship.source,
                        target=relationship.target,
                        rel_type=str(rel_type),
                        pdf_id=self.pdf_id,
                        metadata={
                            "weight": relationship.weight,
                            "context": relationship.context
                        }
                    )
                
                # Add section-concept relationships
                if hasattr(result.concept_network, 'section_concepts'):
                    for section, concepts in result.concept_network.section_concepts.items():
                        for concept in concepts:
                            await vector_store.add_section_concept_relation(
                                section=section,
                                concept=concept,
                                pdf_id=self.pdf_id
                            )
            
            logger.info(f"Successfully ingested content to Neo4j for {self.pdf_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest content to Neo4j: {str(e)}", exc_info=True)
            return False

    async def _save_results(self, result: ProcessingResult) -> None:
        """Save processing results to disk with comprehensive organization."""
        try:
            for dir_path in [
                self.output_dir / "content",
                self.output_dir / "metadata",
                self.output_dir / "assets" / "images"
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Save elements by type
            elements_by_type = defaultdict(list)
            for element in result.elements:
                elements_by_type[element.content_type.value].append(element.dict())

            for ctype, elements_lst in elements_by_type.items():
                path = self.output_dir / "content" / f"{ctype}_elements.json"
                try:
                    async with aiofiles.open(path, "w") as f:
                        await f.write(json.dumps(elements_lst, default=str))
                except Exception as e:
                    logger.warning(f"Failed to write {ctype} elements: {e}")

            # Save chunks
            chunks_path = self.output_dir / "content" / "chunks.json"
            try:
                async with aiofiles.open(chunks_path, "w") as f:
                    await f.write(json.dumps(result.chunks, default=str))
            except Exception as e:
                logger.warning(f"Failed to write chunks: {e}")

            # Save concept network
            network_path = self.output_dir / "metadata" / "concept_network.json"
            try:
                async with aiofiles.open(network_path, "w") as f:
                    await f.write(json.dumps(self.concept_network.dict(), default=str))
            except Exception as e:
                logger.warning(f"Failed to write concept network: {e}")

            # Save domain term counts
            counts_path = self.output_dir / "metadata" / "domain_term_counts.json"
            try:
                async with aiofiles.open(counts_path, "w") as f:
                    await f.write(json.dumps(dict(self.domain_term_counters), default=str))
            except Exception as e:
                logger.warning(f"Failed to write domain term counts: {e}")

            # Save comprehensive metadata
            meta_data = {
                "pdf_id": self.pdf_id,
                "processing_info": {
                    "start_time": self.processing_start.isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                    "config": self.config.dict(),
                },
                "metrics": self.metrics,
                "content_summary": {
                    "total_elements": len(result.elements),
                    "elements_by_type": {k: len(v) for k, v in elements_by_type.items()},
                    "total_chunks": len(result.chunks),
                    "concept_count": len(self.concept_network.concepts),
                    "relationship_count": len(self.concept_network.relationships),
                    "primary_concepts": self.concept_network.primary_concepts,
                    "has_markdown": bool(result.markdown_content),
                    "markdown_path": str(self.markdown_path) if os.path.exists(self.markdown_path) else None,
                    "section_structure": {
                        k: v for k, v in self.concept_network.section_concepts.items()
                    },
                    "domain_term_counts": dict(self.domain_term_counters)
                },
                "neo4j_status": "ingested"  # Add Neo4j status
            }

            # Save metadata
            meta_path = self.output_dir / "metadata" / "processing_result.json"
            try:
                async with aiofiles.open(meta_path, "w") as f:
                    await f.write(json.dumps(meta_data, default=str))
            except Exception as e:
                logger.warning(f"Failed to write metadata: {e}")

            logger.info(f"Successfully saved results for {self.pdf_id}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")


async def process_technical_document(
    pdf_id: str,
    config: Optional[ProcessingConfig] = None,
    openai_client: Optional[AsyncOpenAI] = None,
    research_manager = None,
    output_dir: Optional[Path] = None
) -> ProcessingResult:
    """
    Process technical document with Neo4j integration and LangGraph compatibility.
    Optimized for technical documentation with concept extraction and relationship mapping.

    Args:
        pdf_id: Document identifier
        config: Processing configuration
        openai_client: OpenAI client
        research_manager: Optional research manager for cross-document analysis
        output_dir: Optional custom output directory

    Returns:
        ProcessingResult with LangGraph-ready structured content
    """
    logger.info(f"Starting Neo4j-integrated document processing pipeline for {pdf_id}")
    
    # Default config if not provided
    if not config:
        config = ProcessingConfig(
            pdf_id=pdf_id,
            chunk_size=500,  # Optimal for technical content
            chunk_overlap=100,  # Better context preservation
            embedding_model="text-embedding-3-small",
            process_images=True,
            process_tables=True,
            extract_technical_terms=True,
            extract_relationships=True,
            max_concepts_per_document=200
        )
    
    # Default OpenAI client if not provided
    if not openai_client:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create processor instance
    processor = DocumentProcessor(pdf_id, config, openai_client, research_manager)

    if output_dir:
        processor.output_dir = output_dir
        processor._setup_directories()

    try:
        # Process document with Neo4j integration
        result = await processor.process_document()
        
        # Add LangGraph-specific metadata
        if not result.raw_data:
            result.raw_data = {}
            
        result.raw_data["langgraph"] = {
            "node_ready": True,
            "document_structure": processor.section_hierarchy if hasattr(processor, "section_hierarchy") else [],
            "primary_concepts": [c.name for c in processor.concept_network.concepts[:5]] if processor.concept_network and processor.concept_network.concepts else [],
            "technical_domain": processor._predict_document_category(
                processor._extract_all_technical_terms(result.elements), 
                result.markdown_content
            ),
            "processing_timestamp": datetime.utcnow().isoformat(),
            "neo4j_ready": True  # Indicate Neo4j storage is ready
        }
        
        logger.info(
            f"Completed Neo4j-integrated processing for {pdf_id} with "
            f"{len(result.elements)} elements, "
            f"{len(result.chunks) if hasattr(result, 'chunks') else 0} chunks, and "
            f"{len(processor.concept_network.concepts) if hasattr(processor, 'concept_network') and processor.concept_network else 0} concepts"
        )
        return result

    except Exception as e:
        logger.error(f"Processing pipeline failed: {e}", exc_info=True)
        error_result = ProcessingResult(
            pdf_id=pdf_id,
            elements=[],
            raw_data={
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "langgraph_ready": False,
                "neo4j_ready": False
            }
        )
        try:
            error_path = Path("output") / pdf_id / "metadata" / "processing_error.json"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_path, "w") as f:
                f.write(json.dumps({
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "pdf_id": pdf_id,
                    "langgraph_ready": False,
                    "neo4j_ready": False
                }, indent=2))
        except Exception:
            pass
        raise ProcessingError(f"Document processing pipeline failed: {e}")
"""
Document processor adapter for n8n integration.
This module adapts the existing DocumentProcessor for our API needs.
"""

import asyncio
import io
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, BinaryIO

from openai import AsyncOpenAI
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DoclingDocument

from .types import ProcessingConfig, ProcessingResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processor adapter for n8n integration.
    Simplified version of the existing DocumentProcessor.
    """

    def __init__(
        self,
        pdf_id: str,
        config: ProcessingConfig,
        openai_client: AsyncOpenAI
    ):
        self.pdf_id = pdf_id
        self.config = config
        self.openai_client = openai_client
        self.output_dir = Path("output") / self.pdf_id
        self._setup_directories()
        self.markdown_path = self.output_dir / "content" / "document.md"
        self.docling_doc = None
        self.conversion_result = None
        
        # Track timings and counts for metrics
        self.processing_start = time.time()
        self.metrics = {"timings": {}, "counts": {}}

    def _setup_directories(self):
        """Create necessary directories for document processing."""
        # Create main output dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(self.output_dir / "content", exist_ok=True)
        os.makedirs(self.output_dir / "images", exist_ok=True)
        os.makedirs(self.output_dir / "chunks", exist_ok=True)

    async def process_document(self, content: bytes) -> ProcessingResponse:
        """
        Process a document from binary content.
        
        Args:
            content: PDF binary content
            
        Returns:
            ProcessingResponse with extracted content
        """
        logger.info(f"Starting document processing for PDF ID: {self.pdf_id}")
        start_time = time.time()
        
        try:
            # 1. Convert document to Docling format
            logger.info(f"Converting document {self.pdf_id}")
            self.docling_doc = await self._convert_document(content)
            
            # 2. Extract and save markdown content
            logger.info(f"Exporting document {self.pdf_id} to markdown")
            md_content = await self._extract_markdown(self.docling_doc)
            await self._save_markdown(md_content)
            
            # 3. Extract tables and images
            logger.info(f"Extracting tables and images from {self.pdf_id}")
            tables, images = await self._extract_tables_and_images(self.docling_doc)
            
            # 4. Extract technical terms if enabled
            technical_terms = []
            if self.config.extract_technical_terms:
                logger.info(f"Extracting technical terms from {self.pdf_id}")
                technical_terms = await self._extract_technical_terms(md_content)
            
            # 5. Extract procedures and parameters if enabled
            procedures = []
            parameters = []
            if self.config.extract_procedures:
                logger.info(f"Extracting procedures and parameters from {self.pdf_id}")
                procedures, parameters = await self._extract_procedures_and_parameters(md_content)
            
            # 6. Extract concept relationships if enabled
            concept_relationships = []
            if self.config.extract_relationships:
                logger.info(f"Extracting concept relationships from {self.pdf_id}")
                concept_relationships = await self._extract_relationships(md_content, technical_terms)
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            return ProcessingResponse(
                pdf_id=self.pdf_id,
                markdown=md_content,
                tables=tables,
                images=images,
                metadata=self.metrics,
                technical_terms=technical_terms,
                procedures=procedures,
                parameters=parameters,
                concept_relationships=concept_relationships,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    async def _convert_document(self, content: bytes) -> DoclingDocument:
        """
        Convert PDF binary content to Docling document format.
        
        Args:
            content: PDF binary content
            
        Returns:
            DoclingDocument object
        """
        converter = DocumentConverter()
        
        ocr_options = TesseractCliOcrOptions(enabled=True)
        pdf_options = PdfPipelineOptions(
            perform_ocr=True,
            ocr_options=ocr_options,
            format_option=PdfFormatOption.default,
            detect_tables=True,
            detect_figures=True
        )
        
        # Create BytesIO object from bytes content
        pdf_content = io.BytesIO(content)
        
        # Start conversion timer
        conversion_start = time.time()
        
        # Convert document
        self.conversion_result = await converter.convert_document(
            input_file=pdf_content,
            input_format=InputFormat.PDF,
            output_dir=self.output_dir,
            doc_id=self.pdf_id,
            options=pdf_options
        )
        
        # Record conversion time
        self.metrics["timings"]["conversion"] = time.time() - conversion_start
        
        return self.conversion_result.document

    async def _extract_markdown(self, doc: DoclingDocument) -> str:
        """
        Extract markdown content from Docling document.
        
        Args:
            doc: Docling document
            
        Returns:
            Markdown content as string
        """
        # In a real implementation, you would use Docling's markdown export functionality
        # For this adaptation, we'll use a simplified approach
        
        markdown_content = ""
        for section in doc.sections:
            if section.heading:
                level = min(section.level, 6)  # Markdown supports headers up to level 6
                markdown_content += f"{'#' * level} {section.heading.text}\n\n"
            
            for item in section.content:
                if hasattr(item, 'text'):
                    markdown_content += f"{item.text}\n\n"
                elif hasattr(item, 'caption'):
                    markdown_content += f"*{item.caption}*\n\n"
        
        return markdown_content

    async def _save_markdown(self, content: str) -> None:
        """
        Save markdown content to file.
        
        Args:
            content: Markdown content to save
        """
        with open(self.markdown_path, "w", encoding="utf-8") as f:
            f.write(content)

    async def _extract_tables_and_images(self, doc: DoclingDocument) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Extract tables and images from Docling document.
        
        Args:
            doc: Docling document
            
        Returns:
            Tuple of (tables, image_paths)
        """
        tables = []
        images = []
        
        # Extract tables
        for item in doc.find_by_type("table"):
            if hasattr(item, 'data') and item.data:
                table_data = {
                    "table_id": str(uuid.uuid4()),
                    "caption": item.caption if hasattr(item, 'caption') else "",
                    "data": item.data if hasattr(item, 'data') else [],
                    "page": item.page_num if hasattr(item, 'page_num') else 0
                }
                tables.append(table_data)
        
        # Extract images
        for item in doc.find_by_type("picture"):
            if hasattr(item, 'source') and item.source:
                # Save image to output directory
                image_filename = f"image_{len(images)}.png"
                image_path = str(self.output_dir / "images" / image_filename)
                
                # In a real implementation, you would save the image here
                # For this adaptation, we'll just record the path
                images.append(image_path)
        
        return tables, images

    async def _extract_technical_terms(self, text: str) -> List[str]:
        """
        Extract technical terms from document text using OpenAI.
        
        Args:
            text: Document text
            
        Returns:
            List of technical terms
        """
        # In a real implementation, you would use your extract_technical_terms utility
        # For this adaptation, we'll use a simplified approach with OpenAI
        
        prompt = f"""
        Extract all technical terms, domain-specific vocabulary, and key concepts from this text.
        Return only a JSON array of strings, with no explanation or additional formatting.
        
        Text:
        {text[:4000]}...
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        try:
            import json
            terms_dict = json.loads(result)
            return terms_dict.get("terms", [])
        except Exception as e:
            logger.error(f"Error parsing technical terms: {str(e)}")
            return []

    async def _extract_procedures_and_parameters(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract procedures and parameters from document text.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (procedures, parameters)
        """
        # In a real implementation, you would use your extract_procedures_and_parameters utility
        # For this adaptation, we'll use a simplified approach
        
        procedures = []
        parameters = []
        
        # Simple procedure extraction logic
        import re
        proc_sections = re.split(r'\n#{1,3}\s+', text)
        for i, section in enumerate(proc_sections):
            if i == 0:  # Skip first split result (before first heading)
                continue
                
            if re.search(r'\b(procedure|step|instruction|how\s+to)\b', section.lower()):
                proc_title = section.split('\n')[0].strip()
                proc_id = f"proc_{i}"
                
                procedures.append({
                    "id": proc_id,
                    "title": proc_title,
                    "content": section,
                    "steps": [],
                    "parameters": []
                })
                
                # Look for parameters in this procedure
                param_matches = re.finditer(r'\*\*([^:]+):\*\*\s*([^\n]+)', section)
                for j, match in enumerate(param_matches):
                    param_name = match.group(1).strip()
                    param_value = match.group(2).strip()
                    
                    param_id = f"param_{i}_{j}"
                    param = {
                        "id": param_id,
                        "name": param_name,
                        "value": param_value,
                        "procedure_id": proc_id
                    }
                    
                    parameters.append(param)
                    
                    # Add reference to procedure
                    if param not in procedures[-1]["parameters"]:
                        procedures[-1]["parameters"].append(param_id)
        
        return procedures, parameters

    async def _extract_relationships(self, text: str, terms: List[str]) -> List[Dict[str, Any]]:
        """
        Extract concept relationships from document text.
        
        Args:
            text: Document text
            terms: List of technical terms
            
        Returns:
            List of concept relationships
        """
        # In a real implementation, you would use your extract_document_relationships utility
        # For this adaptation, we'll return a simplified skeleton
        
        relationships = []
        
        # Skip for brevity in this adaptation
        # In a real implementation, you would analyze the document to extract relationships
        
        return relationships

"""
Document processor node for LangGraph-based PDF RAG system.
This node handles document extraction, chunking, and concept network building.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from openai import AsyncOpenAI
import os

from app.chat.langgraph.state import GraphState
from app.chat.types import ProcessingConfig, ContentElement, ConceptNetwork
from app.chat.models.conversation import ConversationState, MessageType
from app.chat.errors import DocumentProcessingError

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """LangGraph node for processing documents"""

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        """Initialize document processor with optional OpenAI client"""
        self.openai_client = openai_client or AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def process_document(self, pdf_id: str, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
        """
        Process a document, extract its content, and store in vector database.

        Args:
            pdf_id: ID of the PDF to process
            config: Optional processing configuration

        Returns:
            Processing results including extracted content and metadata
        """
        from app.chat.vector_stores import Neo4jVectorStore, get_vector_store

        try:
            logger.info(f"Processing document {pdf_id} with LangGraph processor")

            # Initialize processing results
            result = {
                "pdf_id": pdf_id,
                "status": "processing",
                "elements": [],
                "metadata": {},
                "timestamp": datetime.utcnow().isoformat(),
            }

            # 1. Extract document content
            extracted_content = await self._extract_document_content(pdf_id)

            # 2. Process and chunk the content
            processed_elements = await self._process_and_chunk_content(extracted_content)
            result["elements"] = processed_elements

            # 3. Store content in vector database
            vector_store = get_vector_store()
            if not vector_store.initialized:
                raise DocumentProcessingError(f"Vector store initialization failed for {pdf_id}")

            # 4. Add elements to vector store
            await self._store_document_elements(vector_store, processed_elements, pdf_id)

            # 5. Generate document summary
            summary = await self._generate_document_summary(processed_elements)
            result["summary"] = summary

            # 6. Extract technical terms and concepts
            concepts = await self._extract_technical_concepts(processed_elements)
            result["concepts"] = concepts

            # Update status
            result["status"] = "complete"
            result["element_count"] = len(processed_elements)

            logger.info(f"Document {pdf_id} processing complete: {len(processed_elements)} elements extracted")

            return result

        except Exception as e:
            logger.error(f"Document processing error for {pdf_id}: {str(e)}", exc_info=True)
            return {
                "pdf_id": pdf_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _extract_document_content(self, pdf_id: str) -> List[Dict[str, Any]]:
        """Extract raw content from PDF document"""
        # For now, this is a placeholder. In a real implementation,
        # this would use a PDF extraction library
        logger.info(f"Extracting content from document {pdf_id}")

        # Simulate PDF extraction
        return [
            {
                "type": "text",
                "content": f"This is sample content from PDF {pdf_id}, page 1.",
                "page": 1,
                "metadata": {"section": "introduction"}
            },
            {
                "type": "text",
                "content": f"This is sample content from PDF {pdf_id}, page 2.",
                "page": 2,
                "metadata": {"section": "body"}
            }
        ]

    async def _process_and_chunk_content(self, content: List[Dict[str, Any]]) -> List[ContentElement]:
        """Process and chunk the document content"""
        from app.chat.utils.processing import create_markdown_chunker

        logger.info(f"Processing and chunking content: {len(content)} raw elements")

        # Convert to ContentElement format
        elements = []
        for item in content:
            element = ContentElement(
                element_id=f"element_{len(elements)}",
                content=item["content"],
                content_type=item.get("type", "text"),
                pdf_id="sample_pdf",
                metadata={
                    "page_number": item.get("page", 1),
                    "section": item.get("metadata", {}).get("section", ""),
                    "content_type": item.get("type", "text"),
                }
            )
            elements.append(element)

        return elements

    async def _store_document_elements(
        self,
        vector_store: Any,
        elements: List[ContentElement],
        pdf_id: str
    ) -> None:
        """Store document elements in vector database"""
        logger.info(f"Storing {len(elements)} elements for document {pdf_id}")

        # Batch process elements
        for i, element in enumerate(elements):
            try:
                # Add the document to the vector store
                await vector_store.add_content_element(element, pdf_id)

                if i > 0 and i % 10 == 0:
                    logger.info(f"Stored {i}/{len(elements)} elements for document {pdf_id}")

            except Exception as e:
                logger.error(f"Error storing element {i} for document {pdf_id}: {str(e)}")
                # Continue processing other elements

        logger.info(f"Completed storing all {len(elements)} elements for document {pdf_id}")

    async def _generate_document_summary(self, elements: List[ContentElement]) -> str:
        """Generate a summary of the document using LLM"""
        try:
            # Combine elements into a single text for summarization
            combined_text = "\n".join([el.content for el in elements[:10]])  # Use first 10 elements

            # Generate summary using OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a technical document summarizer. Create a concise summary of the following document excerpt:"},
                    {"role": "user", "content": combined_text}
                ],
                max_tokens=200
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating document summary: {str(e)}")
            return "Error generating summary"

    async def _extract_technical_concepts(self, elements: List[ContentElement]) -> Dict[str, Any]:
        """Extract technical concepts and terms from document elements"""
        from app.chat.utils.extraction import extract_technical_terms, extract_concept_relationships

        try:
            # Combine content for concept extraction
            combined_text = "\n".join([el.content for el in elements])

            # Extract technical terms
            terms = extract_technical_terms(combined_text)

            # Extract concept relationships between terms
            relationships = extract_concept_relationships(combined_text, set(terms))

            return {
                "terms": terms,
                "relationships": relationships
            }
        except Exception as e:
            logger.error(f"Error extracting technical concepts: {str(e)}")
            return {"terms": [], "relationships": []}


async def process_document(state: GraphState) -> GraphState:
    """
    Process a document and extract structured content.
    This node is the entry point for document processing in the LangGraph.

    Args:
        state: Current graph state

    Returns:
        Updated graph state
    """
    # Get PDF ID from conversation metadata or document_state
    pdf_id = None

    if state.conversation_state and state.conversation_state.metadata:
        pdf_id = state.conversation_state.metadata.get("pdf_id")

    if not pdf_id and state.document_state:
        if isinstance(state.document_state, dict):
            pdf_id = state.document_state.get("pdf_id")
        else:
            # For backward compatibility
            pdf_id = getattr(state.document_state, "pdf_id", None)

    if not pdf_id:
        logger.warning("No PDF ID specified in state")
        if state.conversation_state:
            state.conversation_state.add_message(
                MessageType.SYSTEM,
                "Document processing failed: No PDF ID specified"
            )
        return state

    logger.info(f"Processing document with PDF ID: {pdf_id}")

    # Create processor
    processor = DocumentProcessor()

    # Process document
    try:
        result = await processor.process_document(pdf_id)

        # Update state with processing results
        state.document_state = {
            "pdf_id": pdf_id,
            "status": result["status"],
            "element_count": result.get("element_count", 0),
            "summary": result.get("summary", ""),
            "concepts": result.get("concepts", {}),
            "timestamp": result.get("timestamp", datetime.utcnow().isoformat())
        }

        # Add system message with processing result
        if state.conversation_state:
            if result["status"] == "complete":
                state.conversation_state.add_message(
                    MessageType.SYSTEM,
                    f"Document processed successfully. Extracted {result.get('element_count', 0)} elements."
                )
            else:
                state.conversation_state.add_message(
                    MessageType.SYSTEM,
                    f"Document processing failed: {result.get('error', 'Unknown error')}"
                )

    except Exception as e:
        logger.error(f"Error in document processing node: {str(e)}", exc_info=True)
        if state.conversation_state:
            state.conversation_state.add_message(
                MessageType.SYSTEM,
                f"Document processing failed due to an error: {str(e)}"
            )

    return state

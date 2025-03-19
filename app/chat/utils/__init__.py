"""
Utilities for the LangGraph-based PDF RAG system.
Modularized for better organization and maintainability.
"""

# Core utilities for document processing
from .tokenization import (
    get_tokenizer,
    OpenAITokenizerWrapper
)

# Extraction utilities for technical term processing
from .extraction import (
    extract_technical_terms,
    extract_concept_relationships,
    extract_hierarchy_relationships,
    find_best_term_match,
    RelationType
)

# Document processing utilities
from .processing import (
    process_markdown_text,
    create_markdown_chunker,
    normalize_metadata_for_vectorstore,
    normalize_pdf_id,
    detect_content_types
)

# File system utilities
from .document import (
    generate_document_summary,
    create_directory_if_not_exists,
    generate_file_hash
)

# LangGraph-specific helpers
from .langgraph_helpers import (
    extract_technical_terms_simple,
    format_conversation_for_llm,
    parse_citations_from_text,
    elements_to_context,
    create_empty_graph_state
)

# Setup logging
def setup_logging(pdf_id: str, output_dir: str = "output") -> None:
    """
    Set up logging for document processing.

    Args:
        pdf_id: Document identifier
        output_dir: Output directory
    """
    import logging
    import os
    from pathlib import Path

    log_dir = Path(output_dir) / pdf_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "processing.log"

    # Configure logger
    logger = logging.getLogger()
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler if it doesn't exist
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        logger.addHandler(handler)

def validate_content_element_class():
    """
    Validate that the ContentElement class has the expected structure.
    This helps identify attribute mismatches before they cause runtime errors.
    """
    from app.chat.types import ContentElement, ContentType
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Create a test instance
        test_element = ContentElement(
            element_id="test",
            content="Test content",
            content_type=ContentType.TEXT,
            pdf_id="test_pdf",
            metadata={}
        )

        # Check required attributes
        required_attrs = [
            'element_id', 'content', 'content_type', 'pdf_id', 'metadata'
        ]

        for attr in required_attrs:
            if not hasattr(test_element, attr):
                logger.error(f"ContentElement class missing required attribute: {attr}")
                return False

        # Verify content_type is an enum or has expected structure
        if hasattr(test_element.content_type, 'value'):
            logger.info(f"content_type has value attribute: {test_element.content_type.value}")
        else:
            logger.warning(f"content_type has no 'value' attribute: {type(test_element.content_type)}")

        logger.info("ContentElement class validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating ContentElement class: {str(e)}")
        return False        

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
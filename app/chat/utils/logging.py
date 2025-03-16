"""
Logging utilities for document processing.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from .document import create_directory_if_not_exists
from .extraction import SPACY_AVAILABLE

__all__ = [
    "setup_logging",
    "get_processor_metrics"
]

logger = logging.getLogger(__name__)


def setup_logging(pdf_id: str, output_dir: str = "output") -> None:
    """
    Set up logging for document processing with enhanced error tracking.

    Args:
        pdf_id: Document identifier
        output_dir: Base output directory
    """
    try:
        log_dir = Path(output_dir) / pdf_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"processing_{timestamp}.log"

        # Create file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)

        # Create formatter with detailed information
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
                for h in root_logger.handlers):
            root_logger.addHandler(handler)
            logger.info(f"Added log handler: {log_file}")
    except Exception as e:
        print(f"Logging setup error: {e}")
        logger.error(f"Logging setup error: {e}")


def get_processor_metrics() -> Dict[str, Any]:
    """
    Get comprehensive metrics about document processing.

    Returns:
        Dictionary with document processing metrics
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
        "spacy_available": SPACY_AVAILABLE,
        "tokenizer_info": {
            "default_model": "cl100k_base",
            "openai_wrapper_available": True
        },
        "system_info": {
            "platform": os.environ.get("PLATFORM", "unknown"),
            "cpu_count": os.cpu_count(),
            "max_ram": os.environ.get("MAX_RAM", "unknown")
        },
        "application_info": {
            "module_version": "1.0.0",
            "utils_modularized": True
        }
    }

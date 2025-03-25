"""
FastAPI application for Docling PDF processing.
Provides API endpoints for n8n integration.
"""

import os
import logging
from typing import Optional
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

from src.document_fetcher import DocumentProcessor
from src.types import ProcessingConfig, ProcessingResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Docling n8n API",
    description="API for processing PDFs with Docling and integrating with n8n",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_openai_client():
    """Get OpenAI client with API key from environment."""
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not found in environment!")
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    return AsyncOpenAI(api_key=OPENAI_API_KEY)

@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {"status": "Docling n8n API is running"}

@app.post("/extract", response_model=ProcessingResponse)
async def extract_document(
    pdf_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    extract_technical_terms: bool = Form(True),
    extract_procedures: bool = Form(True),
    extract_relationships: bool = Form(True),
    process_images: bool = Form(True),
    process_tables: bool = Form(True),
    openai_client: AsyncOpenAI = Depends(get_openai_client)
):
    """
    Extract content from a PDF document.
    
    Parameters:
    - pdf_id: Optional identifier for the PDF (will be generated if not provided)
    - file: PDF file to process
    - extract_technical_terms: Whether to extract technical terms
    - extract_procedures: Whether to extract procedures and parameters
    - extract_relationships: Whether to extract concept relationships
    - process_images: Whether to process images
    - process_tables: Whether to process tables
    
    Returns:
    - ProcessingResponse with extracted content and metadata
    """
    logger.info(f"Received extract request for file: {file.filename}")
    
    # Generate PDF ID if not provided
    if not pdf_id:
        pdf_id = f"doc_{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated PDF ID: {pdf_id}")
    
    try:
        # Read file content
        content = await file.read()
        
        # Create processing config
        config = ProcessingConfig(
            pdf_id=pdf_id,
            extract_technical_terms=extract_technical_terms,
            extract_procedures=extract_procedures,
            extract_relationships=extract_relationships,
            process_images=process_images,
            process_tables=process_tables,
        )
        
        # Process document
        processor = DocumentProcessor(
            pdf_id=pdf_id,
            config=config,
            openai_client=openai_client
        )
        
        # Process document and get results
        result = await processor.process_document(content)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

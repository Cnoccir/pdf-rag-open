# Docling n8n API

A FastAPI service that wraps the Docling PDF parser for n8n integration.

## Overview

This API provides an HTTP interface to the powerful Docling PDF processing pipeline. It extracts rich content from PDF documents including:

- Text content as Markdown
- Tables with structure preserved
- Images with metadata
- Technical terms and domain-specific vocabulary
- Procedures and parameters
- Concept relationships

The extracted data is returned in a structured format ready for use in n8n workflows or for ingestion into vector stores like Qdrant or document stores like MongoDB.

## Getting Started

### Prerequisites

- Python 3.8+
- Tesseract OCR
- OpenAI API key (for technical term extraction)

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Running the API

Start the FastAPI service:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker

Build and run using Docker:

```bash
docker build -t docling-n8n-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_api_key_here docling-n8n-api
```

## API Endpoints

### POST /extract

Extract content from a PDF document.

**Parameters:**
- `pdf_id` (form field, optional): Identifier for the PDF
- `file` (file upload): PDF file to process
- `extract_technical_terms` (form field, optional): Whether to extract technical terms
- `extract_procedures` (form field, optional): Whether to extract procedures and parameters
- `extract_relationships` (form field, optional): Whether to extract concept relationships
- `process_images` (form field, optional): Whether to process images
- `process_tables` (form field, optional): Whether to process tables

**Response:**
JSON object with the following structure:
- `pdf_id`: Document identifier
- `markdown`: Extracted markdown content
- `tables`: Array of extracted tables
- `images`: Array of image paths
- `metadata`: Processing metadata
- `technical_terms`: Array of extracted technical terms
- `procedures`: Array of extracted procedures
- `parameters`: Array of extracted parameters
- `concept_relationships`: Array of concept relationships
- `processing_time`: Processing time in seconds

## Integration with n8n

This API is designed to be used with the custom n8n node `DoclingExtractor` that's included in the companion project.

## License

MIT

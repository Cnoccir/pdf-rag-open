# n8n-nodes-docling

This is an n8n community node that integrates the Docling PDF parser with n8n workflows.

The node allows you to extract rich content from PDF documents including text (as Markdown), tables, images, and structured metadata. The extracted data can be prepared for ingestion into vector stores like Qdrant or document stores like MongoDB.

## Prerequisites

- n8n (version 0.214.0 or later recommended)
- Docling n8n API service running (see companion project)

## Installation

Follow these steps to install this custom node in your n8n instance:

### Local Installation (Development)

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/username/n8n-nodes-docling.git
   ```

2. Navigate to the project directory:
   ```bash
   cd n8n-nodes-docling
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Build the project:
   ```bash
   npm run build
   ```

5. Link the package to your n8n installation:
   ```bash
   npm link
   cd ~/.n8n
   npm link n8n-nodes-docling
   ```

6. Restart n8n

### Installation via npm (Production)

Once published to npm, you can install directly in your n8n instance:

```bash
cd ~/.n8n
npm install n8n-nodes-docling
```

Then restart n8n.

## Usage

After installation, the "Docling Extractor" node will be available in the n8n nodes panel under the "Transform" category.

### Node Configuration

- **Docling API URL**: URL of the Docling extraction API (e.g., http://localhost:8000/extract)
- **PDF Document**: Name of the binary property containing the PDF file
- **PDF ID**: Optional identifier for the PDF (will be generated if not provided)
- **Options**:
  - Extract Technical Terms: Whether to extract technical terms
  - Extract Procedures: Whether to extract procedures and parameters
  - Extract Relationships: Whether to extract concept relationships
  - Process Images: Whether to process images
  - Process Tables: Whether to process tables
- **Output Format**:
  - Combined Object: Return all data in a single JSON object
  - Separate Items: Return separate items for markdown, tables, images, etc.
- **Prepare for Qdrant**: Prepare the extracted data for ingestion into Qdrant
- **Prepare for MongoDB**: Prepare the extracted data for ingestion into MongoDB

### Example Workflow

1. Use an HTTP Request node or a local file node to get a PDF document
2. Connect it to the Docling Extractor node
3. Configure the Docling Extractor node with your API URL and options
4. Process the extracted data as needed in your workflow
5. Optionally, use the prepared data for Qdrant or MongoDB to store in your vector/document database

## Development

### Build

```bash
npm run build
```

### Lint

```bash
npm run lint
```

### Publish

```bash
npm publish
```

## License

MIT

"""
PDF Generator utility for testing.
Provides functions to create sample PDFs with technical content for testing purposes.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

logger = logging.getLogger(__name__)

def create_test_pdf(output_path=None, content_type="technical"):
    """
    Create a sample PDF for testing.
    
    Args:
        output_path: Path to save the PDF (if None, a temporary file will be created)
        content_type: Type of content to include (technical, general, or mixed)
        
    Returns:
        Path to the generated PDF file
    """
    # Create output directory if it doesn't exist
    if output_path is None:
        test_files_dir = Path(__file__).parent.parent / "test_files"
        test_files_dir.mkdir(exist_ok=True)
        filename = f"sample_{content_type}_{uuid.uuid4().hex[:8]}.pdf"
        output_path = test_files_dir / filename
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create document content
    content = []
    
    # Title
    if content_type == "technical":
        content.append(Paragraph("Technical Systems Documentation", title_style))
        content.append(Spacer(1, 12))
        content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        content.append(Spacer(1, 24))
        
        # Introduction
        content.append(Paragraph("1. Introduction", heading_style))
        content.append(Paragraph(
            "This technical document describes the architecture and implementation of a distributed "
            "system for processing and analyzing large volumes of data. The system consists of "
            "multiple interconnected components that work together to provide real-time analytics "
            "and data visualization capabilities.", normal_style))
        content.append(Spacer(1, 12))
        
        # System Architecture
        content.append(Paragraph("2. System Architecture", heading_style))
        content.append(Paragraph(
            "The system follows a microservices architecture pattern with the following key components:", 
            normal_style))
        content.append(Spacer(1, 6))
        
        # Component list
        components = [
            "Data Ingestion Service: Handles incoming data streams from various sources",
            "Storage Service: Manages data persistence across distributed databases",
            "Processing Engine: Performs data transformations and analytics",
            "API Gateway: Provides unified access to all system capabilities",
            "Visualization Dashboard: Presents results in an intuitive interface"
        ]
        
        for component in components:
            content.append(Paragraph(f"• {component}", normal_style))
        
        content.append(Spacer(1, 12))
        
        # Technical Details
        content.append(Paragraph("3. Technical Details", heading_style))
        
        # Data Flow Section
        content.append(Paragraph("3.1 Data Flow", subheading_style))
        content.append(Paragraph(
            "Data flows through the system in the following sequence:", normal_style))
        content.append(Spacer(1, 6))
        content.append(Paragraph(
            "1. Raw data is received by the Data Ingestion Service via Kafka streams. "
            "2. The data is validated, transformed into a standardized format, and stored. "
            "3. The Processing Engine retrieves data batches for analysis. "
            "4. Results are stored in a time-series database. "
            "5. The API Gateway exposes endpoints for querying the processed data. "
            "6. The dashboard visualizes the results in real-time.", normal_style))
        content.append(Spacer(1, 12))
        
        # Performance Metrics
        content.append(Paragraph("3.2 Performance Metrics", subheading_style))
        content.append(Paragraph(
            "The system is designed to handle the following workloads:", normal_style))
        content.append(Spacer(1, 6))
        
        # Table of performance metrics
        data = [
            ['Component', 'Throughput', 'Latency', 'Availability'],
            ['Data Ingestion', '100K events/sec', '<50ms', '99.99%'],
            ['Storage Service', '10K writes/sec', '<100ms', '99.999%'],
            ['Processing Engine', '50K records/sec', '<200ms', '99.9%'],
            ['API Gateway', '5K requests/sec', '<20ms', '99.99%']
        ]
        
        table = Table(data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 12))
        
        # Implementation Details
        content.append(Paragraph("4. Implementation Details", heading_style))
        content.append(Paragraph("4.1 Technology Stack", subheading_style))
        content.append(Paragraph(
            "The system utilizes the following technologies:", normal_style))
        content.append(Spacer(1, 6))
        
        tech_stack = [
            "Programming Languages: Python, Java, Go",
            "Messaging: Apache Kafka, RabbitMQ",
            "Storage: PostgreSQL, Redis, Elasticsearch",
            "Processing: Apache Spark, Flink",
            "Containerization: Docker, Kubernetes",
            "Monitoring: Prometheus, Grafana",
            "API: RESTful services, GraphQL"
        ]
        
        for tech in tech_stack:
            content.append(Paragraph(f"• {tech}", normal_style))
        
        content.append(Spacer(1, 12))
        
        # Code Example
        content.append(Paragraph("4.2 Code Example", subheading_style))
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=8,
            spaceBefore=4,
            spaceAfter=4,
            leftIndent=36,
            rightIndent=36,
            backColor=colors.lightgrey
        )
        
        code_example = """
def process_data_batch(batch_id, data):
    \"\"\"Process a batch of incoming data.\"\"\"
    results = []
    try:
        # Transform data
        transformed = transform_data(data)
        
        # Apply analytics models
        for item in transformed:
            analysis = apply_models(item)
            results.append(analysis)
            
        # Store results
        store_batch_results(batch_id, results)
        
        return {
            "batch_id": batch_id,
            "processed_count": len(results),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {str(e)}")
        return {
            "batch_id": batch_id,
            "status": "error",
            "error": str(e)
        }
"""
        content.append(Paragraph(code_example, code_style))
        content.append(Spacer(1, 12))
        
        # Deployment
        content.append(Paragraph("5. Deployment Architecture", heading_style))
        content.append(Paragraph(
            "The system is deployed across multiple availability zones to ensure "
            "high availability and disaster recovery capabilities. Each component "
            "is containerized and orchestrated using Kubernetes, with autoscaling "
            "policies based on CPU utilization and request volume.", normal_style))
        content.append(Spacer(1, 12))
        
        # Conclusion
        content.append(Paragraph("6. Conclusion", heading_style))
        content.append(Paragraph(
            "This documentation provides an overview of the system architecture "
            "and implementation. For more detailed information, please refer to "
            "the component-specific documentation and the API references.", normal_style))
        
    elif content_type == "general":
        # General content for non-technical PDF
        content.append(Paragraph("Company Overview", title_style))
        # ... add general content here
    else:  # mixed
        # Mix of technical and general content
        content.append(Paragraph("Project Documentation", title_style))
        # ... add mixed content here
    
    # Build PDF
    doc.build(content)
    logger.info(f"Generated test PDF at {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    # Generate a sample PDF when run directly
    logging.basicConfig(level=logging.INFO)
    pdf_path = create_test_pdf()
    print(f"Created sample PDF at: {pdf_path}")

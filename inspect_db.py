#!/usr/bin/env python
"""
Script to examine PDF data in the database for a specific PDF ID.
This version includes proper Flask application context handling.
"""

import json
import sys
from pprint import pprint

# Add these imports for your application
from app.web.db import db
from app.web.db.models import Pdf
from app.web import create_app  # Import the app factory function

# The PDF ID to examine
PDF_ID = "7d4cdabc-90a2-4eae-b8fe-00a29171d225"

def inspect_pdf_data():
    """Retrieve and display the PDF data, with special focus on metadata."""
    try:
        # Create Flask app and set up application context
        app = create_app()

        # Use the app context to access the database
        with app.app_context():
            # Query the database
            pdf = db.session.execute(db.select(Pdf).filter_by(id=PDF_ID)).scalar_one_or_none()

            if not pdf:
                print(f"PDF with ID {PDF_ID} not found in the database.")
                return

            # Print basic PDF information
            print("\n=== PDF RECORD INFORMATION ===")
            print(f"ID: {pdf.id}")
            print(f"Name: {pdf.name}")
            print(f"User ID: {pdf.user_id}")
            print(f"Description: {pdf.description}")
            print(f"Category: {pdf.category}")
            print(f"Processed: {pdf.processed}")
            print(f"Error: {pdf.error}")
            print(f"Created At: {pdf.created_at}")
            print(f"Updated At: {pdf.updated_at}")

            # Get metadata as dictionary
            metadata = pdf.get_metadata()

            print("\n=== METADATA CONTENT ===")
            if not metadata:
                print("No metadata found (document_meta is empty or invalid JSON)")
            else:
                print(f"Metadata size: {len(json.dumps(metadata))} characters")

                # Check for document summary
                if 'document_summary' in metadata:
                    print("\n=== DOCUMENT SUMMARY ===")
                    summary = metadata['document_summary']

                    # Print key fields
                    for key in ['title', 'primary_concepts', 'key_insights', 'section_structure', 'generated_at']:
                        if key in summary:
                            value = summary[key]
                            if isinstance(value, list) and len(value) > 5:
                                print(f"{key}: {value[:5]} ... ({len(value)} items)")
                            else:
                                print(f"{key}: {value}")
                        else:
                            print(f"{key}: Not found")
                else:
                    print("No document_summary found in metadata")

                # Check for predicted category
                if 'predicted_category' in metadata:
                    print(f"\nPredicted category: {metadata['predicted_category']}")
                else:
                    print("\nNo predicted_category found in metadata")

                # Print all top-level keys in metadata
                print("\nAll metadata keys:")
                print(", ".join(metadata.keys()))

                # Pretty print the full metadata if it's not too large
                metadata_str = json.dumps(metadata, indent=2)
                if len(metadata_str) < 10000:  # Limit to 10KB to avoid flooding the console
                    print("\n=== FULL METADATA CONTENT ===")
                    print(metadata_str)
                else:
                    print(f"\nFull metadata is too large to display ({len(metadata_str)} characters)")

                    # Print just the document_summary part if available
                    if 'document_summary' in metadata:
                        print("\n=== DOCUMENT SUMMARY DETAIL ===")
                        print(json.dumps(metadata['document_summary'], indent=2))

    except Exception as e:
        print(f"Error retrieving PDF data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_pdf_data()
    sys.exit(0)

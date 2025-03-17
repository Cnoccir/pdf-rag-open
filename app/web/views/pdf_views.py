from flask import Blueprint, g, jsonify, send_file, request, current_app
import io
import logging
from werkzeug.exceptions import Unauthorized
from app.web.hooks import login_required, handle_file_upload, load_model
from app.web.db.models import Pdf, Conversation
from app.web.tasks.embeddings import process_document
from app.web import files
from app.web.db import db
from app.chat.types import ProcessingConfig, ResearchManager
# Update import to use the new location
from app.chat.vector_stores import TechDocVectorStore, get_vector_store
import uuid
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

bp = Blueprint("pdf", __name__, url_prefix="/api/pdfs")

@bp.route("/", methods=["GET"])
@login_required
def list():
    pdfs = Pdf.query.filter_by(user_id=g.user.id, is_deleted=False).all()
    return Pdf.as_dicts(pdfs)  # Keep original format

@bp.route("/<string:pdf_id>/serve", methods=["GET"])
@login_required
@load_model(Pdf)
def serve_pdf(pdf):
    if pdf.user_id != g.user.id:
        return jsonify({"error": "Unauthorized"}), 403

    pdf_content = files.download_file_content(pdf.id)
    if pdf_content is None:
        return jsonify({"error": "PDF not found"}), 404

    return send_file(
        io.BytesIO(pdf_content),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'{pdf.name}'
    )

@bp.route("/<string:pdf_id>/content", methods=["GET"])
@login_required
@load_model(Pdf)
def get_pdf_content(pdf):
    """Endpoint for viewing PDF content (for thumbnails)"""
    if pdf.user_id != g.user.id:
        return jsonify({"error": "Unauthorized"}), 403

    pdf_content = files.download_file_content(pdf.id)
    if pdf_content is None:
        return jsonify({"error": "PDF not found"}), 404

    return send_file(
        io.BytesIO(pdf_content),
        mimetype='application/pdf',
        as_attachment=False,
        download_name=f'{pdf.name}'
    )

@bp.route("/", methods=["POST"])
@login_required
@handle_file_upload
def upload_file(file_id, file_path, file_name):
    pdf_id = str(uuid.uuid4())
    res, status_code = files.upload(file_path, pdf_id)
    if status_code >= 400:
        return res, status_code

    # Get additional metadata from request
    data = request.form
    description = data.get('description', '')
    category = data.get('category', 'general')

    pdf = Pdf.create(
        id=pdf_id,
        name=file_name,
        user_id=g.user.id,
        description=description,
        category=category
    )

    process_document.delay(pdf.id)

    return pdf.as_dict()  # Keep original format

@bp.route("/<string:pdf_id>", methods=["GET"])
@login_required
@load_model(Pdf)
def show(pdf):
    """
    Enhanced endpoint to show PDF details with summary information.
    """
    if pdf.user_id != g.user.id:
        return jsonify({"error": "Unauthorized"}), 403

    # Get PDF metadata with summary info
    metadata = pdf.get_metadata() if hasattr(pdf, 'get_metadata') else {}

    # Extract summary info - check both possible locations
    summary_info = metadata.get('document_summary', {})
    if not summary_info and 'summary' in metadata:
        summary_info = metadata.get('summary', {})

    # Additional fallbacks for summary data
    if not summary_info:
        # Try looking in other potential metadata locations
        for key in ['documentSummary', 'summary_data', 'doc_summary']:
            if key in metadata and metadata[key]:
                summary_info = metadata[key]
                break

    # Log what we found for debugging
    logger.info(f"Summary info for PDF {pdf.id}: {summary_info}")

    # Get category information
    category_info = {
        'predicted_category': metadata.get('predicted_category', 'general'),
        'current_category': pdf.category
    }

    # Truncate summary fields for display - Fix the isinstance check
    if 'key_insights' in summary_info and isinstance(summary_info['key_insights'], type([])):
        summary_info['key_insights'] = summary_info['key_insights'][:3]  # Limit to top 3 insights

    if 'primary_concepts' in summary_info and isinstance(summary_info['primary_concepts'], type([])):
        summary_info['primary_concepts'] = summary_info['primary_concepts'][:5]  # Limit to top 5 concepts

    # Ensure primary_concepts is not empty
    if not summary_info.get('primary_concepts'):
        # Try to extract from metadata.predicted_category
        if metadata.get('predicted_category'):
            summary_info['primary_concepts'] = [metadata.get('predicted_category')]

    # Build the response with PDF data included in metadata
    pdf_dict = pdf.as_dict()

    # Make sure metadata includes summary
    if 'metadata' in pdf_dict and not pdf_dict['metadata'].get('document_summary'):
        pdf_dict['metadata']['document_summary'] = summary_info

    response = {
        "pdf": pdf_dict,
        "download_url": f"/api/pdfs/{pdf.id}/serve",
        "summary": summary_info,
        "category_info": category_info
    }

    return jsonify(response)

@bp.route("/<string:pdf_id>/trigger-embedding", methods=["POST"])
@login_required
def trigger_embedding(pdf_id):
    pdf = Pdf.query.get(pdf_id)
    if not pdf:
        return jsonify({"error": "PDF not found"}), 404

    if pdf.user_id != g.user.id:
        return jsonify({"error": "Unauthorized"}), 403

    if pdf.processed:
        pdf.update(processed=False, error=None)  # Reset the processed flag

    process_document.delay(pdf_id)

    return jsonify({"message": "Embedding process triggered successfully"}), 200

@bp.route("/<string:pdf_id>", methods=["PATCH"])
@login_required
@load_model(Pdf)
def update(pdf):
    """Update PDF metadata"""
    if pdf.user_id != g.user.id:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    allowed_fields = ['description', 'category']
    update_data = {k: v for k, v in data.items() if k in allowed_fields}

    if update_data:
        pdf.update(**update_data)

    return pdf.as_dict()  # Keep original format

@bp.route("/<string:pdf_id>/summary", methods=["GET"])
@login_required
@load_model(Pdf)
def get_summary(pdf):
    """
    Get comprehensive document summary.
    """
    if pdf.user_id != g.user.id:
        return jsonify({"error": "Unauthorized"}), 403

    summary = {}
    metadata = pdf.get_metadata() if hasattr(pdf, 'get_metadata') else None
    if metadata and 'document_summary' in metadata:
        summary = metadata['document_summary']

    return jsonify({
        "pdf_id": pdf.id,
        "name": pdf.name,
        "summary": summary
    })

@bp.route("/<string:pdf_id>/category", methods=["PUT"])
@login_required
@load_model(Pdf)
def update_category(pdf):
    """
    Update PDF category manually.
    """
    if pdf.user_id != g.user.id:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    if not data or 'category' not in data:
        return jsonify({"error": "Category is required"}), 400

    # Valid categories
    valid_categories = ["general", "tridium", "honeywell", "johnson_controls"]

    category = data['category']
    if category not in valid_categories:
        return jsonify({"error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"}), 400

    # Update the category
    pdf.category = category
    pdf.save()

    return jsonify({
        "message": "Category updated successfully",
        "pdf": pdf.as_dict()
    })

# Update original handler to include summary information
@bp.route("/<string:pdf_id>", methods=["GET", "PATCH", "DELETE"])
@login_required
@load_model(Pdf)
def handle_pdf(pdf):
    """Handle PDF operations including GET, PATCH, and DELETE"""
    if pdf.user_id != g.user.id:
        return jsonify({"error": "Unauthorized"}), 403

    if request.method == "GET":
        # Extract summary and category information using get_metadata()
        summary_info = {}
        category_info = {}
        metadata = pdf.get_metadata() if hasattr(pdf, 'get_metadata') else None

        if metadata:
            if 'document_summary' in metadata:
                summary_info = metadata['document_summary']
            if 'predicted_category' in metadata:
                category_info = {
                    'predicted_category': metadata['predicted_category'],
                    'current_category': pdf.category
                }

        response = {
            "pdf": pdf.as_dict(),
            "download_url": f"/api/pdfs/{pdf.id}/serve",
            "summary": summary_info,
            "category_info": category_info
        }
        return jsonify(response)

    elif request.method == "PATCH":
        data = request.get_json()
        allowed_fields = ['description', 'category']
        update_data = {k: v for k, v in data.items() if k in allowed_fields}

        if update_data:
            pdf.update(**update_data)
        return pdf.as_dict()

    elif request.method == "DELETE":
        try:
            # Create proper configuration
            config = ProcessingConfig(
                pdf_id=pdf.id,
                embedding_model="text-embedding-3-small",
                embedding_dimensions=1536
            )

            # Create ResearchManager instance
            research_manager = ResearchManager(primary_pdf_id=pdf.id)

            # Create store instance with both config and research manager
            # Use get_vector_store() instead of creating a new instance
            store = get_vector_store()

            try:
                # Create a coroutine and run it
                delete_coroutine = store.delete_vectors(pdf.id)
                if delete_coroutine is not None:
                    asyncio.run(delete_coroutine)
                else:
                    logger.warning(f"No deletion coroutine returned for PDF {pdf.id}")
            except Exception as vector_err:
                logger.error(f"Vector deletion error: {str(vector_err)}")
                # Continue with soft delete even if vector deletion fails

            # Soft delete the PDF record
            pdf.is_deleted = True
            db.session.commit()

            return jsonify({"message": "Document successfully deleted."}), 200

        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return jsonify({"error": f"Failed to delete document: {str(e)}"}), 500
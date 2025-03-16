import json
from app.web.db import db
from .base import BaseModel
from datetime import datetime
import shortuuid

class Pdf(BaseModel):
    id: str = db.Column(
        db.String(22), primary_key=True, default=lambda: shortuuid.uuid()
    )
    name: str = db.Column(db.String(80), nullable=False)
    user_id: int = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    user = db.relationship("User", back_populates="pdfs")
    is_deleted: bool = db.Column(db.Boolean, default=False, nullable=False)
    processed: bool = db.Column(db.Boolean, default=False)
    error: str = db.Column(db.Text, nullable=True)
    description: str = db.Column(db.Text, nullable=True)
    category: str = db.Column(db.String(50), default='general', nullable=False)
    created_at: datetime = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at: datetime = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Use a different name that won't conflict
    document_meta: str = db.Column(db.Text, nullable=True)

    conversations = db.relationship(
        "Conversation",
        back_populates="pdf",
        order_by="desc(Conversation.created_on)",
    )

    def get_metadata(self):
        """Get metadata as a dictionary"""
        if self.document_meta:
            try:
                return json.loads(self.document_meta)
            except json.JSONDecodeError:
                return {}
        return {}

    def update_metadata(self, new_metadata):
        """
        Update PDF metadata while preserving existing metadata.
        Adds new data to the metadata JSON field without overwriting existing values.

        Args:
            new_metadata: Dictionary with new metadata to add/update

        Returns:
            Updated metadata dictionary
        """
        current = self.get_metadata()
        current.update(new_metadata)
        self.document_meta = json.dumps(current)
        db.session.commit()
        return current

    def as_dict(self):
        result = {
            "id": self.id,
            "name": self.name,
            "user_id": self.user_id,
            "is_deleted": self.is_deleted,
            "processed": self.processed,
            "error": self.error,
            "description": self.description,
            "category": self.category,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.get_metadata()  # Include metadata in dict representation
        }
        return result

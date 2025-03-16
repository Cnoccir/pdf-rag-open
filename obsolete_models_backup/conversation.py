import uuid
from app.web.db import db
from .base import BaseModel
from sqlalchemy import Boolean, Column, String, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

class Conversation(BaseModel):
    id: str = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_on = db.Column(db.DateTime, server_default=db.func.now())
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    title: str = db.Column(db.String(255), default="Untitled Conversation")
    summary: str = db.Column(db.Text, nullable=True)
    retriever: str = db.Column(db.String)
    memory: str = db.Column(db.String)
    llm: str = db.Column(db.String)
    pdf_id: int = db.Column(db.Integer, db.ForeignKey("pdf.id"), nullable=False)
    pdf = db.relationship("Pdf", back_populates="conversations")
    user_id: int = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    user = db.relationship("User", back_populates="conversations")
    messages = db.relationship(
        "Message", back_populates="conversation", order_by="Message.created_on"
    )
    is_deleted = db.Column(Boolean, default=False)

    # Rename json_metadata column to avoid conflicts with SQLAlchemy's metadata
    json_metadata = db.Column(JSON, nullable=True)

    # Rename property methods for consistency with json_metadata column
    @property
    def metadata_dict(self):
        """Property that provides access to json_metadata for backward compatibility."""
        return self.json_metadata

    @metadata_dict.setter
    def metadata_dict(self, value):
        """Setter that allows setting json_metadata for backward compatibility."""
        self.json_metadata = value

    def as_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "pdf_id": self.pdf_id,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "messages": [m.as_dict() for m in self.messages],
            # Use metadata key in API response for backward compatibility
            "metadata": self.json_metadata
        }

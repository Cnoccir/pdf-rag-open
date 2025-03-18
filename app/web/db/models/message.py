import json
import uuid
from datetime import datetime
from app.web.db import db
from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from .base import BaseModel
import logging
logger = logging.getLogger(__name__)

class Message(BaseModel):
    __tablename__ = "message"

    id = db.Column(db.String(), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_on = db.Column(db.DateTime, server_default=db.func.now())
    role = db.Column(db.String(), nullable=False)
    content = db.Column(db.String(), nullable=False)
    conversation_id = db.Column(db.String(), db.ForeignKey("conversation.id"), nullable=False)
    conversation = db.relationship("Conversation", back_populates="messages")
    meta_json = db.Column(db.Text, nullable=True, default="{}")

    def as_dict(self):
        """Convert message to dictionary with metadata"""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "metadata": self.msg_metadata,
            "created_on": self.created_on.isoformat()
        }

    @property
    def msg_metadata(self) -> dict:
        """Safely load metadata from JSON"""
        if not self.meta_json:
            return {}
        try:
            # Ensure we're getting a dictionary
            metadata = json.loads(self.meta_json)
            if not isinstance(metadata, dict):
                logger.warning(f"Metadata for message {self.id} is not a dictionary, converting")
                return {"raw_value": metadata}
            return metadata
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse metadata JSON for message {self.id}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error accessing metadata for message {self.id}: {str(e)}")
            return {}

    @msg_metadata.setter
    def msg_metadata(self, value: dict) -> None:
        """Store metadata as JSON with validation"""
        if value is None:
            self.meta_json = "{}"
            return

        if not isinstance(value, dict):
            logger.warning(f"Attempted to set non-dict metadata for message {self.id}")
            # Convert to dict if possible
            try:
                value = {"value": value}
            except:
                value = {}

        try:
            # Handle any non-serializable values
            self.meta_json = json.dumps(value, default=str)
        except TypeError as e:
            logger.warning(f"Failed to serialize metadata for message {self.id}: {str(e)}")
            # Fallback to string conversion for all values
            sanitized = {}
            for k, v in value.items():
                try:
                    # Test if key/value can be serialized
                    json.dumps({k: v})
                    sanitized[k] = v
                except:
                    sanitized[k] = str(v)
            self.meta_json = json.dumps(sanitized)
        except Exception as e:
            logger.error(f"Unexpected error setting metadata for message {self.id}: {str(e)}")
            self.meta_json = "{}"

    # For backward compatibility
    def get_metadata(self) -> dict:
        """Legacy method for metadata access"""
        return self.msg_metadata

    def set_metadata(self, metadata: dict) -> None:
        """Legacy method for metadata setting"""
        self.msg_metadata = metadata

    @classmethod
    def create(cls, conversation_id: str, role: str, content: str, metadata: dict = None) -> "Message":
        """Create a new message with proper validation"""
        try:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
            if metadata:
                # Store as JSON string in meta_json
                message.meta_json = json.dumps(metadata)
            db.session.add(message)
            db.session.commit()
            return message
        except Exception as e:
            db.session.rollback()
            raise e

    def as_lc_message(self) -> HumanMessage | AIMessage | SystemMessage | ToolMessage:
        """Convert to LangChain message with metadata"""
        metadata = self.msg_metadata
        additional_kwargs = metadata.get("additional_kwargs", {})

        # Map roles to message types
        if self.role == "user":
            return HumanMessage(
                content=self.content,
                additional_kwargs=additional_kwargs
            )
        elif self.role == "assistant":
            return AIMessage(
                content=self.content, 
                additional_kwargs=additional_kwargs
            )
        elif self.role == "system":
            return SystemMessage(
                content=self.content,
                additional_kwargs=additional_kwargs
            )
        elif self.role == "tool":
            # Extract tool-specific metadata
            tool_call_id = metadata.get("tool_call_id", "")
            name = metadata.get("name", "")

            return ToolMessage(
                content=self.content,
                tool_call_id=tool_call_id,
                name=name,
                additional_kwargs=additional_kwargs
            )
        else:
            # Fallback to human message for unknown roles
            return HumanMessage(
                content=self.content,
                additional_kwargs={"role": self.role, **additional_kwargs}
            )

    @classmethod
    def from_lc_message(cls, conversation_id: str, message: HumanMessage | AIMessage | SystemMessage | ToolMessage) -> "Message":
        """Create from a LangChain message"""
        # Map message types to roles
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            # Handle unknown message types
            role = "unknown"

        # Extract content
        content = message.content

        # Extract metadata
        metadata = getattr(message, "additional_kwargs", {}).copy()

        # Include message-specific metadata
        if isinstance(message, ToolMessage):
            metadata["tool_call_id"] = message.tool_call_id
            metadata["name"] = message.name

        # Include any other metadata attached to the message
        if hasattr(message, "metadata") and message.metadata:
            # Ensure we don't overwrite existing keys
            for key, value in message.metadata.items():
                if key not in metadata:
                    metadata[key] = value

        # Create and return the message using the create method
        try:
            return cls.create(
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to create message from LangChain message: {str(e)}")
            # Fallback to direct instance creation in case DB transaction fails
            instance = cls(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
            instance.msg_metadata = metadata
            return instance

    def save(self):
        """Save the message to the database"""
        db.session.add(self)
        db.session.commit()
        return self

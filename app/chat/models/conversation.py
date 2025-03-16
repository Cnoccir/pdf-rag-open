"""
Conversation state models for LangGraph architecture.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from uuid import uuid4
from enum import Enum, auto

class MessageType(str, Enum):
    """Message types in a conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class Message(BaseModel):
    """Message in a conversation"""
    type: MessageType  # "user", "assistant", "system", "tool"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation"""
        return {
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary representation"""
        if isinstance(data.get("created_at"), str):
            try:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                data["created_at"] = datetime.now()
        return cls(**data)

class Citation(BaseModel):
    """Citation for a message"""
    text: str
    document_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationState(BaseModel):
    """Full conversation state for LangGraph architecture"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "Untitled Conversation"
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, type: MessageType, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to the conversation"""
        message = Message(
            type=type,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation state to dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Create conversation state from dictionary representation"""
        # Handle ISO format dates
        for date_field in ["created_at", "updated_at"]:
            if isinstance(data.get(date_field), str):
                try:
                    data[date_field] = datetime.fromisoformat(data[date_field])
                except (ValueError, TypeError):
                    data[date_field] = datetime.now()
        
        # Handle messages
        if "messages" in data:
            data["messages"] = [
                Message.from_dict(msg) if isinstance(msg, dict) else msg
                for msg in data["messages"]
            ]
            
        return cls(**data)
    
    def get_message_history(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """Get formatted message history for sending to LLM"""
        messages = []
        for msg in self.messages:
            if not include_system and msg.type == MessageType.SYSTEM:
                continue
            messages.append({
                "role": "assistant" if msg.type == MessageType.ASSISTANT else 
                        "system" if msg.type == MessageType.SYSTEM else 
                        "tool" if msg.type == MessageType.TOOL else "user",
                "content": msg.content
            })
        return messages
    
    def to_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation for UI display"""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages),
            "pdf_id": self.metadata.get("pdf_id"),
            "research_mode": self.metadata.get("research_mode", False)
        }

class ConversationSummary(BaseModel):
    """Summary of a conversation for listing"""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    pdf_id: Optional[str] = None
    research_mode: bool = False
    
    @classmethod
    def from_state(cls, state: ConversationState) -> "ConversationSummary":
        """Create summary from conversation state"""
        return cls(
            id=state.id,
            title=state.title,
            created_at=state.created_at,
            updated_at=state.updated_at,
            message_count=len(state.messages),
            pdf_id=state.metadata.get("pdf_id"),
            research_mode=state.metadata.get("research_mode", False)
        )

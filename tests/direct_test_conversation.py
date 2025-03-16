"""
Direct test for ConversationState without going through app initialization
"""

import sys
import os
from datetime import datetime
from enum import Enum, auto
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4

# Directly define MessageType here to avoid imports
class MessageType(str, Enum):
    """Message types in a conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

# Directly define Message here to avoid imports
class Message(BaseModel):
    """Message in a conversation"""
    type: MessageType
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
        """Create a message from dictionary representation"""
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

# Directly define ConversationState here to avoid imports
class ConversationState(BaseModel):
    """State model for conversations in LangGraph architecture"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    messages: List[Message] = Field(default_factory=list)
    pdf_id: Optional[str] = None
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
        """Convert conversation to dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "pdf_id": self.pdf_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Create a conversation from dictionary representation"""
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            
        messages_data = data.pop("messages", [])
        conversation = cls(**data)
        
        for msg_data in messages_data:
            conversation.messages.append(Message.from_dict(msg_data))
            
        return conversation
    
    def get_formatted_history(self, include_system: bool = True) -> List[Dict[str, str]]:
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

def test_conversation_creation():
    """Test creating a conversation and adding messages"""
    # Create conversation
    conversation = ConversationState(
        title="Test Conversation",
        pdf_id="test-pdf-123",
        metadata={"test": "value"}
    )
    
    print(f"Created conversation with ID: {conversation.id}")
    print(f"Title: {conversation.title}")
    print(f"PDF ID: {conversation.pdf_id}")
    print(f"Metadata: {conversation.metadata}")
    
    # Add messages
    conversation.add_message(MessageType.SYSTEM, "You are a helpful assistant.")
    conversation.add_message(MessageType.USER, "Hello, can you help me?")
    conversation.add_message(MessageType.ASSISTANT, "Of course! How can I help you today?")
    
    print(f"Added {len(conversation.messages)} messages to the conversation")
    
    # Get formatted history
    formatted_history = conversation.get_formatted_history()
    print("\nFormatted history:")
    for msg in formatted_history:
        print(f"- {msg['role']}: {msg['content']}")
    
    # Test serialization/deserialization
    data_dict = conversation.to_dict()
    print("\nSerialized conversation to dictionary")
    
    recreated = ConversationState.from_dict(data_dict)
    print(f"Recreated conversation from dictionary. ID matches: {recreated.id == conversation.id}")
    
    return conversation

if __name__ == "__main__":
    print("Testing ConversationState model directly...")
    test_conversation = test_conversation_creation()
    print("\nTest completed successfully!")

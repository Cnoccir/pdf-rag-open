"""
Test script for validating the ConversationState model in our LangGraph migration.
This tests the core state model independently of other components.
"""

import os
import sys
import unittest
from datetime import datetime
from uuid import uuid4

# Add the parent directory to sys.path to import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chat.langgraph.state import GraphState, MessageType
from app.chat.models.conversation import ConversationState, Message


class TestConversationState(unittest.TestCase):
    """Test cases for the ConversationState model"""
    
    def test_conversation_creation(self):
        """Test creating a new conversation"""
        conversation = ConversationState(
            id=str(uuid4()),
            title="Test Conversation",
            messages=[],
            metadata={
                "created_by": "test_user",
                "research_mode": False
            }
        )
        
        self.assertIsNotNone(conversation.id)
        self.assertEqual(conversation.title, "Test Conversation")
        self.assertEqual(len(conversation.messages), 0)
        self.assertFalse(conversation.metadata.get("research_mode", False))
    
    def test_add_message(self):
        """Test adding messages to a conversation"""
        conversation = ConversationState(
            id=str(uuid4()),
            title="Test Conversation",
            messages=[]
        )
        
        # Add user message
        user_msg = Message(
            id=str(uuid4()),
            type=MessageType.USER,
            content="Hello, this is a test message",
            timestamp=datetime.utcnow().isoformat()
        )
        conversation.messages.append(user_msg)
        
        # Add assistant message
        asst_msg = Message(
            id=str(uuid4()),
            type=MessageType.ASSISTANT,
            content="I am responding to your test",
            timestamp=datetime.utcnow().isoformat()
        )
        conversation.messages.append(asst_msg)
        
        # Check messages
        self.assertEqual(len(conversation.messages), 2)
        self.assertEqual(conversation.messages[0].content, "Hello, this is a test message")
        self.assertEqual(conversation.messages[0].type, MessageType.USER)
        self.assertEqual(conversation.messages[1].content, "I am responding to your test")
        self.assertEqual(conversation.messages[1].type, MessageType.ASSISTANT)
    
    def test_get_message_history(self):
        """Test getting message history from a conversation"""
        conversation = ConversationState(
            id=str(uuid4()),
            title="Test Conversation",
            messages=[]
        )
        
        # Add several messages
        conversation.messages.append(Message(
            id=str(uuid4()),
            type=MessageType.USER,
            content="First user message",
            timestamp=datetime.utcnow().isoformat()
        ))
        
        conversation.messages.append(Message(
            id=str(uuid4()),
            type=MessageType.ASSISTANT,
            content="First assistant response",
            timestamp=datetime.utcnow().isoformat()
        ))
        
        conversation.messages.append(Message(
            id=str(uuid4()),
            type=MessageType.USER,
            content="Second user message",
            timestamp=datetime.utcnow().isoformat()
        ))
        
        conversation.messages.append(Message(
            id=str(uuid4()),
            type=MessageType.ASSISTANT,
            content="Second assistant response",
            timestamp=datetime.utcnow().isoformat()
        ))
        
        # Check history
        history = conversation.messages
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].content, "First user message")
        self.assertEqual(history[1].content, "First assistant response")
        self.assertEqual(history[2].content, "Second user message")
        self.assertEqual(history[3].content, "Second assistant response")
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization"""
        # Create original conversation
        original = ConversationState(
            id=str(uuid4()),
            title="Serialization Test",
            messages=[],
            metadata={"test_key": "test_value"}
        )
        
        # Add a message
        original.messages.append(Message(
            id=str(uuid4()),
            type=MessageType.USER,
            content="Test message for serialization",
            timestamp=datetime.utcnow().isoformat()
        ))
        
        # Convert to dict
        conv_dict = original.model_dump()
        
        # Recreate from dict
        recreated = ConversationState(**conv_dict)
        
        # Compare
        self.assertEqual(original.id, recreated.id)
        self.assertEqual(original.title, recreated.title)
        self.assertEqual(len(original.messages), len(recreated.messages))
        self.assertEqual(original.messages[0].content, recreated.messages[0].content)
        self.assertEqual(original.metadata.get("test_key"), recreated.metadata.get("test_key"))


if __name__ == "__main__":
    unittest.main()

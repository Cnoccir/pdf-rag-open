"""
Test script for the LangGraph-based memory system.

This script tests the conversation history tracking and memory management
in the new LangGraph architecture.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add app directory to path
sys.path.insert(0, os.path.abspath('.'))

# Import required modules
from app.chat.chat_manager import ChatManager
from app.chat.types import ChatArgs, ResearchMode
from app.chat.langgraph.state import GraphState, MessageType


async def test_conversation_memory():
    """Test conversation memory persistence and retrieval."""
    
    # Create a unique conversation ID for testing
    conversation_id = str(uuid.uuid4())
    pdf_id = "sample_pdf_1"  # Replace with an actual PDF ID from your system
    
    # Create chat args
    chat_args = ChatArgs(
        conversation_id=conversation_id,
        pdf_id=pdf_id,
        research_mode=ResearchMode.SINGLE,
        stream_enabled=False
    )
    
    # Create chat manager
    chat_manager = ChatManager(chat_args)
    print(f"Created chat manager with conversation ID: {conversation_id}")
    
    # Test a sequence of queries
    queries = [
        "What are the main components of the system?",
        "Can you explain how the LangGraph architecture works?",
        "Tell me more about conversation memory",
    ]
    
    for i, query in enumerate(queries):
        print(f"\n\n--- Query {i+1}: {query} ---")
        
        # Process query
        result = await chat_manager.aquery(query)
        
        # Print response
        print(f"Response: {result.get('response', 'No response')[:100]}...")
        
        # Print conversation history
        history = chat_manager.get_conversation_history()
        print(f"Conversation history has {len(history)} messages")
        
        # Wait a moment between queries
        await asyncio.sleep(1)
    
    # Print final conversation state
    print("\n\n--- Final Conversation State ---")
    if chat_manager.conversation_state:
        # Print messages
        print(f"Total messages: {len(chat_manager.conversation_state.messages)}")
        print(f"Technical concepts: {chat_manager.conversation_state.technical_concepts}")
        
        # Print last message
        if chat_manager.conversation_state.messages:
            last_msg = chat_manager.conversation_state.messages[-1]
            print(f"Last message: {last_msg.type.value} - {last_msg.content[:50]}...")
    else:
        print("No conversation state available")
    
    return {
        "conversation_id": conversation_id,
        "message_count": len(chat_manager.get_conversation_history())
    }


async def run_tests():
    """Run all tests."""
    print("=== Starting LangGraph Memory System Tests ===")
    
    # Test conversation memory
    result = await test_conversation_memory()
    print(f"\nTest completed. Conversation ID: {result['conversation_id']}")
    print(f"Total messages: {result['message_count']}")
    
    print("\n=== All Tests Completed ===")


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_tests())

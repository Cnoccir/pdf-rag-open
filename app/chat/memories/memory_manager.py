"""
Memory manager for LangGraph architecture.
Handles conversation state persistence and retrieval.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import aiofiles
from pathlib import Path
import uuid

from app.chat.langgraph.state import ConversationState, MessageType

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manager for conversation memory persistence and retrieval"""
    
    def __init__(self, storage_base: str = None):
        """Initialize memory manager with storage path"""
        # Use data directory in root by default
        self.storage_base = storage_base or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
            "data", 
            "conversations"
        )
        
        # Ensure storage directory exists
        os.makedirs(self.storage_base, exist_ok=True)
        logger.info(f"Memory manager initialized with storage at: {self.storage_base}")
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get conversation state by ID"""
        try:
            file_path = os.path.join(self.storage_base, f"{conversation_id}.json")
            
            if not os.path.exists(file_path):
                logger.warning(f"Conversation {conversation_id} not found")
                return None
                
            async with aiofiles.open(file_path, "r") as f:
                data = await f.read()
                conversation_data = json.loads(data)
                
                # Create new conversation state
                conv_state = ConversationState(
                    conversation_id=conversation_data.get("id", conversation_id),
                    title=conversation_data.get("title", "Untitled"),
                    pdf_id=conversation_data.get("pdf_id", ""),
                    metadata=conversation_data.get("metadata", {})
                )
                
                # Add messages
                for msg_data in conversation_data.get("messages", []):
                    msg_type = MessageType(msg_data.get("type", "user"))
                    msg_content = msg_data.get("content", "")
                    msg_metadata = msg_data.get("metadata", {})
                    
                    conv_state.add_message(msg_type, msg_content, msg_metadata)
                
                return conv_state
                
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
            return None
    
    async def save_conversation(self, conversation: ConversationState) -> bool:
        """Save conversation state"""
        try:
            file_path = os.path.join(self.storage_base, f"{conversation.conversation_id}.json")
            
            # Convert to dictionary
            data = {
                "id": conversation.conversation_id,
                "title": conversation.title,
                "pdf_id": conversation.pdf_id,
                "metadata": conversation.metadata,
                "messages": [
                    {
                        "type": msg.type,
                        "content": msg.content,
                        "metadata": msg.metadata,
                        "created_at": msg.created_at.isoformat() if hasattr(msg.created_at, "isoformat") else str(msg.created_at)
                    }
                    for msg in conversation.messages
                ],
                "updated_at": datetime.now().isoformat()
            }
            
            # Write to file
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2))
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.conversation_id}: {str(e)}")
            return False
    
    async def list_conversations(self, pdf_id: Optional[str] = None) -> List[ConversationState]:
        """
        List all available conversations, optionally filtered by PDF ID.
        
        Args:
            pdf_id: Optional PDF ID to filter by
            
        Returns:
            List of conversation states
        """
        try:
            conversations = []
            
            # Get all JSON files in the storage directory
            json_files = [f for f in os.listdir(self.storage_base) if f.endswith(".json")]
            
            for file_name in json_files:
                try:
                    # Extract conversation ID from filename
                    conversation_id = file_name.replace(".json", "")
                    
                    # Load the conversation
                    conversation = await self.get_conversation(conversation_id)
                    if not conversation:
                        continue
                        
                    # Filter by PDF ID if specified
                    if pdf_id and conversation.pdf_id != pdf_id:
                        continue
                        
                    conversations.append(conversation)
                        
                except Exception as e:
                    logger.warning(f"Error processing conversation file {file_name}: {str(e)}")
                    continue
            
            # Sort by updated_at (newest first)
            conversations.sort(key=lambda x: x.updated_at, reverse=True)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: ID of conversation to delete
            
        Returns:
            Success status
        """
        try:
            file_path = os.path.join(self.storage_base, f"{conversation_id}.json")
            
            if not os.path.exists(file_path):
                logger.warning(f"Conversation {conversation_id} not found for deletion")
                return False
                
            os.remove(file_path)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            return False
    
    async def create_conversation(
        self, 
        title: str = None, 
        pdf_id: str = None, 
        metadata: Dict[str, Any] = None,
        id: str = None
    ) -> ConversationState:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            pdf_id: PDF ID
            metadata: Additional metadata
            id: Optional conversation ID (generated if not provided)
            
        Returns:
            New conversation state
        """
        try:
            # Create conversation ID if not provided
            conversation_id = id or str(uuid.uuid4())
            
            # Create new conversation state
            conversation = ConversationState(
                conversation_id=conversation_id,
                title=title or "Untitled Conversation",
                pdf_id=pdf_id or "",
                metadata=metadata or {}
            )
            
            # Save the conversation
            success = await self.save_conversation(conversation)
            if not success:
                raise Exception("Failed to save new conversation")
                
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise e
"""
Memory manager for LangGraph architecture.
Handles conversation state persistence and retrieval.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiofiles
from pathlib import Path

from app.chat.models.conversation import ConversationState, ConversationSummary

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
                return ConversationState.from_dict(conversation_data)
                
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
            return None
            
    async def save_conversation(self, conversation: ConversationState) -> bool:
        """Save conversation state"""
        try:
            file_path = os.path.join(self.storage_base, f"{conversation.id}.json")
            
            # Update timestamp
            conversation.updated_at = datetime.now()
            
            # Convert to dictionary and save
            conversation_data = conversation.to_dict()
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(conversation_data, indent=2))
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.id}: {str(e)}")
            return False
            
    async def list_conversations(self) -> List[ConversationSummary]:
        """List all available conversations"""
        try:
            conversation_summaries = []
            
            # Get all JSON files in the storage directory
            json_files = [f for f in os.listdir(self.storage_base) if f.endswith(".json")]
            
            for file_name in json_files:
                try:
                    # Extract conversation ID from filename
                    conversation_id = file_name.replace(".json", "")
                    
                    # Load the conversation
                    conversation = await self.get_conversation(conversation_id)
                    if conversation:
                        # Create summary
                        summary = ConversationSummary.from_state(conversation)
                        conversation_summaries.append(summary)
                        
                except Exception as e:
                    logger.warning(f"Error processing conversation file {file_name}: {str(e)}")
                    continue
                    
            # Sort by updated_at (newest first)
            conversation_summaries.sort(key=lambda x: x.updated_at, reverse=True)
            
            return conversation_summaries
            
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            return []
            
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
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
            
    async def create_conversation(self, title: str = None, pdf_id: str = None, 
                                metadata: Dict[str, Any] = None) -> ConversationState:
        """Create a new conversation"""
        try:
            # Create new conversation state
            conversation = ConversationState(
                title=title or "Untitled Conversation",
                metadata=metadata or {}
            )
            
            # Set PDF ID if provided
            if pdf_id:
                conversation.metadata["pdf_id"] = pdf_id
                
            # Save the conversation
            success = await self.save_conversation(conversation)
            if not success:
                raise Exception("Failed to save new conversation")
                
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise e

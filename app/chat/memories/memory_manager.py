"""
Enhanced memory manager with robust error handling and recovery mechanisms.
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
import traceback
import sys

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
        """Get conversation state by ID with comprehensive error handling and recovery"""
        try:
            file_path = os.path.join(self.storage_base, f"{conversation_id}.json")

            if not os.path.exists(file_path):
                logger.warning(f"Conversation {conversation_id} not found at {file_path}")
                return None

            async with aiofiles.open(file_path, "r") as f:
                data = await f.read()
                conversation_data = json.loads(data)

                # Extract required fields with defaults
                conv_id = conversation_data.get("id", conversation_id)
                title = conversation_data.get("title", f"Conversation {conversation_id[:8]}")
                pdf_id = conversation_data.get("pdf_id", "")
                metadata = conversation_data.get("metadata", {})

                # Create new conversation state
                conv_state = ConversationState(
                    conversation_id=conv_id,
                    title=title,
                    pdf_id=pdf_id,
                    metadata=metadata
                )

                # Add messages if present
                for msg_data in conversation_data.get("messages", []):
                    try:
                        # Default to "user" if type is missing
                        msg_type_str = msg_data.get("type", "user")

                        # Handle both string and enum types
                        try:
                            msg_type = MessageType(msg_type_str)
                        except ValueError:
                            # If not a valid enum value, map it
                            type_mapping = {
                                "user": MessageType.USER,
                                "assistant": MessageType.AI,
                                "system": MessageType.SYSTEM,
                                "tool": MessageType.TOOL
                            }
                            msg_type = type_mapping.get(msg_type_str, MessageType.USER)

                        msg_content = msg_data.get("content", "")
                        msg_metadata = msg_data.get("metadata", {})

                        # Add message to conversation state
                        conv_state.add_message(msg_type, msg_content, msg_metadata)
                    except Exception as msg_error:
                        logger.error(f"Error adding message to conversation {conversation_id}: {str(msg_error)}")
                        logger.error(traceback.format_exc())
                        # Skip this message but continue processing others

                # Add system message if none exists
                has_system_message = any(msg.type == MessageType.SYSTEM for msg in conv_state.messages)
                if not has_system_message:
                    system_prompt = "You are an AI assistant specialized in answering questions about documents."
                    conv_state.add_message(MessageType.SYSTEM, system_prompt)

                return conv_state

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing conversation JSON for {conversation_id}: {str(e)}")
            logger.error(traceback.format_exc())

            # Create a new conversation rather than returning None
            logger.info(f"Creating a recovery conversation for {conversation_id}")
            return ConversationState(
                conversation_id=conversation_id,
                title=f"Recovered Conversation {conversation_id[:8]}",
                metadata={"recovered": True}
            )

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
            logger.error("Exception traceback:")
            traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in traceback_lines:
                logger.error(line.rstrip())

            # Create a new conversation as a last resort
            try:
                logger.info(f"Creating an emergency recovery conversation for {conversation_id}")
                return ConversationState(
                    conversation_id=conversation_id,
                    title=f"Emergency Recovery {conversation_id[:8]}",
                    metadata={"emergency_recovery": True, "error": str(e)}
                )
            except:
                return None

    async def save_conversation(self, conversation: ConversationState) -> bool:
        """Save conversation state with backup and error handling"""
        if not conversation or not conversation.conversation_id:
            logger.error("Cannot save conversation: Invalid conversation state")
            return False

        try:
            file_path = os.path.join(self.storage_base, f"{conversation.conversation_id}.json")

            # Create backup of existing file
            if os.path.exists(file_path):
                backup_path = f"{file_path}.bak"
                try:
                    os.replace(file_path, backup_path)
                    logger.debug(f"Created backup at {backup_path}")
                except Exception as backup_error:
                    logger.warning(f"Failed to create backup for {file_path}: {str(backup_error)}")

            # Ensure all types are JSON serializable
            # Handle messages with MessageType enums
            messages_data = []
            for msg in conversation.messages:
                try:
                    msg_type = msg.type.value if hasattr(msg.type, 'value') else str(msg.type)
                    msg_content = msg.content or ""
                    msg_metadata = msg.metadata or {}
                    created_at = msg.created_at.isoformat() if hasattr(msg.created_at, 'isoformat') else str(msg.created_at)

                    messages_data.append({
                        "type": msg_type,
                        "content": msg_content,
                        "metadata": msg_metadata,
                        "created_at": created_at
                    })
                except Exception as msg_error:
                    logger.error(f"Error serializing message: {str(msg_error)}")
                    # Skip this message but continue

            # Convert to dictionary with safe defaults
            data = {
                "id": conversation.conversation_id,
                "title": conversation.title or f"Conversation {conversation.conversation_id[:8]}",
                "pdf_id": conversation.pdf_id or "",
                "metadata": conversation.metadata or {},
                "messages": messages_data,
                "updated_at": datetime.now().isoformat()
            }

            # Write to file
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2, default=str))

            return True

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"Error saving conversation {conversation.conversation_id}: {str(e)}")
            logger.error("Exception traceback:")
            traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in traceback_lines:
                logger.error(line.rstrip())
            return False

    async def list_conversations(self, pdf_id: Optional[str] = None) -> List[ConversationState]:
        """
        List all available conversations, optionally filtered by PDF ID.
        Includes optimized loading and error handling.

        Args:
            pdf_id: Optional PDF ID to filter by

        Returns:
            List of conversation states
        """
        try:
            conversations = []

            # Get all JSON files in the storage directory
            try:
                json_files = [f for f in os.listdir(self.storage_base)
                             if f.endswith(".json") and not f.endswith(".bak")]
            except Exception as dir_error:
                logger.error(f"Error listing directory {self.storage_base}: {str(dir_error)}")
                json_files = []

            # Track errors for reporting
            error_count = 0

            for file_name in json_files:
                try:
                    # Extract conversation ID from filename
                    conversation_id = file_name.replace(".json", "")

                    # Try to load conversation first from cache
                    try:
                        file_path = os.path.join(self.storage_base, file_name)

                        # Load just the metadata first (faster)
                        async with aiofiles.open(file_path, "r") as f:
                            data = await f.read()
                            conversation_data = json.loads(data)

                            # Check if filtered by pdf_id
                            stored_pdf_id = conversation_data.get("pdf_id", "")
                            # Convert both to strings for comparison
                            if pdf_id and str(stored_pdf_id) != str(pdf_id):
                                continue

                            # Check if marked as deleted
                            metadata = conversation_data.get("metadata", {})
                            if metadata and metadata.get("is_deleted", False):
                                continue

                            # Create minimal conversation state without messages
                            conv_state = ConversationState(
                                conversation_id=conversation_data.get("id", conversation_id),
                                title=conversation_data.get("title", f"Conversation {conversation_id[:8]}"),
                                pdf_id=stored_pdf_id,
                                metadata=metadata
                            )

                            conversations.append(conv_state)
                    except Exception as load_error:
                        # Try the slower method if direct loading fails
                        logger.warning(f"Error loading conversation file {file_name}: {str(load_error)}")
                        logger.warning("Trying fallback loading method")

                        conversation = await self.get_conversation(conversation_id)
                        if conversation:
                            # Filter by PDF ID if specified
                            if pdf_id and str(conversation.pdf_id) != str(pdf_id):
                                continue

                            # Skip deleted conversations
                            if conversation.metadata and conversation.metadata.get("is_deleted", False):
                                continue

                            conversations.append(conversation)

                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing conversation file {file_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # Sort by updated_at (newest first)
            # Use a safe sort key that handles missing attributes
            def safe_sort_key(x):
                if hasattr(x, 'updated_at'):
                    return x.updated_at
                return datetime.min

            conversations.sort(key=safe_sort_key, reverse=True)

            if error_count > 0:
                logger.warning(f"Encountered {error_count} errors while listing conversations")

            logger.info(f"Found {len(conversations)} conversations" + (f" for PDF {pdf_id}" if pdf_id else ""))
            return conversations

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"Error listing conversations: {str(e)}")
            logger.error("Exception traceback:")
            traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in traceback_lines:
                logger.error(line.rstrip())
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation using soft delete approach.

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

            # We could delete the file, but instead we'll mark it as deleted in the metadata
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if "metadata" not in data:
                    data["metadata"] = {}

                data["metadata"]["is_deleted"] = True
                data["metadata"]["deleted_at"] = datetime.utcnow().isoformat()

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                return True
            except Exception as e:
                logger.error(f"Error marking conversation as deleted: {str(e)}")
                logger.error(traceback.format_exc())

                # Fall back to actual file deletion if metadata update fails
                try:
                    os.remove(file_path)
                    return True
                except:
                    return False

        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def create_conversation(
        self,
        title: str = None,
        pdf_id: str = None,
        metadata: Dict[str, Any] = None,
        id: str = None
    ) -> ConversationState:
        """
        Create a new conversation with proper initialization.

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

            # Default title if not provided
            if not title:
                title = f"Conversation about {pdf_id}" if pdf_id else f"Conversation {conversation_id[:8]}"

            logger.info(f"Creating new conversation: ID={conversation_id}, PDF={pdf_id}")

            # Initialize metadata with defaults if not provided
            if not metadata:
                metadata = {
                    "created_at": datetime.utcnow().isoformat(),
                    "pdf_id": pdf_id
                }

            # Create new conversation state
            conversation = ConversationState(
                conversation_id=conversation_id,
                title=title,
                pdf_id=pdf_id or "",
                metadata=metadata
            )

            # Add system message by default
            system_prompt = "You are an AI assistant specialized in answering questions about documents."
            conversation.add_message(MessageType.SYSTEM, system_prompt)

            # Save the conversation
            success = await self.save_conversation(conversation)
            if not success:
                logger.error(f"Failed to save new conversation {conversation_id}")
                raise Exception("Failed to save new conversation")

            return conversation

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"Error creating conversation: {str(e)}")
            logger.error("Exception traceback:")
            traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in traceback_lines:
                logger.error(line.rstrip())
            raise e

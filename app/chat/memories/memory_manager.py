"""
SQL-based memory manager for conversation persistence.
Simplified implementation with sync interfaces.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import traceback

from app.chat.langgraph.state import ConversationState, MessageType
from app.web.db.models import Conversation as DBConversation, Message as DBMessage
from app.web.db import db

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manager for conversation memory persistence using SQL database"""

    def __init__(self):
        """Initialize memory manager"""
        logger.info("SQL-based Memory manager initialized")

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get conversation state by ID from SQL database

        Args:
            conversation_id: Conversation ID

        Returns:
            ConversationState if found, None otherwise
        """
        try:
            # Find the conversation in the database
            conversation = db.session.execute(
                db.select(DBConversation).filter_by(id=conversation_id, is_deleted=False)
            ).scalar_one_or_none()

            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found in database")
                return None

            # Create new conversation state
            conv_state = ConversationState(
                conversation_id=conversation.id,
                title=conversation.title or f"Conversation {conversation_id[:8]}",
                pdf_id=str(conversation.pdf_id) if conversation.pdf_id else "",
                metadata=conversation.json_metadata or {}
            )

            # Add messages
            for msg in conversation.messages:
                # Skip deleted messages
                if hasattr(msg, 'is_deleted') and msg.is_deleted:
                    continue

                # Map role to MessageType enum
                try:
                    msg_type = getattr(MessageType, msg.role.upper()) if msg.role.upper() in dir(MessageType) else MessageType.USER
                except:
                    # Default to appropriate message type based on role string
                    type_mapping = {
                        "user": MessageType.USER,
                        "assistant": MessageType.AI,
                        "system": MessageType.SYSTEM,
                        "tool": MessageType.TOOL
                    }
                    msg_type = type_mapping.get(msg.role.lower(), MessageType.USER)

                # Get message metadata
                metadata = msg.msg_metadata if hasattr(msg, 'msg_metadata') else msg.get_metadata() if hasattr(msg, 'get_metadata') else {}

                # Add message to conversation state
                conv_state.add_message(
                    type=msg_type,
                    content=msg.content,
                    metadata=metadata
                )

            # Add system message if none exists
            has_system_message = any(msg.type == MessageType.SYSTEM for msg in conv_state.messages)
            if not has_system_message:
                system_prompt = "You are an AI assistant specialized in answering questions about documents."
                conv_state.add_message(MessageType.SYSTEM, system_prompt)

            logger.info(f"Retrieved conversation {conversation_id} with {len(conv_state.messages)} messages")
            return conv_state

        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
            logger.error(traceback.format_exc())

            # Create a new conversation as recovery mechanism
            try:
                logger.info(f"Creating a recovery conversation for {conversation_id}")
                return ConversationState(
                    conversation_id=conversation_id,
                    title=f"Recovered Conversation {conversation_id[:8]}",
                    metadata={"recovered": True, "error": str(e)}
                )
            except:
                return None

    def save_conversation(self, conversation: ConversationState) -> bool:
        """
        Save conversation state to SQL database

        Args:
            conversation: Conversation state to save

        Returns:
            Success status
        """
        if not conversation or not conversation.conversation_id:
            logger.error("Cannot save conversation: Invalid conversation state")
            return False

        try:
            # Find or create conversation in database
            db_conversation = db.session.execute(
                db.select(DBConversation).filter_by(id=conversation.conversation_id)
            ).scalar_one_or_none()

            if not db_conversation:
                # Get PDF ID (handle string PDF IDs)
                try:
                    pdf_id = int(conversation.pdf_id) if conversation.pdf_id else None
                except (ValueError, TypeError):
                    pdf_id = conversation.pdf_id

                # Get user ID from metadata
                user_id = conversation.metadata.get("user_id") if conversation.metadata else None
                if not user_id:
                    # Get the first user as fallback
                    from app.web.db.models import User
                    user = db.session.execute(db.select(User).limit(1)).scalar_one_or_none()
                    user_id = user.id if user else 1

                # Create new conversation
                db_conversation = DBConversation(
                    id=conversation.conversation_id,
                    title=conversation.title or f"Conversation {conversation.conversation_id[:8]}",
                    pdf_id=pdf_id,
                    user_id=user_id,
                    json_metadata=conversation.metadata or {}
                )
                db.session.add(db_conversation)
            else:
                # Update existing conversation
                db_conversation.title = conversation.title or db_conversation.title
                db_conversation.json_metadata = conversation.metadata or db_conversation.json_metadata
                db_conversation.last_updated = datetime.now()

            # Commit to save conversation and get ID
            db.session.commit()

            # Process messages - first get existing messages
            existing_messages = {msg.content: msg for msg in db_conversation.messages}

            # Add or update messages
            for msg in conversation.messages:
                # Skip system messages if we want to keep original system prompt
                if msg.type == MessageType.SYSTEM and msg.content in existing_messages:
                    continue

                # Get message type as string
                msg_type = msg.type.value if hasattr(msg.type, 'value') else str(msg.type).split('.')[-1].lower()

                # Check if message already exists to avoid duplicates
                if msg.content in existing_messages:
                    # Update metadata if needed
                    existing_msg = existing_messages[msg.content]
                    if msg.metadata:
                        # Check which metadata setting function is available
                        if hasattr(existing_msg, 'msg_metadata'):
                            existing_msg.msg_metadata = msg.metadata
                        elif hasattr(existing_msg, 'set_metadata'):
                            existing_msg.set_metadata(msg.metadata)
                        db.session.add(existing_msg)
                else:
                    # Create new message
                    new_message = DBMessage(
                        conversation_id=db_conversation.id,
                        role=msg_type,
                        content=msg.content
                    )

                    # Set metadata using available function
                    if hasattr(new_message, 'msg_metadata'):
                        new_message.msg_metadata = msg.metadata
                    elif hasattr(new_message, 'set_metadata'):
                        new_message.set_metadata(msg.metadata)
                    elif hasattr(new_message, 'meta_json'):
                        new_message.meta_json = json.dumps(msg.metadata)

                    db.session.add(new_message)

            # Commit changes
            db.session.commit()

            logger.info(f"Saved conversation {conversation.conversation_id} with {len(conversation.messages)} messages")
            return True

        except Exception as e:
            logger.error(f"Error saving conversation {conversation.conversation_id}: {str(e)}")
            logger.error(traceback.format_exc())

            # Roll back transaction
            db.session.rollback()
            return False

    def list_conversations(self, pdf_id: Optional[str] = None) -> List[ConversationState]:
        """
        List all available conversations, optionally filtered by PDF ID

        Args:
            pdf_id: Optional PDF ID to filter by

        Returns:
            List of conversation states
        """
        try:
            # Query for conversations
            query = db.select(DBConversation).filter_by(is_deleted=False)

            # Filter by PDF ID if provided
            if pdf_id:
                try:
                    # Try to convert to int for SQL comparison
                    pdf_id_int = int(pdf_id)
                    query = query.filter_by(pdf_id=pdf_id_int)
                except (ValueError, TypeError):
                    # If conversion fails, use string comparison
                    query = query.filter_by(pdf_id=pdf_id)

            # Order by last updated
            query = query.order_by(DBConversation.last_updated.desc())

            # Execute query
            conversations = db.session.execute(query).scalars().all()

            # Convert to conversation states (minimal version for performance)
            result = []
            for conv in conversations:
                # Create simplified conversation state
                conv_state = ConversationState(
                    conversation_id=conv.id,
                    title=conv.title,
                    pdf_id=str(conv.pdf_id) if conv.pdf_id else "",
                    metadata=conv.json_metadata or {}
                )

                # Add just the first few messages for preview
                message_count = 0
                for msg in conv.messages:
                    if msg.role.lower() == "system":
                        continue

                    # Map role to MessageType enum
                    try:
                        msg_type = getattr(MessageType, msg.role.upper()) if msg.role.upper() in dir(MessageType) else MessageType.USER
                    except:
                        # Default to appropriate message type
                        type_mapping = {
                            "user": MessageType.USER,
                            "assistant": MessageType.AI,
                            "system": MessageType.SYSTEM,
                            "tool": MessageType.TOOL
                        }
                        msg_type = type_mapping.get(msg.role.lower(), MessageType.USER)

                    # Add message to conversation state
                    conv_state.add_message(
                        type=msg_type,
                        content=msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                        metadata={}
                    )

                    message_count += 1
                    if message_count >= 2:  # Just load 2 messages for preview
                        break

                result.append(conv_state)

            logger.info(f"Found {len(result)} conversations" + (f" for PDF {pdf_id}" if pdf_id else ""))
            return result

        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def create_conversation(
        self,
        title: Optional[str] = None,
        pdf_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> ConversationState:
        """
        Create a new conversation

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
                    "created_at": datetime.now().isoformat(),
                    "pdf_id": pdf_id
                }

            # Create database entry
            try:
                # Get PDF ID (handle string PDF IDs)
                try:
                    pdf_id_val = int(pdf_id) if pdf_id else None
                except (ValueError, TypeError):
                    pdf_id_val = pdf_id

                # Get user ID from metadata
                user_id = metadata.get("user_id")
                if not user_id:
                    # Get the first user as fallback
                    from app.web.db.models import User
                    user = db.session.execute(db.select(User).limit(1)).scalar_one_or_none()
                    user_id = user.id if user else 1

                # Create DB conversation
                db_conversation = DBConversation(
                    id=conversation_id,
                    title=title,
                    pdf_id=pdf_id_val,
                    user_id=user_id,
                    json_metadata=metadata
                )
                db.session.add(db_conversation)

                # Add system message
                system_message = DBMessage(
                    conversation_id=conversation_id,
                    role="system",
                    content="You are an AI assistant specialized in answering questions about documents."
                )
                db.session.add(system_message)

                # Commit changes
                db.session.commit()

            except Exception as db_error:
                logger.error(f"Database error creating conversation: {str(db_error)}")
                db.session.rollback()

            # Create new conversation state
            conversation = ConversationState(
                conversation_id=conversation_id,
                title=title,
                pdf_id=pdf_id or "",
                metadata=metadata
            )

            # Add system message
            system_prompt = "You are an AI assistant specialized in answering questions about documents."
            conversation.add_message(MessageType.SYSTEM, system_prompt)

            return conversation

        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            logger.error(traceback.format_exc())

            # Return a minimal conversation state
            return ConversationState(
                conversation_id=id or str(uuid.uuid4()),
                title=title or "New Conversation",
                metadata={"error": str(e)}
            )

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation using soft delete approach

        Args:
            conversation_id: ID of conversation to delete

        Returns:
            Success status
        """
        try:
            # Find conversation in database
            conversation = db.session.execute(
                db.select(DBConversation).filter_by(id=conversation_id)
            ).scalar_one_or_none()

            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found for deletion")
                return False

            # Soft delete
            conversation.is_deleted = True

            # Update metadata
            if not conversation.json_metadata:
                conversation.json_metadata = {}

            if isinstance(conversation.json_metadata, dict):
                conversation.json_metadata["is_deleted"] = True
                conversation.json_metadata["deleted_at"] = datetime.now().isoformat()

            # Commit changes
            db.session.commit()

            logger.info(f"Deleted conversation {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            logger.error(traceback.format_exc())

            # Roll back transaction
            db.session.rollback()
            return False

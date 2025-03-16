# Model Refactoring for LangGraph Migration

## Current Model Structure

### LangGraph Models (`app/chat/models/`)
- **ChatArgs**: Configuration for chat sessions
- **Metadata**: Metadata for conversations
- **TechnicalMetadata**: Technical information about documents
- **ConceptMetadata**: Concept mapping for documents

### Legacy Database Models (`app/web/db/models/`)
- **Conversation**: Database model for conversation tracking
- **Message**: Database model for individual messages
- **PDF**: Database model for PDF documents
- **User**: Database model for user information

## Obsolete Models After LangGraph Migration

With our LangGraph migration, several database models are now obsolete:

1. **Conversation**: Replaced by ConversationState in MemoryManager
2. **Message**: Replaced by message objects in ConversationState
3. **Relations between models**: No longer needed as state is managed in memory

## Required Model Changes

### 1. Create Proper ConversationState Model

We need a proper ConversationState model to align with our LangGraph architecture:

```python
# app/chat/models/conversation.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

class Message(BaseModel):
    """Message in a conversation"""
    type: str  # "user", "assistant", "system", "tool"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class ConversationState(BaseModel):
    """Full conversation state for LangGraph architecture"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "Untitled Conversation"
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation"""
        self.messages.append(Message(
            type=type,
            content=content,
            metadata=metadata or {}
        ))
        self.updated_at = datetime.now()
```

### 2. Update ChatArgs to Use New Models

Update ChatArgs to use our new models and remove any unnecessary legacy fields:

```python
# app/chat/models/__init__.py (updated)
class ChatArgs(BaseModel):
    conversation_id: Optional[str] = None
    pdf_id: Optional[str] = None
    research_mode: ResearchMode = ResearchMode.SINGLE
    stream_enabled: bool = False
    # Add more fields as needed
```

## Database Models to Retain

Some database models still serve a purpose and should be retained:

1. **PDF**: Still needed to store PDF metadata and retrieval information
2. **User**: Required for authentication and user management

## Files to Delete / Obsolete Code

The following files contain obsolete code that can be removed:

1. `app/web/db/models/conversation.py` - Replaced by in-memory state
2. `app/web/db/models/message.py` - Replaced by in-memory messages
3. Legacy imports and functions in various files that reference these models

## Implementation Plan

1. Create new ConversationState model
2. Update MemoryManager to use new model
3. Update ChatManager to use new model
4. Remove references to obsolete database models in API layer
5. Delete obsolete model files
6. Test thoroughly to ensure no regressions

## Migration Impact

### Affected Components
- Chat processing
- Conversation history
- Message handling
- Research mode
- UI integration

### Benefits
- Cleaner architecture with proper separation of concerns
- No database dependencies for conversation management
- More efficient state handling
- Better alignment with LangGraph's stateful design pattern

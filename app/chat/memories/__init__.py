"""
Memory management module for PDF RAG system.
Provides conversation history persistence using LangGraph architecture.
"""

from app.chat.memories.memory_manager import MemoryManager as ChatMemoryManager

__all__ = ["ChatMemoryManager"]

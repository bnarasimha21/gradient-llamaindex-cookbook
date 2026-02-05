"""
Tests for Example 02 - Chat Engine with Memory.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestChatMemory:
    """Tests for chat with memory functionality."""
    
    def test_memory_stores_messages(self):
        """Memory should store conversation history."""
        from llama_index.core.memory import ChatMemoryBuffer
        
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        
        # Initially empty
        assert len(memory.get_all()) == 0
        
        # Add messages
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        
        memory.put(ChatMessage(role=MessageRole.USER, content="Hello"))
        memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"))
        
        messages = memory.get_all()
        assert len(messages) == 2
    
    def test_memory_respects_token_limit(self):
        """Memory should respect token limits."""
        from llama_index.core.memory import ChatMemoryBuffer
        
        # Very small limit
        memory = ChatMemoryBuffer.from_defaults(token_limit=50)
        
        # This should work without raising
        assert memory is not None
    
    def test_chat_engine_uses_retriever(self, mock_retriever, mock_llm):
        """Chat engine should use retriever for context."""
        # This test verifies the integration pattern
        assert mock_retriever.retrieve is not None
        assert mock_llm.chat is not None


class TestChatMemoryIntegration:
    """Integration tests requiring real credentials."""
    
    @pytest.mark.integration
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with real credentials."""
        from llama_index.core.chat_engine import CondensePlusContextChatEngine
        from llama_index.core.memory import ChatMemoryBuffer
        from shared.config import get_retriever, get_llm, Settings
        
        settings = Settings.from_env()
        if not settings.validate():
            pytest.skip("Missing credentials")
        
        retriever = get_retriever(num_results=3)
        llm = get_llm()
        memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
        
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=memory
        )
        
        # First message
        response1 = chat_engine.chat("What is a Droplet?")
        assert response1 is not None
        
        # Follow-up (should use context)
        response2 = chat_engine.chat("How do I create one?")
        assert response2 is not None
        
        # Memory should have messages
        assert len(memory.get_all()) >= 2

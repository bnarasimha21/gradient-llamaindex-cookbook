"""
Tests for Example 03 - Streaming Responses.
"""

import pytest
from unittest.mock import MagicMock


class TestStreaming:
    """Tests for streaming functionality."""
    
    def test_stream_returns_chunks(self, mock_llm):
        """Streaming should return multiple chunks."""
        chunks = list(mock_llm.stream_complete("test"))
        
        assert len(chunks) > 0
        assert all(hasattr(c, 'delta') for c in chunks)
    
    def test_chunks_combine_to_response(self, mock_llm):
        """Chunks should combine to form complete response."""
        chunks = list(mock_llm.stream_complete("test"))
        
        full_response = "".join(c.delta for c in chunks)
        assert len(full_response) > 0
    
    def test_stream_is_iterable(self, mock_llm):
        """Stream should be iterable."""
        stream = mock_llm.stream_complete("test")
        
        # Should be iterable
        first_chunk = next(iter(stream))
        assert first_chunk is not None


class TestStreamingIntegration:
    """Integration tests for streaming."""
    
    @pytest.mark.integration
    def test_real_streaming(self):
        """Test streaming with real credentials."""
        from shared.config import get_llm, Settings
        
        settings = Settings.from_env()
        if not settings.validate():
            pytest.skip("Missing credentials")
        
        llm = get_llm()
        
        chunks = []
        for chunk in llm.stream_complete("Say 'hello' in one word."):
            chunks.append(chunk.delta)
        
        full_response = "".join(chunks)
        assert len(full_response) > 0
        assert len(chunks) >= 1

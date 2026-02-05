"""
Tests for Example 11 - Async Batch Processing.
"""

import pytest
import asyncio


class TestAsyncBatch:
    """Tests for async batch functionality."""
    
    @pytest.mark.asyncio
    async def test_async_retrieve(self, mock_retriever):
        """Async retrieval should return nodes."""
        nodes = await mock_retriever.aretrieve("test query")
        
        assert nodes is not None
        assert len(nodes) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_retrieval(self, mock_retriever):
        """Multiple async retrievals should run concurrently."""
        queries = ["query1", "query2", "query3"]
        
        tasks = [mock_retriever.aretrieve(q) for q in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(queries)
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_async_llm(self, mock_llm):
        """Async LLM should return response."""
        response = await mock_llm.acomplete("test prompt")
        
        assert response is not None
        assert str(response) != ""


class TestAsyncBatchIntegration:
    """Integration tests for async batch."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_async_batch(self):
        """Test async batch with real credentials."""
        from shared.config import get_retriever, Settings
        
        settings = Settings.from_env()
        if not settings.validate():
            pytest.skip("Missing credentials")
        
        retriever = get_retriever(num_results=3)
        
        queries = [
            "What is a Droplet?",
            "What is Kubernetes?",
            "What are Spaces?"
        ]
        
        tasks = [retriever.aretrieve(q) for q in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(len(r) > 0 for r in results)

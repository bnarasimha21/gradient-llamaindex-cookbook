"""
Tests for Example 01 - Simple RAG Query Engine.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSimpleRAG:
    """Tests for simple RAG functionality."""
    
    def test_query_engine_returns_response(self, mock_query_engine):
        """Query engine should return a response with source nodes."""
        response = mock_query_engine.query("How do I create a Droplet?")
        
        assert response is not None
        assert response.response is not None
        assert len(response.source_nodes) > 0
    
    def test_response_has_source_nodes(self, mock_query_engine, mock_nodes):
        """Response should include source nodes with scores."""
        response = mock_query_engine.query("What is a Droplet?")
        
        assert response.source_nodes is not None
        assert len(response.source_nodes) == len(mock_nodes)
        
        # Check scores
        for node in response.source_nodes:
            assert node.score is not None
            assert 0 <= node.score <= 1
    
    def test_source_nodes_have_metadata(self, mock_nodes):
        """Source nodes should preserve metadata."""
        for node in mock_nodes:
            assert node.node.metadata is not None
            assert "source" in node.node.metadata
    
    def test_retriever_returns_nodes(self, mock_retriever):
        """Retriever should return NodeWithScore objects."""
        nodes = mock_retriever.retrieve("test query")
        
        assert len(nodes) > 0
        assert all(hasattr(n, 'score') for n in nodes)
        assert all(hasattr(n, 'node') for n in nodes)


class TestSimpleRAGIntegration:
    """Integration tests requiring real credentials."""
    
    @pytest.mark.integration
    def test_real_query(self):
        """Test with real Gradient credentials."""
        from shared.config import get_query_engine, Settings
        
        settings = Settings.from_env()
        if not settings.validate():
            pytest.skip("Missing credentials")
        
        engine = get_query_engine(num_results=3)
        response = engine.query("What is a Droplet?")
        
        assert response is not None
        assert len(str(response)) > 0
        assert len(response.source_nodes) > 0

"""
Pytest configuration and shared fixtures.
"""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock

from llama_index.core.schema import NodeWithScore, TextNode


# Check if real credentials are available
def has_credentials():
    return bool(
        os.getenv("DIGITALOCEAN_ACCESS_TOKEN")
        and os.getenv("MODEL_ACCESS_KEY")
        and os.getenv("GRADIENT_KB_ID")
    )


# Skip marker for integration tests
skip_without_credentials = pytest.mark.skipif(
    not has_credentials(),
    reason="Integration test requires credentials"
)


@pytest.fixture
def mock_nodes():
    """Create mock retrieved nodes."""
    return [
        NodeWithScore(
            node=TextNode(
                text="A Droplet is a DigitalOcean virtual machine. You can create Droplets in various sizes with different CPU, memory, and storage configurations.",
                id_="node-1",
                metadata={"source": "droplets.md"}
            ),
            score=0.95
        ),
        NodeWithScore(
            node=TextNode(
                text="To create a Droplet, go to the DigitalOcean Control Panel, click Create, and select Droplets. Choose your image, size, and datacenter region.",
                id_="node-2",
                metadata={"source": "create-droplet.md"}
            ),
            score=0.88
        ),
        NodeWithScore(
            node=TextNode(
                text="Kubernetes on DigitalOcean (DOKS) is a managed Kubernetes service that lets you deploy containerized applications without managing the control plane.",
                id_="node-3",
                metadata={"source": "kubernetes.md"}
            ),
            score=0.72
        ),
    ]


@pytest.fixture
def mock_retriever(mock_nodes):
    """Create a mock retriever."""
    retriever = MagicMock()
    retriever.retrieve.return_value = mock_nodes
    retriever.aretrieve = AsyncMock(return_value=mock_nodes)
    return retriever


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    
    # Mock complete
    mock_response = MagicMock()
    mock_response.text = "A Droplet is a virtual machine on DigitalOcean. You can create one through the Control Panel or API."
    mock_response.__str__ = lambda self: self.text
    
    llm.complete.return_value = mock_response
    llm.acomplete = AsyncMock(return_value=mock_response)
    
    # Mock chat
    llm.chat.return_value = mock_response
    llm.achat = AsyncMock(return_value=mock_response)
    
    # Mock streaming
    def mock_stream():
        for word in ["A ", "Droplet ", "is ", "a ", "virtual ", "machine."]:
            chunk = MagicMock()
            chunk.delta = word
            yield chunk
    
    llm.stream_complete.return_value = mock_stream()
    
    return llm


@pytest.fixture
def mock_query_engine(mock_retriever, mock_llm, mock_nodes):
    """Create a mock query engine."""
    from llama_index.core.response.schema import Response
    
    engine = MagicMock()
    
    response = Response(
        response="A Droplet is a virtual machine on DigitalOcean.",
        source_nodes=mock_nodes
    )
    
    engine.query.return_value = response
    engine.aquery = AsyncMock(return_value=response)
    
    return engine

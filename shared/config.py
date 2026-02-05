"""
Shared configuration and factory functions for Gradient components.

All examples use these helpers to ensure consistent setup.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from llama_index.retrievers.digitalocean.gradientai import GradientKBRetriever
from llama_index.llms.digitalocean.gradientai import GradientAI
from llama_index.core.query_engine import RetrieverQueryEngine

# Load environment variables
load_dotenv()


@dataclass
class Settings:
    """Configuration settings loaded from environment."""
    
    digitalocean_token: str = os.getenv("DIGITALOCEAN_ACCESS_TOKEN", "")
    model_access_key: str = os.getenv("MODEL_ACCESS_KEY", "")
    knowledge_base_id: str = os.getenv("GRADIENT_KB_ID", "")
    llm_model: str = os.getenv("GRADIENT_LLM_MODEL", "llama3.3-70b-instruct")
    
    def validate(self) -> bool:
        """Check if required credentials are set."""
        return bool(
            self.digitalocean_token 
            and self.model_access_key 
            and self.knowledge_base_id
        )
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls()


def get_retriever(
    settings: Settings | None = None,
    num_results: int = 5,
    **kwargs
) -> GradientKBRetriever:
    """
    Create a GradientKBRetriever instance.
    
    Args:
        settings: Configuration settings (defaults to env vars)
        num_results: Number of results to retrieve
        **kwargs: Additional arguments passed to retriever
        
    Returns:
        Configured GradientKBRetriever
        
    Example:
        >>> retriever = get_retriever(num_results=10)
        >>> nodes = retriever.retrieve("How do I create a Droplet?")
    """
    settings = settings or Settings.from_env()
    
    return GradientKBRetriever(
        knowledge_base_id=settings.knowledge_base_id,
        api_token=settings.digitalocean_token,
        num_results=num_results,
        **kwargs
    )


def get_llm(
    settings: Settings | None = None,
    model: str | None = None,
    **kwargs
) -> GradientAI:
    """
    Create a GradientAI LLM instance.
    
    Args:
        settings: Configuration settings (defaults to env vars)
        model: Model name (defaults to settings.llm_model)
        **kwargs: Additional arguments passed to LLM
        
    Returns:
        Configured GradientAI LLM
        
    Example:
        >>> llm = get_llm()
        >>> response = llm.complete("Explain Kubernetes in one sentence.")
    """
    settings = settings or Settings.from_env()
    
    return GradientAI(
        model=model or settings.llm_model,
        model_access_key=settings.model_access_key,
        **kwargs
    )


def get_query_engine(
    settings: Settings | None = None,
    num_results: int = 5,
    **kwargs
) -> RetrieverQueryEngine:
    """
    Create a complete RAG query engine with Gradient components.
    
    Args:
        settings: Configuration settings (defaults to env vars)
        num_results: Number of results to retrieve
        **kwargs: Additional arguments passed to query engine
        
    Returns:
        Configured RetrieverQueryEngine
        
    Example:
        >>> engine = get_query_engine()
        >>> response = engine.query("How do I set up a load balancer?")
        >>> print(response)
    """
    settings = settings or Settings.from_env()
    
    retriever = get_retriever(settings, num_results=num_results)
    llm = get_llm(settings)
    
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        **kwargs
    )

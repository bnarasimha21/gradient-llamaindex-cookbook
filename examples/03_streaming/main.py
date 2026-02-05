"""
03 - Streaming Responses

Real-time token streaming for better UX.
"""

import sys
import asyncio
sys.path.insert(0, "../..")

from llama_index.core.query_engine import RetrieverQueryEngine

from shared.config import get_retriever, get_llm, Settings


def sync_streaming_demo():
    """Demonstrate synchronous streaming."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Initializing LLM for streaming...")
    llm = get_llm()
    
    prompt = "Explain what DigitalOcean Droplets are in 3 sentences."
    
    print(f"\n📝 Prompt: {prompt}")
    print("\n💬 Streaming response:\n")
    print("-"*40)
    
    # Stream tokens one by one
    response_stream = llm.stream_complete(prompt)
    
    full_response = ""
    for chunk in response_stream:
        token = chunk.delta
        print(token, end="", flush=True)
        full_response += token
    
    print("\n" + "-"*40)
    print(f"\n✅ Total tokens: ~{len(full_response.split())}")


async def async_streaming_demo():
    """Demonstrate asynchronous streaming."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Initializing LLM for async streaming...")
    llm = get_llm()
    
    prompt = "List 5 key features of DigitalOcean Kubernetes."
    
    print(f"\n📝 Prompt: {prompt}")
    print("\n💬 Async streaming response:\n")
    print("-"*40)
    
    # Async stream
    response_stream = await llm.astream_complete(prompt)
    
    async for chunk in response_stream:
        print(chunk.delta, end="", flush=True)
    
    print("\n" + "-"*40)


def rag_streaming_demo():
    """Demonstrate streaming with full RAG pipeline."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Initializing RAG pipeline for streaming...")
    
    retriever = get_retriever(num_results=5)
    llm = get_llm()
    
    # Create streaming-enabled query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        streaming=True  # Enable streaming
    )
    
    query = "How do I set up a load balancer on DigitalOcean?"
    
    print(f"\n📝 Query: {query}")
    print("\n🔍 Retrieving relevant documents...")
    print("\n💬 Streaming RAG response:\n")
    print("-"*40)
    
    # Query with streaming
    streaming_response = query_engine.query(query)
    
    # Print stream (built-in method)
    streaming_response.print_response_stream()
    
    print("\n" + "-"*40)
    
    # Access source nodes after streaming
    print(f"\n📚 Used {len(streaming_response.source_nodes)} source documents")


def main():
    print("="*60)
    print("🌊 Streaming Responses Demo")
    print("="*60)
    
    print("\n\n### Demo 1: Basic Sync Streaming ###\n")
    sync_streaming_demo()
    
    print("\n\n### Demo 2: RAG with Streaming ###\n")
    rag_streaming_demo()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--async", dest="use_async", action="store_true",
                        help="Run async streaming demo")
    args = parser.parse_args()
    
    if args.use_async:
        print("="*60)
        print("🌊 Async Streaming Demo")
        print("="*60)
        asyncio.run(async_streaming_demo())
    else:
        main()

"""
11 - Async Batch Processing

Process multiple queries concurrently.
"""

import sys
import asyncio
import time
sys.path.insert(0, "../..")

from shared.config import get_retriever, get_llm, get_query_engine, Settings


async def single_async_query():
    """Demonstrate basic async retrieval."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Single Async Query Demo\n")
    
    retriever = get_retriever(num_results=3)
    
    query = "What is a Droplet?"
    
    print(f"📝 Query: {query}")
    print("⏳ Retrieving asynchronously...")
    
    start = time.time()
    nodes = await retriever.aretrieve(query)
    elapsed = time.time() - start
    
    print(f"✅ Retrieved {len(nodes)} nodes in {elapsed:.2f}s")
    
    for i, node in enumerate(nodes, 1):
        print(f"\n  {i}. [Score: {node.score:.3f}]")
        print(f"     {node.node.text[:100]}...")


async def batch_retrieval_demo(num_queries: int = 5):
    """Compare sequential vs concurrent batch retrieval."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print(f"🔧 Batch Retrieval Demo ({num_queries} queries)\n")
    
    retriever = get_retriever(num_results=3)
    
    # Generate test queries
    queries = [
        "How do I create a Droplet?",
        "What is Kubernetes?",
        "How do I set up a database?",
        "What are Spaces?",
        "How do I configure DNS?",
        "What is App Platform?",
        "How do I use Functions?",
        "What is a VPC?",
        "How do I set up a firewall?",
        "What are load balancers?",
    ][:num_queries]
    
    # Sequential execution
    print("📊 Sequential Execution:")
    start = time.time()
    sequential_results = []
    for q in queries:
        nodes = await retriever.aretrieve(q)
        sequential_results.append(nodes)
    sequential_time = time.time() - start
    print(f"   ⏱️ Time: {sequential_time:.2f}s")
    
    # Concurrent execution
    print("\n📊 Concurrent Execution:")
    start = time.time()
    tasks = [retriever.aretrieve(q) for q in queries]
    concurrent_results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start
    print(f"   ⏱️ Time: {concurrent_time:.2f}s")
    
    # Summary
    print("\n" + "="*50)
    print("📈 Results")
    print("="*50)
    print(f"Queries: {num_queries}")
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time/concurrent_time:.1f}x")


async def rate_limited_batch(num_queries: int = 10, max_concurrent: int = 3):
    """Batch processing with rate limiting."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print(f"🔧 Rate-Limited Batch ({num_queries} queries, max {max_concurrent} concurrent)\n")
    
    retriever = get_retriever(num_results=3)
    llm = get_llm()
    
    # Semaphore limits concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_query(query: str, idx: int):
        async with semaphore:
            print(f"  [{idx}] Starting: {query[:30]}...")
            
            # Retrieve
            nodes = await retriever.aretrieve(query)
            
            # Generate response
            context = "\n".join([n.node.text[:200] for n in nodes[:2]])
            prompt = f"Based on this context, briefly answer: {query}\n\nContext: {context}"
            response = await llm.acomplete(prompt)
            
            print(f"  [{idx}] Done: {query[:30]}...")
            return {"query": query, "response": str(response)[:100]}
    
    queries = [
        "How do I create a Droplet?",
        "What is Kubernetes?",
        "How do I set up PostgreSQL?",
        "What are Spaces?",
        "How do I configure DNS?",
        "What is App Platform?",
        "How do I use Functions?",
        "What is a VPC?",
        "How do I set up a firewall?",
        "What are load balancers?",
    ][:num_queries]
    
    start = time.time()
    tasks = [process_query(q, i) for i, q in enumerate(queries, 1)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    print("\n" + "="*50)
    print(f"✅ Processed {num_queries} queries in {elapsed:.2f}s")
    print(f"   Average: {elapsed/num_queries:.2f}s per query")
    print(f"   Max concurrent: {max_concurrent}")
    print("="*50)
    
    print("\n📋 Results preview:")
    for r in results[:3]:
        print(f"  Q: {r['query'][:40]}...")
        print(f"  A: {r['response'][:60]}...\n")


async def full_async_rag():
    """Complete async RAG pipeline."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Full Async RAG Pipeline\n")
    
    retriever = get_retriever(num_results=5)
    llm = get_llm()
    
    query = "What are the best practices for securing a Kubernetes cluster on DigitalOcean?"
    
    print(f"📝 Query: {query}\n")
    
    # Async retrieve
    start = time.time()
    print("🔍 Retrieving documents...")
    nodes = await retriever.aretrieve(query)
    retrieve_time = time.time() - start
    print(f"   Retrieved {len(nodes)} docs in {retrieve_time:.2f}s")
    
    # Prepare context
    context = "\n\n".join([
        f"Document {i+1}:\n{node.node.text[:500]}"
        for i, node in enumerate(nodes)
    ])
    
    # Async generate
    print("💭 Generating response...")
    gen_start = time.time()
    prompt = f"""Answer this question based on the provided documents.

Question: {query}

Documents:
{context}

Answer:"""
    
    response = await llm.acomplete(prompt)
    gen_time = time.time() - gen_start
    print(f"   Generated in {gen_time:.2f}s")
    
    total_time = time.time() - start
    
    print("\n" + "="*60)
    print(f"💬 Response:\n{response}")
    print("="*60)
    print(f"\n⏱️ Timing: Retrieve {retrieve_time:.2f}s + Generate {gen_time:.2f}s = {total_time:.2f}s total")


def main():
    print("="*60)
    print("⚡ Async Batch Processing Demo")
    print("="*60 + "\n")
    
    asyncio.run(single_async_query())
    print("\n")
    asyncio.run(full_async_rag())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, help="Run batch demo with N queries")
    parser.add_argument("--rate-limited", type=int, help="Run rate-limited batch with N queries")
    args = parser.parse_args()
    
    if args.batch:
        asyncio.run(batch_retrieval_demo(args.batch))
    elif args.rate_limited:
        asyncio.run(rate_limited_batch(args.rate_limited))
    else:
        main()

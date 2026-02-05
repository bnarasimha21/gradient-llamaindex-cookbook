"""
04 - Reranking for Better Retrieval

Two-stage retrieval: retrieve many, rerank to best.
"""

import sys
sys.path.insert(0, "../..")

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

from shared.config import get_retriever, get_llm, Settings


def compare_with_without_reranking():
    """Compare retrieval quality with and without reranking."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Setting up comparison...")
    
    # Components
    retriever = get_retriever(num_results=10)  # Get more for reranking
    llm = get_llm()
    
    # Reranker (local, free)
    print("📥 Loading reranker model (first run downloads ~100MB)...")
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3  # Keep top 3 after reranking
    )
    
    # Two query engines
    engine_no_rerank = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
    )
    
    engine_with_rerank = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=[reranker]
    )
    
    query = "How do I configure autoscaling for my application?"
    
    # Without reranking
    print("\n" + "="*60)
    print("❌ WITHOUT Reranking (raw retrieval)")
    print("="*60)
    
    response1 = engine_no_rerank.query(query)
    
    print(f"\n📝 Query: {query}\n")
    print("📚 Retrieved documents (by initial score):")
    for i, node in enumerate(response1.source_nodes[:5], 1):
        score = node.score or 0
        preview = node.node.text[:80].replace('\n', ' ')
        print(f"  {i}. [Score: {score:.4f}] {preview}...")
    
    print(f"\n💬 Response:\n{str(response1)[:500]}...")
    
    # With reranking
    print("\n" + "="*60)
    print("✅ WITH Reranking (reranked top 3)")
    print("="*60)
    
    response2 = engine_with_rerank.query(query)
    
    print(f"\n📝 Query: {query}\n")
    print("📚 Retrieved documents (after reranking):")
    for i, node in enumerate(response2.source_nodes, 1):
        score = node.score or 0
        preview = node.node.text[:80].replace('\n', ' ')
        print(f"  {i}. [Score: {score:.4f}] {preview}...")
    
    print(f"\n💬 Response:\n{str(response2)[:500]}...")


def main():
    print("="*60)
    print("🔄 Reranking Demo")
    print("="*60)
    print("\nThis demo compares retrieval with and without reranking.")
    print("Reranking uses a cross-encoder model to re-score documents.\n")
    
    try:
        compare_with_without_reranking()
    except ImportError:
        print("❌ Missing dependency. Install with:")
        print("   pip install sentence-transformers")


if __name__ == "__main__":
    main()

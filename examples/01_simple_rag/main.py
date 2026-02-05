"""
01 - Simple RAG Query Engine

Basic retrieval-augmented generation with Gradient.
"""

import sys
sys.path.insert(0, "../..")

from shared.config import get_query_engine, Settings


def main():
    # Validate credentials
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials. Copy .env.example to .env and fill in values.")
        return
    
    # Create query engine
    print("🔧 Initializing RAG pipeline...")
    query_engine = get_query_engine(num_results=5)
    
    # Example queries about DigitalOcean
    queries = [
        "How do I create a Droplet?",
        "What is a Kubernetes cluster?",
        "How do I set up a managed database?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"📝 Query: {query}")
        print("="*60)
        
        response = query_engine.query(query)
        
        print(f"\n💬 Response:\n{response}")
        
        # Show source nodes
        if response.source_nodes:
            print(f"\n📚 Sources ({len(response.source_nodes)} documents):")
            for i, node in enumerate(response.source_nodes[:3], 1):
                score = node.score or 0
                text_preview = node.node.text[:100].replace('\n', ' ')
                print(f"  {i}. [Score: {score:.3f}] {text_preview}...")


if __name__ == "__main__":
    main()

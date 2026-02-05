"""
07 - Router Query Engine

Route queries to different engines based on topic.
"""

import sys
sys.path.insert(0, "../..")

from llama_index.core.query_engine import RouterQueryEngine, RetrieverQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from shared.config import get_retriever, get_llm, Settings


def main():
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Initializing Router Query Engine...")
    
    llm = get_llm()
    
    # Create multiple query engines for different topics
    # In production, these might use different KBs or metadata filters
    
    # Engine 1: Compute (Droplets, VMs)
    compute_retriever = get_retriever(num_results=5)
    compute_engine = RetrieverQueryEngine.from_args(
        retriever=compute_retriever,
        llm=llm
    )
    
    # Engine 2: Kubernetes
    k8s_retriever = get_retriever(num_results=5)
    k8s_engine = RetrieverQueryEngine.from_args(
        retriever=k8s_retriever,
        llm=llm
    )
    
    # Engine 3: Databases
    db_retriever = get_retriever(num_results=5)
    db_engine = RetrieverQueryEngine.from_args(
        retriever=db_retriever,
        llm=llm
    )
    
    # Engine 4: Storage (Spaces, Volumes)
    storage_retriever = get_retriever(num_results=5)
    storage_engine = RetrieverQueryEngine.from_args(
        retriever=storage_retriever,
        llm=llm
    )
    
    # Wrap as tools with clear descriptions
    query_engine_tools = [
        QueryEngineTool(
            query_engine=compute_engine,
            metadata=ToolMetadata(
                name="compute_docs",
                description=(
                    "Use for questions about Droplets, virtual machines, "
                    "CPU, RAM, compute resources, SSH access, and VM management."
                )
            )
        ),
        QueryEngineTool(
            query_engine=k8s_engine,
            metadata=ToolMetadata(
                name="kubernetes_docs",
                description=(
                    "Use for questions about Kubernetes (DOKS), containers, "
                    "pods, deployments, services, Helm, and container orchestration."
                )
            )
        ),
        QueryEngineTool(
            query_engine=db_engine,
            metadata=ToolMetadata(
                name="database_docs",
                description=(
                    "Use for questions about managed databases including "
                    "PostgreSQL, MySQL, Redis, MongoDB, connection strings, and backups."
                )
            )
        ),
        QueryEngineTool(
            query_engine=storage_engine,
            metadata=ToolMetadata(
                name="storage_docs",
                description=(
                    "Use for questions about Spaces (object storage), "
                    "Volumes (block storage), CDN, and data storage solutions."
                )
            )
        ),
    ]
    
    # Create router
    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=query_engine_tools,
        verbose=True  # Shows routing decisions
    )
    
    # Test queries for different topics
    queries = [
        ("How do I resize a Droplet?", "compute"),
        ("How do I deploy to Kubernetes?", "kubernetes"),
        ("How do I backup my PostgreSQL database?", "database"),
        ("How do I upload files to Spaces?", "storage"),
        ("What's the connection string format for Redis?", "database"),
    ]
    
    for query, expected_route in queries:
        print("\n" + "="*70)
        print(f"📝 Query: {query}")
        print(f"🎯 Expected route: {expected_route}")
        print("="*70 + "\n")
        
        response = router_engine.query(query)
        
        print(f"\n💬 Response:\n{response}")
        print("-"*70)


if __name__ == "__main__":
    main()

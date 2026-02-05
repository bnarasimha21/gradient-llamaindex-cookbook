"""
08 - Sub-Question Query Engine

Break complex queries into sub-questions for better answers.
"""

import sys
sys.path.insert(0, "../..")

from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.question_gen import LLMQuestionGenerator

from shared.config import get_query_engine, get_llm, Settings


def main():
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Initializing Sub-Question Query Engine...")
    
    # Base query engine
    query_engine = get_query_engine(num_results=5)
    llm = get_llm()
    
    # Wrap as tool
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="digitalocean_docs",
            description=(
                "DigitalOcean documentation covering Droplets, Kubernetes, "
                "App Platform, Databases, Spaces, Functions, and all products."
            )
        )
    )
    
    # Question generator (decomposes complex queries)
    question_gen = LLMQuestionGenerator.from_defaults(llm=llm)
    
    # Sub-question query engine
    sub_q_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[tool],
        llm=llm,
        question_gen=question_gen,
        verbose=True,  # Shows sub-questions
        use_async=False  # Set True for parallel sub-question answering
    )
    
    # Complex queries that benefit from decomposition
    complex_queries = [
        (
            "Compare Droplets and App Platform for hosting a high-traffic Python web application. "
            "Consider ease of deployment, scaling, and cost."
        ),
        (
            "What's the best way to set up a production-ready Kubernetes cluster "
            "with a managed PostgreSQL database and Redis cache?"
        ),
        (
            "How do Spaces and Volumes differ? When should I use each for storing "
            "user uploads in my application?"
        ),
    ]
    
    for query in complex_queries:
        print("\n" + "="*70)
        print("📝 Complex Query:")
        print(f"   {query}")
        print("="*70)
        
        print("\n🔍 Decomposing into sub-questions...\n")
        
        response = sub_q_engine.query(query)
        
        print("\n" + "-"*70)
        print("💬 Synthesized Response:")
        print("-"*70)
        print(f"\n{response}\n")
        
        # Show sources
        if response.source_nodes:
            print(f"📚 Based on {len(response.source_nodes)} source documents")


def simple_vs_subquestion():
    """Compare simple query vs sub-question approach."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔬 Comparing Simple RAG vs Sub-Question RAG\n")
    
    query_engine = get_query_engine(num_results=5)
    llm = get_llm()
    
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="digitalocean_docs",
            description="DigitalOcean documentation"
        )
    )
    
    sub_q_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[tool],
        llm=llm,
        verbose=True
    )
    
    query = "What are the pros and cons of using DigitalOcean Kubernetes vs deploying containers on Droplets?"
    
    print("="*70)
    print(f"Query: {query}")
    print("="*70)
    
    # Simple RAG
    print("\n### Simple RAG Response ###\n")
    simple_response = query_engine.query(query)
    print(simple_response)
    
    # Sub-question RAG
    print("\n### Sub-Question RAG Response ###\n")
    sub_q_response = sub_q_engine.query(query)
    print(sub_q_response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", 
                        help="Compare simple vs sub-question approach")
    args = parser.parse_args()
    
    if args.compare:
        simple_vs_subquestion()
    else:
        main()

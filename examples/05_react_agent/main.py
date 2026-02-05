"""
05 - ReAct Agent with Knowledge Base

Agent that reasons about when to search the KB.
"""

import sys
sys.path.insert(0, "../..")

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from shared.config import get_query_engine, get_llm, Settings


def main():
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Initializing ReAct Agent...")
    
    # Create query engine (will be wrapped as tool)
    query_engine = get_query_engine(num_results=5)
    llm = get_llm()
    
    # Wrap as a tool for the agent
    kb_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="digitalocean_docs",
            description=(
                "Search DigitalOcean documentation. Use this tool to find information "
                "about Droplets, Kubernetes, Databases, Spaces, Functions, Load Balancers, "
                "VPCs, and other DigitalOcean products and features. "
                "Input should be a search query."
            )
        )
    )
    
    # Create ReAct agent
    agent = ReActAgent.from_tools(
        tools=[kb_tool],
        llm=llm,
        verbose=True,  # Shows reasoning steps
        max_iterations=5
    )
    
    # Test queries - mix of searchable and direct
    queries = [
        # Needs KB search
        "How do I create a Droplet with a specific SSH key?",
        # Comparison query - may need multiple searches
        "What's the difference between Droplets and App Platform?",
        # Might not need search
        "What is 2 + 2?",
    ]
    
    for query in queries:
        print("\n" + "="*70)
        print(f"👤 User: {query}")
        print("="*70 + "\n")
        
        response = agent.chat(query)
        
        print("\n" + "-"*70)
        print(f"🤖 Final Answer: {response}")
        print("-"*70)


def interactive_mode():
    """Interactive agent chat."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    query_engine = get_query_engine(num_results=5)
    llm = get_llm()
    
    kb_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="digitalocean_docs",
            description="Search DigitalOcean documentation for product information."
        )
    )
    
    agent = ReActAgent.from_tools(
        tools=[kb_tool],
        llm=llm,
        verbose=True
    )
    
    print("🤖 ReAct Agent Ready (type 'quit' to exit)")
    print("   I can search DigitalOcean docs and reason about your questions.\n")
    
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if not user_input:
            continue
        
        response = agent.chat(user_input)
        print(f"\n🤖 Agent: {response}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        main()

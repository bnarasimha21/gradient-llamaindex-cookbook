"""
06 - Multi-Tool Agent

Agent with KB search, calculator, and datetime tools.
"""

import sys
from datetime import datetime
sys.path.insert(0, "../..")

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata

from shared.config import get_query_engine, get_llm, Settings


# Custom tool functions
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A math expression like "48 * 12" or "100 / 4"
    
    Returns:
        The result as a string
    """
    try:
        # Safety: only allow math operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math operations allowed"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def get_current_datetime() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current datetime in readable format
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


def get_days_until(target_date: str) -> str:
    """
    Calculate days until a target date.
    
    Args:
        target_date: Date in YYYY-MM-DD format
    
    Returns:
        Number of days until that date
    """
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d")
        today = datetime.now()
        delta = (target - today).days
        if delta < 0:
            return f"That date was {abs(delta)} days ago"
        return f"{delta} days until {target_date}"
    except:
        return "Error: Please use YYYY-MM-DD format"


def main():
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Initializing Multi-Tool Agent...")
    
    # Create tools
    query_engine = get_query_engine(num_results=5)
    llm = get_llm()
    
    # Tool 1: KB Search
    kb_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="digitalocean_docs",
            description=(
                "Search DigitalOcean documentation for information about "
                "products, features, pricing, and technical guides."
            )
        )
    )
    
    # Tool 2: Calculator
    calc_tool = FunctionTool.from_defaults(
        fn=calculate,
        name="calculator",
        description=(
            "Calculate mathematical expressions. "
            "Input should be a valid math expression like '48 * 12' or '100 / 4'."
        )
    )
    
    # Tool 3: Current DateTime
    datetime_tool = FunctionTool.from_defaults(
        fn=get_current_datetime,
        name="current_datetime",
        description="Get the current date and time. No input required."
    )
    
    # Tool 4: Days Until
    days_tool = FunctionTool.from_defaults(
        fn=get_days_until,
        name="days_until",
        description="Calculate days until a target date. Input: date in YYYY-MM-DD format."
    )
    
    # Create multi-tool agent
    agent = ReActAgent.from_tools(
        tools=[kb_tool, calc_tool, datetime_tool, days_tool],
        llm=llm,
        verbose=True,
        max_iterations=10
    )
    
    # Test queries that require different tools
    queries = [
        # Needs KB
        "How do I create a managed PostgreSQL database?",
        # Needs calculator
        "If a Droplet costs $48 per month, what's the cost for 3 Droplets for a year?",
        # Needs datetime
        "What's today's date?",
        # Multi-tool: KB + calculator
        "What's the cheapest Droplet and how much would 10 of them cost per year?",
        # Days until
        "How many days until 2026-12-31?",
    ]
    
    for query in queries:
        print("\n" + "="*70)
        print(f"👤 User: {query}")
        print("="*70 + "\n")
        
        response = agent.chat(query)
        
        print("\n" + "-"*70)
        print(f"🤖 Final Answer: {response}")
        print("-"*70)


if __name__ == "__main__":
    main()

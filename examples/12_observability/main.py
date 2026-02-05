"""
12 - Observability & Tracing

Debug and monitor RAG pipelines with callbacks.
"""

import sys
sys.path.insert(0, "../..")

from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)
from llama_index.core.query_engine import RetrieverQueryEngine

from shared.config import get_retriever, get_llm, Settings


def basic_debug_demo():
    """Basic debugging with LlamaDebugHandler."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Setting up Observability with LlamaDebugHandler...\n")
    
    # Create debug handler
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([debug_handler])
    
    # Create components with callback manager
    retriever = get_retriever(
        num_results=5,
        callback_manager=callback_manager
    )
    
    llm = get_llm(
        callback_manager=callback_manager
    )
    
    # Create query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        callback_manager=callback_manager
    )
    
    # Query
    query = "How do I create a Droplet with SSH keys?"
    
    print("="*60)
    print(f"📝 Query: {query}")
    print("="*60 + "\n")
    
    response = query_engine.query(query)
    
    # Response
    print("\n" + "="*60)
    print("💬 Response:")
    print("="*60)
    print(f"\n{response}\n")
    
    # Debug information
    print("="*60)
    print("🔍 Debug Information")
    print("="*60)
    
    # Event timing
    print("\n⏱️ Event Timing:")
    event_pairs = debug_handler.get_event_pairs()
    for event_type, events in event_pairs.items():
        for event in events:
            if hasattr(event, 'duration'):
                print(f"  {event_type}: {event.duration:.3f}s")
    
    # LLM inputs/outputs
    print("\n🤖 LLM Details:")
    llm_ios = debug_handler.get_llm_inputs_outputs()
    for i, io in enumerate(llm_ios, 1):
        if hasattr(io, 'prompt') and io.prompt:
            prompt_preview = str(io.prompt)[:200]
            print(f"  [{i}] Prompt: {prompt_preview}...")
        if hasattr(io, 'output') and io.output:
            output_preview = str(io.output)[:150]
            print(f"  [{i}] Output: {output_preview}...")


class VerboseHandler(LlamaDebugHandler):
    """Custom handler that prints events as they happen."""
    
    def on_event_start(self, event_type, payload=None, event_id=None, **kwargs):
        super().on_event_start(event_type, payload, event_id, **kwargs)
        
        if event_type == CBEventType.RETRIEVE:
            print(f"  🔍 Starting retrieval...")
        elif event_type == CBEventType.LLM:
            print(f"  🤖 Starting LLM call...")
        elif event_type == CBEventType.QUERY:
            print(f"  📝 Starting query processing...")
    
    def on_event_end(self, event_type, payload=None, event_id=None, **kwargs):
        super().on_event_end(event_type, payload, event_id, **kwargs)
        
        if event_type == CBEventType.RETRIEVE:
            nodes = payload.get("nodes", []) if payload else []
            print(f"  ✅ Retrieved {len(nodes)} documents")
        elif event_type == CBEventType.LLM:
            print(f"  ✅ LLM response received")
        elif event_type == CBEventType.QUERY:
            print(f"  ✅ Query complete")


def verbose_demo():
    """Demo with verbose event printing."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Setting up Verbose Observability...\n")
    
    # Custom verbose handler
    verbose_handler = VerboseHandler()
    callback_manager = CallbackManager([verbose_handler])
    
    retriever = get_retriever(num_results=5, callback_manager=callback_manager)
    llm = get_llm(callback_manager=callback_manager)
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        callback_manager=callback_manager
    )
    
    queries = [
        "What is a Droplet?",
        "How do I set up Kubernetes?",
    ]
    
    for query in queries:
        print("="*60)
        print(f"📝 Query: {query}")
        print("="*60)
        
        response = query_engine.query(query)
        
        print(f"\n💬 Response: {str(response)[:200]}...")
        print()


def cost_tracking_demo():
    """Demo showing how to track token usage."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Setting up Cost Tracking...\n")
    
    class TokenTracker(LlamaDebugHandler):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
        
        def on_event_end(self, event_type, payload=None, event_id=None, **kwargs):
            super().on_event_end(event_type, payload, event_id, **kwargs)
            
            if event_type == CBEventType.LLM and payload:
                response = payload.get("response")
                if response and hasattr(response, "raw"):
                    raw = response.raw
                    if hasattr(raw, "usage"):
                        self.total_prompt_tokens += getattr(raw.usage, "prompt_tokens", 0)
                        self.total_completion_tokens += getattr(raw.usage, "completion_tokens", 0)
        
        def get_stats(self):
            return {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
            }
    
    tracker = TokenTracker()
    callback_manager = CallbackManager([tracker])
    
    retriever = get_retriever(num_results=5, callback_manager=callback_manager)
    llm = get_llm(callback_manager=callback_manager)
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        callback_manager=callback_manager
    )
    
    queries = [
        "What is a Droplet?",
        "How do I create a Kubernetes cluster?",
        "What are managed databases?",
    ]
    
    print(f"Running {len(queries)} queries...\n")
    
    for query in queries:
        print(f"  • {query}")
        response = query_engine.query(query)
    
    print("\n" + "="*50)
    print("💰 Token Usage Summary")
    print("="*50)
    stats = tracker.get_stats()
    print(f"  Prompt tokens:     {stats['prompt_tokens']:,}")
    print(f"  Completion tokens: {stats['completion_tokens']:,}")
    print(f"  Total tokens:      {stats['total_tokens']:,}")
    
    # Estimate cost (example rates)
    # Note: Actual Gradient pricing may differ
    cost_per_1k_input = 0.001
    cost_per_1k_output = 0.002
    estimated_cost = (
        (stats["prompt_tokens"] / 1000) * cost_per_1k_input +
        (stats["completion_tokens"] / 1000) * cost_per_1k_output
    )
    print(f"  Estimated cost:    ${estimated_cost:.4f}")


def main():
    print("="*60)
    print("🔍 Observability & Tracing Demo")
    print("="*60 + "\n")
    
    basic_debug_demo()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Run verbose demo")
    parser.add_argument("--cost", action="store_true", help="Run cost tracking demo")
    args = parser.parse_args()
    
    if args.verbose:
        verbose_demo()
    elif args.cost:
        cost_tracking_demo()
    else:
        main()

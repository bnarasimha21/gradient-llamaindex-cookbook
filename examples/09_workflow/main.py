"""
09 - Workflow: Research Assistant

Multi-step workflow: retrieve → summarize → verify → respond.
"""

import sys
import asyncio
sys.path.insert(0, "../..")

from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    Event,
    StartEvent,
    StopEvent
)
from llama_index.core.schema import NodeWithScore

from shared.config import get_retriever, get_llm, Settings


# Define events for workflow steps
class RetrievalDoneEvent(Event):
    """Fired when retrieval completes."""
    nodes: list[NodeWithScore]
    query: str


class SummaryDoneEvent(Event):
    """Fired when summarization completes."""
    summary: str
    query: str


class VerificationDoneEvent(Event):
    """Fired when fact verification completes."""
    verified_content: str
    confidence: str


class ResearchWorkflow(Workflow):
    """
    A research assistant workflow that:
    1. Retrieves relevant documents
    2. Summarizes key points
    3. Verifies facts for accuracy
    4. Formats final response
    """
    
    def __init__(self, retriever, llm, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.llm = llm
    
    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrievalDoneEvent:
        """Step 1: Retrieve relevant documents from Gradient KB."""
        query = ev.query
        ctx.write_event_to_stream(Event(msg=f"🔍 Retrieving documents for: {query}"))
        
        # Use async retrieval
        nodes = await self.retriever.aretrieve(query)
        
        # Store in context for later steps
        await ctx.set("original_query", query)
        await ctx.set("source_nodes", nodes)
        
        ctx.write_event_to_stream(Event(msg=f"📚 Found {len(nodes)} relevant documents"))
        
        return RetrievalDoneEvent(nodes=nodes, query=query)
    
    @step
    async def summarize(self, ctx: Context, ev: RetrievalDoneEvent) -> SummaryDoneEvent:
        """Step 2: Summarize the retrieved information."""
        ctx.write_event_to_stream(Event(msg="📝 Summarizing key points..."))
        
        # Combine retrieved text
        context_text = "\n\n".join([
            f"Document {i+1}:\n{node.node.text}"
            for i, node in enumerate(ev.nodes[:5])
        ])
        
        # Summarize with LLM
        summary_prompt = f"""Based on the following documents, extract and summarize the key points relevant to the question: "{ev.query}"

{context_text}

Provide a clear, structured summary of the main points:"""
        
        summary_response = await self.llm.acomplete(summary_prompt)
        summary = str(summary_response)
        
        await ctx.set("summary", summary)
        
        return SummaryDoneEvent(summary=summary, query=ev.query)
    
    @step
    async def verify(self, ctx: Context, ev: SummaryDoneEvent) -> VerificationDoneEvent:
        """Step 3: Verify the accuracy of the summary."""
        ctx.write_event_to_stream(Event(msg="✅ Verifying accuracy..."))
        
        verify_prompt = f"""Review the following summary for accuracy and completeness regarding the question: "{ev.query}"

Summary:
{ev.summary}

Provide:
1. Any corrections or clarifications needed
2. A confidence level (HIGH/MEDIUM/LOW)
3. The verified and improved content

Response:"""
        
        verification = await self.llm.acomplete(verify_prompt)
        verified_content = str(verification)
        
        # Simple confidence extraction (in production, parse more carefully)
        confidence = "HIGH" if "HIGH" in verified_content.upper() else "MEDIUM"
        
        return VerificationDoneEvent(
            verified_content=verified_content,
            confidence=confidence
        )
    
    @step
    async def respond(self, ctx: Context, ev: VerificationDoneEvent) -> StopEvent:
        """Step 4: Format final response."""
        ctx.write_event_to_stream(Event(msg="💬 Preparing final response..."))
        
        original_query = await ctx.get("original_query")
        source_nodes = await ctx.get("source_nodes")
        
        # Format final response
        format_prompt = f"""Based on this verified information, provide a clear and helpful answer to: "{original_query}"

Verified Content:
{ev.verified_content}

Provide a well-formatted, user-friendly response:"""
        
        final_response = await self.llm.acomplete(format_prompt)
        
        # Build result with metadata
        result = {
            "response": str(final_response),
            "confidence": ev.confidence,
            "sources_used": len(source_nodes),
            "workflow_steps": ["retrieve", "summarize", "verify", "respond"]
        }
        
        return StopEvent(result=result)


async def main():
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Initializing Research Workflow...")
    
    retriever = get_retriever(num_results=5)
    llm = get_llm()
    
    # Create workflow
    workflow = ResearchWorkflow(
        retriever=retriever,
        llm=llm,
        verbose=True,
        timeout=120.0
    )
    
    queries = [
        "How do I set up auto-scaling for a DigitalOcean Kubernetes cluster?",
        "What are the backup options for managed PostgreSQL?",
    ]
    
    for query in queries:
        print("\n" + "="*70)
        print(f"📝 Research Query: {query}")
        print("="*70 + "\n")
        
        # Run workflow with streaming events
        handler = workflow.run(query=query)
        
        async for event in handler.stream_events():
            if hasattr(event, 'msg'):
                print(f"  {event.msg}")
        
        # Get final result
        result = await handler
        
        print("\n" + "-"*70)
        print("📊 Workflow Result:")
        print("-"*70)
        print(f"\n💬 Response:\n{result['response']}")
        print(f"\n🎯 Confidence: {result['confidence']}")
        print(f"📚 Sources: {result['sources_used']} documents")
        print(f"🔄 Steps: {' → '.join(result['workflow_steps'])}")


if __name__ == "__main__":
    asyncio.run(main())

"""
10 - RAG Evaluation

Measure retrieval relevance and answer faithfulness.
"""

import sys
import asyncio
sys.path.insert(0, "../..")

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

from shared.config import get_query_engine, get_llm, Settings


async def evaluate_single_query():
    """Evaluate a single query-response pair."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Setting up RAG Evaluation...")
    
    query_engine = get_query_engine(num_results=5)
    llm = get_llm()
    
    # Create evaluators (using same LLM as judge)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    
    query = "How do I create a Droplet on DigitalOcean?"
    
    print(f"\n📝 Query: {query}\n")
    print("Generating response...")
    
    # Get response
    response = query_engine.query(query)
    
    print(f"\n💬 Response:\n{str(response)[:500]}...\n")
    print(f"📚 Retrieved {len(response.source_nodes)} documents\n")
    
    # Evaluate faithfulness
    print("⏳ Evaluating faithfulness (is answer grounded in sources?)...")
    faith_result = await faithfulness_evaluator.aevaluate_response(
        response=response
    )
    
    # Evaluate relevancy
    print("⏳ Evaluating relevancy (is answer relevant to query?)...")
    rel_result = await relevancy_evaluator.aevaluate_response(
        query=query,
        response=response
    )
    
    # Results
    print("\n" + "="*50)
    print("📊 Evaluation Results")
    print("="*50)
    
    print(f"\n✅ Faithfulness: {faith_result.score or 'N/A'}")
    print(f"   Feedback: {faith_result.feedback or 'N/A'}")
    
    print(f"\n🎯 Relevancy: {rel_result.score or 'N/A'}")
    print(f"   Feedback: {rel_result.feedback or 'N/A'}")
    
    # Interpretation
    print("\n" + "-"*50)
    print("📈 Interpretation:")
    
    if faith_result.score and faith_result.score >= 0.8:
        print("   ✅ High faithfulness - answer is well-grounded in sources")
    elif faith_result.score:
        print("   ⚠️ Low faithfulness - answer may include hallucinations")
    
    if rel_result.score and rel_result.score >= 0.8:
        print("   ✅ High relevancy - answer addresses the question well")
    elif rel_result.score:
        print("   ⚠️ Low relevancy - answer may be off-topic")


async def batch_evaluation():
    """Evaluate multiple queries for more reliable metrics."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    print("🔧 Setting up Batch Evaluation...")
    
    query_engine = get_query_engine(num_results=5)
    llm = get_llm()
    
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    
    # Test queries
    queries = [
        "How do I create a Droplet?",
        "What is DigitalOcean Kubernetes?",
        "How do I set up a managed PostgreSQL database?",
        "What are Spaces used for?",
        "How do I configure a load balancer?",
    ]
    
    print(f"\n📋 Evaluating {len(queries)} queries...\n")
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query[:50]}...")
        
        # Get response
        response = query_engine.query(query)
        
        # Evaluate
        faith = await faithfulness_evaluator.aevaluate_response(response=response)
        rel = await relevancy_evaluator.aevaluate_response(query=query, response=response)
        
        results.append({
            "query": query,
            "faithfulness": faith.score,
            "relevancy": rel.score
        })
    
    # Aggregate results
    print("\n" + "="*60)
    print("📊 Batch Evaluation Results")
    print("="*60)
    
    faith_scores = [r["faithfulness"] for r in results if r["faithfulness"]]
    rel_scores = [r["relevancy"] for r in results if r["relevancy"]]
    
    print(f"\n📈 Faithfulness:")
    print(f"   Average: {sum(faith_scores)/len(faith_scores):.2f}" if faith_scores else "   No scores")
    print(f"   Min: {min(faith_scores):.2f}" if faith_scores else "")
    print(f"   Max: {max(faith_scores):.2f}" if faith_scores else "")
    
    print(f"\n🎯 Relevancy:")
    print(f"   Average: {sum(rel_scores)/len(rel_scores):.2f}" if rel_scores else "   No scores")
    print(f"   Min: {min(rel_scores):.2f}" if rel_scores else "")
    print(f"   Max: {max(rel_scores):.2f}" if rel_scores else "")
    
    print("\n📋 Per-Query Results:")
    for r in results:
        f_score = f"{r['faithfulness']:.2f}" if r['faithfulness'] else "N/A"
        r_score = f"{r['relevancy']:.2f}" if r['relevancy'] else "N/A"
        print(f"   • {r['query'][:40]}... | F: {f_score} | R: {r_score}")


def main():
    print("="*60)
    print("📊 RAG Evaluation Demo")
    print("="*60)
    
    asyncio.run(evaluate_single_query())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation")
    args = parser.parse_args()
    
    if args.batch:
        asyncio.run(batch_evaluation())
    else:
        main()

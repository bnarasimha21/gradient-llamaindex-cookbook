"""
02 - Chat Engine with Memory

Conversational RAG with session history.
"""

import sys
sys.path.insert(0, "../..")

from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

from shared.config import get_retriever, get_llm, Settings


def main():
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials. Copy .env.example to .env and fill in values.")
        return
    
    print("🔧 Initializing Chat Engine with Memory...")
    
    # Initialize components
    retriever = get_retriever(num_results=5)
    llm = get_llm()
    
    # Create memory buffer (stores conversation history)
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    
    # Create chat engine
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        memory=memory,
        verbose=True,  # Shows condensed queries
        system_prompt=(
            "You are a helpful assistant for DigitalOcean documentation. "
            "Answer questions based on the provided context. "
            "If you don't know, say so."
        )
    )
    
    # Simulate a multi-turn conversation
    conversation = [
        "What is a Droplet?",
        "How do I create one?",           # "one" refers to Droplet
        "Can I resize it later?",          # "it" refers to Droplet
        "What about backups?",             # continues Droplet context
    ]
    
    print("\n" + "="*60)
    print("💬 Starting Conversation")
    print("="*60)
    
    for user_message in conversation:
        print(f"\n👤 User: {user_message}")
        
        response = chat_engine.chat(user_message)
        
        print(f"\n🤖 Assistant: {response}")
        print("-"*40)
    
    # Show memory contents
    print("\n📝 Conversation Memory:")
    for msg in memory.get_all():
        role = "👤" if msg.role == "user" else "🤖"
        content = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else msg.content
        print(f"  {role} {content}")


def interactive_mode():
    """Run in interactive mode for manual testing."""
    settings = Settings.from_env()
    if not settings.validate():
        print("❌ Missing credentials.")
        return
    
    retriever = get_retriever(num_results=5)
    llm = get_llm()
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        memory=memory,
        system_prompt="You are a helpful DigitalOcean documentation assistant."
    )
    
    print("💬 Chat with DigitalOcean Docs (type 'quit' to exit)")
    print("-"*50)
    
    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if not user_input:
            continue
            
        response = chat_engine.chat(user_input)
        print(f"\n🤖 Assistant: {response}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true", 
                        help="Run in interactive chat mode")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        main()

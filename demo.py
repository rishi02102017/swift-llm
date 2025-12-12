#!/usr/bin/env python3
"""
SWIFT-LLM Demo Script

Demonstrates the key features of SWIFT-LLM:
1. Semantic caching with instant responses
2. Intelligent query routing
3. Multi-tier model inference
4. Response validation

Usage:
    python demo.py

Make sure to set your API keys in .env file or environment:
    GROQ_API_KEY="your-groq-key"
    OPENAI_API_KEY="your-openai-key"  # Optional
"""

import os
import time

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from swift_llm import SwiftLLM


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_response(response, query: str) -> None:
    """Print a formatted response."""
    print(f"\n[Query]: {query}")
    print("-" * 50)
    print(f"[Response]: {response.text[:200]}..." if len(response.text) > 200 else f"[Response]: {response.text}")
    print("-" * 50)
    print(f"   Latency: {response.total_latency_ms:.1f}ms")
    print(f"   Cache Hit: {'Yes' if response.cache_hit else 'No'}")
    print(f"   Model: {response.model_used} (Tier {response.model_tier})")
    print(f"   Confidence: {response.confidence_score:.2f}")
    print(f"   Cost: ${response.estimated_cost:.6f}")


def demo_basic_usage():
    """Demonstrate basic query functionality."""
    print_header("Demo 1: Basic Usage")
    
    swift = SwiftLLM()
    
    # Simple factual query
    query = "What is the capital of France?"
    response = swift.query(query)
    print_response(response, query)
    
    return swift


def demo_caching(swift: SwiftLLM):
    """Demonstrate semantic caching."""
    print_header("Demo 2: Semantic Caching")
    
    # First query (cache miss)
    query1 = "Explain what machine learning is in simple terms."
    print("\n[First query - will be cached]:")
    response1 = swift.query(query1)
    print_response(response1, query1)
    
    # Similar query (should hit cache)
    query2 = "What is machine learning? Explain simply."
    print("\n[Similar query - should hit cache]:")
    response2 = swift.query(query2)
    print_response(response2, query2)
    
    if response2.cache_hit:
        speedup = response1.total_latency_ms / response2.total_latency_ms
        print(f"\n[Cache hit] {speedup:.1f}x faster than original query")


def demo_routing(swift: SwiftLLM):
    """Demonstrate intelligent query routing."""
    print_header("Demo 3: Query Complexity Routing")
    
    queries = [
        ("Hi", "Simple greeting"),
        ("What is Python?", "Basic factual"),
        ("Compare Python and JavaScript for web development", "Comparison query"),
        ("Write a Python function to calculate fibonacci numbers recursively with memoization", "Complex code request"),
    ]
    
    for query, description in queries:
        print(f"\n[{description}]:")
        response = swift.query(query, skip_cache=True)  # Skip cache to show routing
        print(f"   Query: {query[:50]}...")
        print(f"   Routed to: Tier {response.model_tier} ({response.model_used})")
        print(f"   Latency: {response.total_latency_ms:.1f}ms")


def demo_conversation(swift: SwiftLLM):
    """Demonstrate multi-turn conversation."""
    print_header("Demo 4: Multi-turn Conversation")
    
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "I'm learning Python."},
        {"role": "assistant", "content": "That's great! Python is an excellent language to learn. What would you like to know?"},
        {"role": "user", "content": "What are lists and how do I use them?"},
    ]
    
    print("\n[Conversation history]:")
    for msg in messages:
        role_label = msg["role"].upper()
        print(f"   [{role_label}]: {msg['content'][:50]}...")
    
    response = swift.chat(messages)
    print(f"\n[Assistant response]:")
    print(f"   {response.text[:300]}...")
    print(f"\n   Latency: {response.total_latency_ms:.1f}ms")


def demo_stats(swift: SwiftLLM):
    """Show collected statistics."""
    print_header("Demo 5: Performance Statistics")
    swift.print_stats()


def main():
    """Run the full demo."""
    print("\n" + "=" * 60)
    print("   SWIFT-LLM: Semantic-Aware Intelligent Fast Inference")
    print("=" * 60)
    
    # Check for API keys
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n[Warning] No API keys found!")
        print("   Set GROQ_API_KEY for fast, free inference:")
        print("   export GROQ_API_KEY='your-key-here'")
        print("\n   Get a free key at: https://console.groq.com/")
        print("\n   Running in demo mode with limited functionality...\n")
    
    try:
        # Run demos
        swift = demo_basic_usage()
        demo_caching(swift)
        demo_routing(swift)
        demo_conversation(swift)
        demo_stats(swift)
        
        # Save cache
        swift.save_cache()
        
        print("\n" + "=" * 60)
        print("  Demo completed successfully")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n[Error]: {e}")
        print("\nMake sure you have set up your API keys:")
        print("  export GROQ_API_KEY='your-key'")
        raise


if __name__ == "__main__":
    main()

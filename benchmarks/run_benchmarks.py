#!/usr/bin/env python3
"""
SWIFT-LLM Benchmark Suite

Evaluates the performance of SWIFT-LLM across multiple dimensions:
- Latency (with and without caching)
- Cache hit rates
- Routing accuracy
- Cost savings

Usage:
    python -m benchmarks.run_benchmarks
    python -m benchmarks.run_benchmarks --samples 100
    python -m benchmarks.run_benchmarks --no-cache
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import random

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from swift_llm import SwiftLLM
from swift_llm.core.config import Config


# Sample queries for benchmarking
BENCHMARK_QUERIES = {
    "simple": [
        "What is 2+2?",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
        "What is the capital of Japan?",
        "How many days in a week?",
        "What is H2O?",
        "Who painted the Mona Lisa?",
        "What planet is closest to the sun?",
        "What is the largest ocean?",
        "How many continents are there?",
    ],
    "moderate": [
        "Explain the difference between a list and a tuple in Python.",
        "What are the main causes of climate change?",
        "How does photosynthesis work?",
        "Describe the process of machine learning model training.",
        "What is the difference between HTTP and HTTPS?",
        "Explain how a blockchain works.",
        "What are the benefits of renewable energy?",
        "How do vaccines work to prevent diseases?",
        "Describe the water cycle.",
        "What is object-oriented programming?",
    ],
    "complex": [
        "Compare and contrast supervised and unsupervised machine learning, providing examples of each.",
        "Explain the concept of attention mechanisms in transformer models and why they are important.",
        "Describe the CAP theorem in distributed systems and its implications for database design.",
        "Analyze the trade-offs between microservices and monolithic architectures.",
        "Explain how gradient descent optimization works in neural networks.",
        "Compare REST and GraphQL APIs, discussing when to use each.",
        "Describe the principles of SOLID design in software engineering.",
        "Explain the concept of eventual consistency in distributed databases.",
        "Analyze the security implications of different authentication methods.",
        "Discuss the ethical considerations in AI development.",
    ],
    "code": [
        "Write a Python function to reverse a string.",
        "Implement a binary search algorithm in Python.",
        "Write a function to check if a number is prime.",
        "Create a Python class for a simple linked list.",
        "Write a function to find the nth Fibonacci number.",
        "Implement quicksort in Python.",
        "Write a function to detect palindromes.",
        "Create a simple REST API endpoint using Flask.",
        "Write a function to merge two sorted arrays.",
        "Implement a basic LRU cache in Python.",
    ],
}


def generate_similar_queries(original: str, count: int = 3) -> List[str]:
    """Generate semantically similar queries for cache testing."""
    variations = [
        f"Can you tell me {original.lower()}",
        f"I want to know {original.lower()}",
        f"Please explain {original.lower()}",
        f"{original} Please be concise.",
        f"Briefly, {original.lower()}",
    ]
    return random.sample(variations, min(count, len(variations)))


def run_latency_benchmark(
    swift: SwiftLLM,
    queries: List[str],
    category: str,
) -> Dict[str, Any]:
    """Benchmark latency for a set of queries."""
    results = {
        "category": category,
        "total_queries": len(queries),
        "latencies_ms": [],
        "cache_hits": 0,
        "cache_misses": 0,
        "tier_distribution": {},
        "total_cost": 0.0,
    }
    
    print(f"\n  Running {category} queries ({len(queries)} samples)...")
    
    for i, query in enumerate(queries):
        response = swift.query(query)
        
        results["latencies_ms"].append(response.total_latency_ms)
        results["total_cost"] += response.estimated_cost
        
        if response.cache_hit:
            results["cache_hits"] += 1
        else:
            results["cache_misses"] += 1
        
        tier = response.model_tier
        results["tier_distribution"][tier] = results["tier_distribution"].get(tier, 0) + 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(queries)} queries...")
    
    # Calculate statistics
    latencies = results["latencies_ms"]
    results["avg_latency_ms"] = sum(latencies) / len(latencies)
    results["min_latency_ms"] = min(latencies)
    results["max_latency_ms"] = max(latencies)
    
    sorted_latencies = sorted(latencies)
    results["p50_latency_ms"] = sorted_latencies[len(latencies) // 2]
    results["p95_latency_ms"] = sorted_latencies[int(len(latencies) * 0.95)]
    results["p99_latency_ms"] = sorted_latencies[int(len(latencies) * 0.99)]
    
    results["cache_hit_rate"] = results["cache_hits"] / len(queries)
    
    return results


def run_cache_effectiveness_benchmark(
    swift: SwiftLLM,
    base_queries: List[str],
) -> Dict[str, Any]:
    """Benchmark cache effectiveness with similar queries."""
    results = {
        "original_queries": len(base_queries),
        "similar_queries_per_original": 3,
        "original_latencies": [],
        "similar_latencies": [],
        "cache_hits_on_similar": 0,
    }
    
    print("\n  Testing cache effectiveness with similar queries...")
    
    for i, query in enumerate(base_queries[:10]):  # Limit for demo
        # First, run original query
        response1 = swift.query(query)
        results["original_latencies"].append(response1.total_latency_ms)
        
        # Then run similar queries
        similar = generate_similar_queries(query, 3)
        for sim_query in similar:
            response2 = swift.query(sim_query)
            results["similar_latencies"].append(response2.total_latency_ms)
            if response2.cache_hit:
                results["cache_hits_on_similar"] += 1
    
    total_similar = len(results["similar_latencies"])
    results["similar_cache_hit_rate"] = (
        results["cache_hits_on_similar"] / total_similar if total_similar > 0 else 0
    )
    results["avg_original_latency"] = (
        sum(results["original_latencies"]) / len(results["original_latencies"])
        if results["original_latencies"] else 0
    )
    results["avg_similar_latency"] = (
        sum(results["similar_latencies"]) / len(results["similar_latencies"])
        if results["similar_latencies"] else 0
    )
    
    if results["avg_similar_latency"] > 0:
        results["speedup_factor"] = results["avg_original_latency"] / results["avg_similar_latency"]
    else:
        results["speedup_factor"] = 1.0
    
    return results


def run_routing_accuracy_benchmark(swift: SwiftLLM) -> Dict[str, Any]:
    """Benchmark routing decisions against expected tiers."""
    # Expected tier mappings (approximate)
    test_cases = [
        ("Hi", 1, "greeting"),
        ("What is 2+2?", 1, "simple math"),
        ("What is Python?", 2, "basic factual"),
        ("Explain machine learning", 2, "explanation"),
        ("Compare Python and Java", 3, "comparison"),
        ("Write a binary search in Python", 3, "code simple"),
        ("Implement a transformer attention mechanism from scratch", 4, "code complex"),
    ]
    
    results = {
        "total_tests": len(test_cases),
        "correct_routing": 0,
        "routing_details": [],
    }
    
    print("\n  Testing routing accuracy...")
    
    for query, expected_tier, description in test_cases:
        response = swift.query(query, skip_cache=True)
        actual_tier = response.model_tier
        
        # Allow +/- 1 tier tolerance
        is_correct = abs(actual_tier - expected_tier) <= 1
        if is_correct:
            results["correct_routing"] += 1
        
        results["routing_details"].append({
            "query": query[:30] + "..." if len(query) > 30 else query,
            "description": description,
            "expected_tier": expected_tier,
            "actual_tier": actual_tier,
            "correct": is_correct,
        })
    
    results["accuracy"] = results["correct_routing"] / results["total_tests"]
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a formatted way."""
    print("\n" + "=" * 70)
    print("                    SWIFT-LLM BENCHMARK RESULTS")
    print("=" * 70)
    
    # Latency results
    if "latency_results" in results:
        print("\n[LATENCY BENCHMARKS]")
        print("-" * 50)
        for category, data in results["latency_results"].items():
            print(f"\n  {category.upper()}:")
            print(f"    Queries:        {data['total_queries']}")
            print(f"    Avg Latency:    {data['avg_latency_ms']:.1f}ms")
            print(f"    P50 Latency:    {data['p50_latency_ms']:.1f}ms")
            print(f"    P95 Latency:    {data['p95_latency_ms']:.1f}ms")
            print(f"    Cache Hit Rate: {data['cache_hit_rate']:.1%}")
            print(f"    Total Cost:     ${data['total_cost']:.4f}")
    
    # Cache effectiveness
    if "cache_results" in results:
        print("\n[CACHE EFFECTIVENESS]")
        print("-" * 50)
        data = results["cache_results"]
        print(f"    Similar Query Hit Rate: {data['similar_cache_hit_rate']:.1%}")
        print(f"    Avg Original Latency:   {data['avg_original_latency']:.1f}ms")
        print(f"    Avg Similar Latency:    {data['avg_similar_latency']:.1f}ms")
        print(f"    Speedup Factor:         {data['speedup_factor']:.1f}x")
    
    # Routing accuracy
    if "routing_results" in results:
        print("\n[ROUTING ACCURACY]")
        print("-" * 50)
        data = results["routing_results"]
        print(f"    Accuracy: {data['accuracy']:.1%}")
        print("\n    Routing Details:")
        for detail in data["routing_details"]:
            status = "[OK]" if detail["correct"] else "[X]"
            print(f"      {status} '{detail['query']}' -> Tier {detail['actual_tier']} (expected {detail['expected_tier']})")
    
    # Overall stats
    if "swift_stats" in results:
        print("\n[OVERALL STATISTICS]")
        print("-" * 50)
        stats = results["swift_stats"]
        if "metrics" in stats:
            m = stats["metrics"]
            print(f"    Total Queries:      {m.get('total_queries', 'N/A')}")
            print(f"    Overall Cache Rate: {m.get('cache_hit_rate', 0):.1%}")
            print(f"    Avg Latency:        {m.get('avg_latency_ms', 0):.1f}ms")
            print(f"    Total Cost:         ${m.get('total_cost', 0):.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="SWIFT-LLM Benchmark Suite")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per category")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("     SWIFT-LLM BENCHMARK SUITE")
    print("=" * 50)
    
    # Initialize SWIFT-LLM
    print("\n[Initializing SWIFT-LLM...]")
    swift = SwiftLLM(enable_cache=not args.no_cache)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "samples_per_category": args.samples,
            "cache_enabled": not args.no_cache,
        },
    }
    
    # Run latency benchmarks
    print("\n" + "=" * 50)
    print("Running Latency Benchmarks...")
    print("=" * 50)
    
    results["latency_results"] = {}
    for category, queries in BENCHMARK_QUERIES.items():
        sample_queries = queries[:args.samples]
        results["latency_results"][category] = run_latency_benchmark(
            swift, sample_queries, category
        )
    
    # Run cache effectiveness benchmark
    if not args.no_cache:
        print("\n" + "=" * 50)
        print("Running Cache Effectiveness Benchmark...")
        print("=" * 50)
        
        results["cache_results"] = run_cache_effectiveness_benchmark(
            swift, BENCHMARK_QUERIES["moderate"]
        )
    
    # Run routing accuracy benchmark
    print("\n" + "=" * 50)
    print("Running Routing Accuracy Benchmark...")
    print("=" * 50)
    
    results["routing_results"] = run_routing_accuracy_benchmark(swift)
    
    # Get overall stats
    results["swift_stats"] = swift.get_stats()
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        # Remove non-serializable data
        clean_results = json.loads(json.dumps(results, default=str))
        with open(output_path, "w") as f:
            json.dump(clean_results, f, indent=2)
        print(f"\n[Results saved to: {output_path}]")
    
    # Save to default location
    default_output = Path("benchmarks/results") / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    default_output.parent.mkdir(parents=True, exist_ok=True)
    clean_results = json.loads(json.dumps(results, default=str))
    with open(default_output, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"[Results saved to: {default_output}]")
    
    print("\n[Benchmark complete]")


if __name__ == "__main__":
    main()

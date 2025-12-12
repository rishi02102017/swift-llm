"""
Cache Warming Utilities for SWIFT-LLM.

Pre-populate the semantic cache with common queries to ensure
fast responses from system startup.
"""

from typing import List, Tuple, Optional
from swift_llm.cache.semantic_cache import SemanticCache


# Common factual queries that are frequently asked
COMMON_FACTUAL_QUERIES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of United Kingdom?", "The capital of the United Kingdom is London."),
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("What is the capital of China?", "The capital of China is Beijing."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
]

# Technical definition queries
TECH_DEFINITION_QUERIES = [
    ("What is machine learning?", 
     "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."),
    ("What is deep learning?",
     "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to learn representations of data with multiple levels of abstraction. It's particularly effective for complex tasks like image recognition and natural language processing."),
    ("What is artificial intelligence?",
     "Artificial intelligence (AI) is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. AI encompasses machine learning, natural language processing, robotics, and more."),
    ("What is a neural network?",
     "A neural network is a computing system inspired by biological neural networks in the human brain. It consists of interconnected nodes (neurons) organized in layers that process information and learn patterns from data."),
    ("What is Python?",
     "Python is a high-level, interpreted programming language known for its clear syntax and readability. It supports multiple programming paradigms and is widely used in web development, data science, artificial intelligence, and automation."),
    ("What is an API?",
     "An API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate with each other. It defines the methods and data formats that programs can use to interact."),
    ("What is a database?",
     "A database is an organized collection of structured data stored electronically. It allows efficient storage, retrieval, and management of data. Common types include relational databases (SQL) and NoSQL databases."),
    ("What is cloud computing?",
     "Cloud computing is the delivery of computing services over the internet, including servers, storage, databases, networking, software, and analytics. It offers faster innovation, flexible resources, and economies of scale."),
]

# Simple greeting responses
GREETING_QUERIES = [
    ("Hello", "Hello! How can I help you today?"),
    ("Hi", "Hi there! What can I assist you with?"),
    ("Hey", "Hey! How can I help you?"),
    ("Good morning", "Good morning! How can I assist you today?"),
    ("Good afternoon", "Good afternoon! What can I help you with?"),
    ("Good evening", "Good evening! How may I assist you?"),
]

# Math basics
MATH_QUERIES = [
    ("What is 2+2?", "2 + 2 = 4"),
    ("What is 10 times 10?", "10 times 10 equals 100."),
    ("What is the square root of 16?", "The square root of 16 is 4."),
    ("What is 100 divided by 5?", "100 divided by 5 equals 20."),
]


def get_warmup_queries(
    include_factual: bool = True,
    include_tech: bool = True,
    include_greetings: bool = True,
    include_math: bool = True,
) -> List[Tuple[str, str, int]]:
    """
    Get a list of queries for cache warming.
    
    Args:
        include_factual: Include common factual queries
        include_tech: Include technical definition queries
        include_greetings: Include greeting queries
        include_math: Include basic math queries
        
    Returns:
        List of (query, response, tier) tuples
    """
    queries = []
    
    if include_greetings:
        queries.extend([(q, r, 1) for q, r in GREETING_QUERIES])
    
    if include_math:
        queries.extend([(q, r, 1) for q, r in MATH_QUERIES])
    
    if include_factual:
        queries.extend([(q, r, 1) for q, r in COMMON_FACTUAL_QUERIES])
    
    if include_tech:
        queries.extend([(q, r, 2) for q, r in TECH_DEFINITION_QUERIES])
    
    return queries


def warm_cache(
    cache: SemanticCache,
    queries: Optional[List[Tuple[str, str, int]]] = None,
    verbose: bool = True,
) -> int:
    """
    Pre-populate the cache with common queries.
    
    Args:
        cache: The semantic cache to warm
        queries: Optional list of (query, response, tier) tuples
        verbose: Print progress
        
    Returns:
        Number of entries added
    """
    if queries is None:
        queries = get_warmup_queries()
    
    if verbose:
        print(f"Warming cache with {len(queries)} queries...")
    
    count = cache.warm_cache(queries)
    
    if verbose:
        print(f"Cache warmed with {count} entries.")
        print(f"Cache size: {len(cache)}")
    
    return count


def warm_cache_from_file(
    cache: SemanticCache,
    file_path: str,
    verbose: bool = True,
) -> int:
    """
    Warm cache from a JSON file containing query-response pairs.
    
    Expected format:
    [
        {"query": "...", "response": "...", "tier": 2},
        ...
    ]
    
    Args:
        cache: The semantic cache to warm
        file_path: Path to JSON file
        verbose: Print progress
        
    Returns:
        Number of entries added
    """
    import json
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    queries = [
        (item["query"], item["response"], item.get("tier", 2))
        for item in data
    ]
    
    return warm_cache(cache, queries, verbose)


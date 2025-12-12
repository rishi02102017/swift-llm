"""
SWIFT-LLM: Semantic-Aware Intelligent Fast Inference with Tiered Routing

A multi-layer optimization framework for production-grade LLM systems.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from swift_llm.core.pipeline import SwiftLLM
from swift_llm.core.config import Config
from swift_llm.cache.semantic_cache import SemanticCache
from swift_llm.router.routing_policy import QueryRouter

__all__ = [
    "SwiftLLM",
    "Config", 
    "SemanticCache",
    "QueryRouter",
]


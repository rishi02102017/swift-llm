"""
Query Router for SWIFT-LLM.

Orchestrates the routing decision based on query complexity,
cache status, and model availability.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import time

from swift_llm.router.complexity_classifier import ComplexityClassifier, QueryFeatures
from swift_llm.router.model_registry import ModelRegistry, ModelInfo
from swift_llm.cache.semantic_cache import SemanticCache, CacheResult
from swift_llm.core.config import RouterConfig


@dataclass
class RoutingDecision:
    """Result of the routing decision."""
    
    # Primary decision
    tier: int
    model_name: str
    model_info: Optional[ModelInfo]
    
    # From cache check
    cache_hit: bool
    cached_response: Optional[str] = None
    cache_similarity: float = 0.0
    
    # From classification
    complexity_score: float = 0.0
    query_features: Optional[QueryFeatures] = None
    
    # Timing
    routing_time_ms: float = 0.0
    
    # Reasoning
    routing_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier": self.tier,
            "model_name": self.model_name,
            "cache_hit": self.cache_hit,
            "cache_similarity": self.cache_similarity,
            "complexity_score": self.complexity_score,
            "routing_time_ms": self.routing_time_ms,
            "routing_reason": self.routing_reason,
        }


class QueryRouter:
    """
    Intelligent query router for SWIFT-LLM.
    
    Combines semantic caching with complexity-based model selection
    to optimize for latency, cost, and quality.
    
    Routing Flow:
    1. Check semantic cache for similar queries
    2. If cache miss, classify query complexity
    3. Select appropriate model tier
    4. Return routing decision
    
    Example:
        router = QueryRouter()
        decision = router.route("What is machine learning?")
        
        if decision.cache_hit:
            return decision.cached_response
        else:
            response = inference_engine.generate(
                query, 
                model=decision.model_name
            )
    """
    
    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        classifier: Optional[ComplexityClassifier] = None,
        registry: Optional[ModelRegistry] = None,
        config: Optional[RouterConfig] = None,
    ):
        """
        Initialize the router.
        
        Args:
            cache: Semantic cache instance (creates new if None)
            classifier: Complexity classifier (creates new if None)
            registry: Model registry (creates new if None)
            config: Router configuration
        """
        self.config = config or RouterConfig()
        
        # Initialize components
        self.cache = cache  # May be None, handled in route()
        self.classifier = classifier or ComplexityClassifier(
            tier_thresholds=self.config.tier_thresholds
        )
        self.registry = registry or ModelRegistry()
        
        # Statistics
        self.stats = {
            "total_routes": 0,
            "cache_routes": 0,
            "tier_distribution": {1: 0, 2: 0, 3: 0, 4: 0},
        }
    
    def route(self, query: str, skip_cache: bool = False) -> RoutingDecision:
        """
        Route a query to the appropriate model tier.
        
        Args:
            query: The user query to route
            skip_cache: If True, skip cache lookup
            
        Returns:
            RoutingDecision with tier, model, and cache info
        """
        start_time = time.perf_counter()
        self.stats["total_routes"] += 1
        
        # Step 1: Check cache (if available and not skipped)
        cache_result = CacheResult(hit=False)
        if self.cache is not None and not skip_cache:
            cache_result = self.cache.lookup(query)
            
            if cache_result.hit:
                self.stats["cache_routes"] += 1
                self.stats["tier_distribution"][1] += 1
                
                routing_time = (time.perf_counter() - start_time) * 1000
                
                return RoutingDecision(
                    tier=1,
                    model_name="cache",
                    model_info=self.registry.get("cache"),
                    cache_hit=True,
                    cached_response=cache_result.response,
                    cache_similarity=cache_result.similarity,
                    complexity_score=0.0,
                    routing_time_ms=routing_time,
                    routing_reason=f"Cache hit with {cache_result.similarity:.2%} similarity",
                )
        
        # Step 2: Classify query complexity
        tier, complexity_score, features = self.classifier.classify(query)
        
        # Step 3: Get model for tier
        model_info = self.registry.get_for_tier(tier)
        model_name = model_info.name if model_info else f"tier_{tier}_model"
        
        # Step 4: Apply any overrides or adjustments
        final_tier, final_model, reason = self._apply_routing_rules(
            tier, model_info, features, complexity_score
        )
        
        # Update statistics
        self.stats["tier_distribution"][final_tier] = \
            self.stats["tier_distribution"].get(final_tier, 0) + 1
        
        routing_time = (time.perf_counter() - start_time) * 1000
        
        return RoutingDecision(
            tier=final_tier,
            model_name=final_model.name if final_model else model_name,
            model_info=final_model,
            cache_hit=False,
            cache_similarity=cache_result.similarity if cache_result else 0.0,
            complexity_score=complexity_score,
            query_features=features,
            routing_time_ms=routing_time,
            routing_reason=reason,
        )
    
    def _apply_routing_rules(
        self,
        tier: int,
        model_info: Optional[ModelInfo],
        features: QueryFeatures,
        complexity_score: float,
    ) -> Tuple[int, Optional[ModelInfo], str]:
        """
        Apply additional routing rules and adjustments.
        
        Returns:
            Tuple of (final_tier, final_model, reason)
        """
        reason_parts = [f"Complexity: {complexity_score:.2f}"]
        
        # Rule 0: Simple factual "What is X?" queries should stay tier 2
        if (features.word_count < 8 and 
            features.question_words > 0 and 
            not features.has_code_request and
            not features.has_comparison and
            not features.has_reasoning_request):
            tier = min(tier, 2)
            model_info = self.registry.get_for_tier(tier)
            reason_parts.append("Simple factual query")
        
        # Rule 1: Code WRITING requests should use at least tier 3
        if features.has_code_request and tier < 3:
            tier = 3
            model_info = self.registry.get_for_tier(tier)
            reason_parts.append("Upgraded for code request")
        
        # Rule 2: Very long queries might need more context
        if features.word_count > 100 and tier < 3:
            tier = 3
            model_info = self.registry.get_for_tier(tier)
            reason_parts.append("Upgraded for long query")
        
        # Rule 3: Multiple questions need better model
        if features.has_multiple_questions and tier < 3:
            tier = 3
            model_info = self.registry.get_for_tier(tier)
            reason_parts.append("Upgraded for multiple questions")
        
        # Rule 4: Comparison requests benefit from better reasoning
        if features.has_comparison and tier < 3:
            tier = 3
            model_info = self.registry.get_for_tier(tier)
            reason_parts.append("Upgraded for comparison request")
        
        # Rule 5: Check model availability, fallback if needed
        if model_info is None:
            model_info = self.registry.get_fallback_model(tier)
            if model_info:
                tier = model_info.tier
                reason_parts.append(f"Fallback to tier {tier}")
        
        # Default to tier 2 if everything fails
        if model_info is None:
            tier = self.config.default_tier
            model_info = self.registry.get_for_tier(tier)
            reason_parts.append(f"Default to tier {tier}")
        
        return tier, model_info, "; ".join(reason_parts)
    
    def get_escalation_model(self, current_tier: int) -> Optional[ModelInfo]:
        """Get the next tier model for escalation."""
        return self.registry.get_next_tier_model(current_tier)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total = self.stats["total_routes"]
        cache_rate = self.stats["cache_routes"] / total if total > 0 else 0
        
        tier_pcts = {}
        for tier, count in self.stats["tier_distribution"].items():
            tier_pcts[f"tier_{tier}_pct"] = count / total if total > 0 else 0
        
        return {
            **self.stats,
            "cache_route_rate": cache_rate,
            **tier_pcts,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total_routes": 0,
            "cache_routes": 0,
            "tier_distribution": {1: 0, 2: 0, 3: 0, 4: 0},
        }


# Convenience function
def create_router(
    enable_cache: bool = True,
    cache_threshold: float = 0.85,
    tier_thresholds: Optional[list] = None,
) -> QueryRouter:
    """
    Create a query router with common settings.
    
    Args:
        enable_cache: Whether to enable semantic caching
        cache_threshold: Similarity threshold for cache hits
        tier_thresholds: Complexity thresholds for tier assignment
        
    Returns:
        Configured QueryRouter instance
    """
    from swift_llm.cache.semantic_cache import SemanticCache
    from swift_llm.core.config import CacheConfig
    
    cache = None
    if enable_cache:
        cache_config = CacheConfig(similarity_threshold=cache_threshold)
        cache = SemanticCache(config=cache_config)
    
    config = RouterConfig(
        tier_thresholds=tier_thresholds or [0.25, 0.5, 0.75]
    )
    
    return QueryRouter(cache=cache, config=config)


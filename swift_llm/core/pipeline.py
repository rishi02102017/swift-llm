"""
Main Pipeline for SWIFT-LLM.

Orchestrates all components: cache, router, inference, and validation.
"""

import atexit
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from swift_llm.core.config import Config
from swift_llm.core.metrics import MetricsLogger, QueryMetrics, Timer
from swift_llm.cache.semantic_cache import SemanticCache
from swift_llm.router.routing_policy import QueryRouter, RoutingDecision
from swift_llm.router.model_registry import ModelRegistry
from swift_llm.inference.api_inference import APIInference
from swift_llm.inference.base import Message, InferenceResult
from swift_llm.validation.validator import ResponseValidator, ValidationResult


@dataclass
class SwiftResponse:
    """Response from the SWIFT-LLM system."""
    
    # Main content
    text: str
    
    # Source information
    cache_hit: bool
    model_used: str
    model_tier: int
    
    # Performance
    total_latency_ms: float
    cache_lookup_ms: float = 0.0
    routing_ms: float = 0.0
    inference_ms: float = 0.0
    validation_ms: float = 0.0
    
    # Quality
    confidence_score: float = 1.0
    was_escalated: bool = False
    escalation_count: int = 0
    
    # Cost
    estimated_cost: float = 0.0
    tokens_used: int = 0
    
    # Metadata
    query_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "cache_hit": self.cache_hit,
            "model_used": self.model_used,
            "model_tier": self.model_tier,
            "total_latency_ms": self.total_latency_ms,
            "confidence_score": self.confidence_score,
            "was_escalated": self.was_escalated,
            "estimated_cost": self.estimated_cost,
            "tokens_used": self.tokens_used,
            "query_id": self.query_id,
        }


class SwiftLLM:
    """
    SWIFT-LLM: Semantic-Aware Intelligent Fast Inference with Tiered Routing.
    
    The main entry point for the optimization framework.
    
    Features:
    - Semantic caching for instant responses to similar queries
    - Intelligent query routing based on complexity
    - Multi-tier model inference (Groq, OpenAI)
    - Response validation with automatic escalation
    - Comprehensive metrics collection
    
    Example:
        # Basic usage
        swift = SwiftLLM()
        response = swift.query("What is the capital of France?")
        print(response.text)
        print(f"Latency: {response.total_latency_ms}ms")
        
        # With conversation history
        response = swift.chat([
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What's the weather like?"},
        ])
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        enable_cache: bool = True,
        enable_metrics: bool = True,
    ):
        """
        Initialize the SWIFT-LLM system.
        
        Args:
            config: Configuration object (uses defaults if None)
            enable_cache: Whether to enable semantic caching
            enable_metrics: Whether to collect metrics
        """
        self.config = config or Config.from_env()
        
        # Initialize components
        print("Initializing SWIFT-LLM...")
        
        # Semantic cache
        self.cache = None
        if enable_cache and self.config.cache.enabled:
            print("  Loading semantic cache...")
            self.cache = SemanticCache(config=self.config.cache)
        
        # Model registry
        self.registry = ModelRegistry()
        
        # Query router
        self.router = QueryRouter(
            cache=self.cache,
            registry=self.registry,
            config=self.config.router,
        )
        
        # Inference backends
        print("  Initializing inference backends...")
        self.inference = APIInference(
            groq_api_key=self.config.inference.groq_api_key,
            openai_api_key=self.config.inference.openai_api_key,
        )
        
        # Print available backends
        available = self.inference.list_available_backends()
        if available:
            for backend in available:
                print(f"    [OK] {backend}")
        else:
            print("    [Warning] No API backends available. Set GROQ_API_KEY or OPENAI_API_KEY.")
        
        # Validator
        self.validator = ResponseValidator(
            confidence_threshold=self.config.validation.confidence_threshold,
            cache_threshold=0.5,  # Lower threshold to cache more responses
            enable_escalation=self.config.validation.enable_auto_escalation,
        )
        
        # Metrics
        self.metrics = MetricsLogger(
            log_file=self.config.metrics_file,
            enabled=enable_metrics,
        )
        
        # Register automatic cache persistence on exit
        atexit.register(self._on_exit)
        
        print("SWIFT-LLM initialized successfully.\n")
    
    def _on_exit(self) -> None:
        """Called on program exit to persist cache."""
        if self.cache:
            try:
                self.cache.save()
            except Exception:
                pass  # Silently handle errors during exit
    
    def query(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        skip_cache: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> SwiftResponse:
        """
        Process a single query through the SWIFT-LLM pipeline.
        
        Args:
            query: The user's query
            system_prompt: Optional system prompt
            skip_cache: If True, skip cache lookup
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            SwiftResponse with result and metadata
        """
        start_time = time.perf_counter()
        query_id = self.metrics.generate_query_id()
        
        # Initialize timing
        cache_time = 0.0
        routing_time = 0.0
        inference_time = 0.0
        validation_time = 0.0
        
        # Step 1: Route the query (includes cache lookup)
        with Timer() as route_timer:
            routing_decision = self.router.route(query, skip_cache=skip_cache)
        routing_time = route_timer.elapsed_ms
        cache_time = routing_decision.routing_time_ms  # Cache lookup is part of routing
        
        # Check for cache hit
        if routing_decision.cache_hit:
            total_time = (time.perf_counter() - start_time) * 1000
            
            response = SwiftResponse(
                text=routing_decision.cached_response,
                cache_hit=True,
                model_used="cache",
                model_tier=1,
                total_latency_ms=total_time,
                cache_lookup_ms=cache_time,
                routing_ms=routing_time,
                confidence_score=1.0,  # Cached responses are trusted
                query_id=query_id,
            )
            
            self._log_metrics(query, response, routing_decision)
            return response
        
        # Step 2: Generate response
        messages = self._build_messages(query, system_prompt)
        
        with Timer() as inference_timer:
            inference_result = self.inference.generate(
                tier=routing_decision.tier,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        inference_time = inference_timer.elapsed_ms
        
        # Step 3: Validate response
        with Timer() as validation_timer:
            validation = self.validator.validate(
                response=inference_result.text,
                query=query,
                inference_result=inference_result,
            )
        validation_time = validation_timer.elapsed_ms
        
        # Step 4: Handle escalation if needed
        escalation_count = 0
        current_tier = routing_decision.tier
        
        while (
            validation.should_escalate and
            escalation_count < self.config.validation.max_escalations
        ):
            escalation_count += 1
            
            # Get next tier model
            next_model = self.router.get_escalation_model(current_tier)
            if next_model is None:
                break
            
            current_tier = next_model.tier
            
            # Retry with better model
            with Timer() as retry_timer:
                inference_result = self.inference.generate(
                    tier=current_tier,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            inference_time += retry_timer.elapsed_ms
            
            # Re-validate
            validation = self.validator.validate(
                response=inference_result.text,
                query=query,
                inference_result=inference_result,
            )
        
        # Step 5: Cache the response if appropriate
        cache_conditions = (self.cache is not None, validation.should_cache, inference_result.success)
        if all(cache_conditions):
            try:
                self.cache.store(
                    query=query,
                    response=inference_result.text,
                    model_tier=current_tier,
                    confidence_score=validation.confidence.score,
                )
            except Exception as e:
                print(f"[Cache Error] {e}")
        
        # Build response
        total_time = (time.perf_counter() - start_time) * 1000
        
        response = SwiftResponse(
            text=inference_result.text,
            cache_hit=False,
            model_used=inference_result.model,
            model_tier=current_tier,
            total_latency_ms=total_time,
            cache_lookup_ms=cache_time,
            routing_ms=routing_time,
            inference_ms=inference_time,
            validation_ms=validation_time,
            confidence_score=validation.confidence.score,
            was_escalated=escalation_count > 0,
            escalation_count=escalation_count,
            estimated_cost=inference_result.estimated_cost,
            tokens_used=inference_result.tokens_input + inference_result.tokens_output,
            query_id=query_id,
        )
        
        self._log_metrics(query, response, routing_decision)
        return response
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        skip_cache: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> SwiftResponse:
        """
        Process a multi-turn conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            skip_cache: If True, skip cache lookup
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            SwiftResponse with result
        """
        # Extract the latest user message for routing
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            raise ValueError("No user messages found")
        
        latest_query = user_messages[-1]["content"]
        
        # For chat, we use full message history
        # Convert to Message objects
        chat_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        
        # Route based on latest query (for cache and complexity)
        start_time = time.perf_counter()
        query_id = self.metrics.generate_query_id()
        
        with Timer() as route_timer:
            routing_decision = self.router.route(latest_query, skip_cache=skip_cache)
        
        # Skip cache for multi-turn (context matters)
        # But use routing for tier selection
        
        # Generate with full context
        with Timer() as inference_timer:
            inference_result = self.inference.generate(
                tier=routing_decision.tier,
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return SwiftResponse(
            text=inference_result.text,
            cache_hit=False,
            model_used=inference_result.model,
            model_tier=routing_decision.tier,
            total_latency_ms=total_time,
            routing_ms=route_timer.elapsed_ms,
            inference_ms=inference_timer.elapsed_ms,
            confidence_score=1.0,  # Skip validation for chat
            estimated_cost=inference_result.estimated_cost,
            tokens_used=inference_result.tokens_input + inference_result.tokens_output,
            query_id=query_id,
        )
    
    def _build_messages(
        self,
        query: str,
        system_prompt: Optional[str] = None,
    ) -> List[Message]:
        """Build message list for inference."""
        messages = []
        
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        
        messages.append(Message(role="user", content=query))
        
        return messages
    
    def _log_metrics(
        self,
        query: str,
        response: SwiftResponse,
        routing: RoutingDecision,
    ) -> None:
        """Log query metrics."""
        metrics = QueryMetrics(
            query_id=response.query_id,
            timestamp=response.timestamp.isoformat(),
            query_text=query[:200],  # Truncate for storage
            query_length=len(query),
            predicted_tier=routing.tier,
            actual_tier=response.model_tier,
            complexity_score=routing.complexity_score,
            cache_hit=response.cache_hit,
            cache_similarity=routing.cache_similarity,
            total_latency_ms=response.total_latency_ms,
            cache_lookup_ms=response.cache_lookup_ms,
            routing_ms=response.routing_ms,
            inference_ms=response.inference_ms,
            validation_ms=response.validation_ms,
            response_length=len(response.text),
            confidence_score=response.confidence_score,
            was_escalated=response.was_escalated,
            escalation_count=response.escalation_count,
            model_used=response.model_used,
            tokens_used=response.tokens_used,
            estimated_cost=response.estimated_cost,
        )
        self.metrics.log_query(metrics)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "metrics": self.metrics.get_aggregate_metrics().__dict__,
            "router": self.router.get_stats(),
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    def print_stats(self) -> None:
        """Print performance summary."""
        self.metrics.print_summary()
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            print(f"Cache Size: {cache_stats['cache_size']}/{cache_stats['max_size']}")
            print(f"Cache Hit Rate: {cache_stats['hit_rate']:.1%}")
    
    def save_cache(self) -> None:
        """Persist cache to disk."""
        if self.cache:
            self.cache.save()
    
    def clear_cache(self) -> None:
        """Clear all cached entries."""
        if self.cache:
            self.cache.clear()


# Convenience function
def create_swift_llm(
    enable_cache: bool = True,
    cache_threshold: float = 0.85,
    groq_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> SwiftLLM:
    """
    Create a SWIFT-LLM instance with common settings.
    
    Args:
        enable_cache: Whether to enable semantic caching
        cache_threshold: Similarity threshold for cache hits
        groq_api_key: Groq API key (or use GROQ_API_KEY env var)
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        
    Returns:
        Configured SwiftLLM instance
    """
    import os
    
    # Set API keys if provided
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    config = Config.from_env()
    config.cache.similarity_threshold = cache_threshold
    config.cache.enabled = enable_cache
    
    return SwiftLLM(config=config)


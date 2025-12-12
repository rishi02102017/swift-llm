"""
Model Registry for SWIFT-LLM.

Manages available models across different tiers and backends.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ModelBackend(Enum):
    """Supported model backends."""
    CACHE = "cache"
    GROQ = "groq"
    OPENAI = "openai"
    LOCAL_MLX = "mlx"
    LOCAL_LLAMACPP = "llamacpp"
    ANTHROPIC = "anthropic"


@dataclass
class ModelInfo:
    """Information about a model."""
    
    name: str
    backend: ModelBackend
    tier: int
    
    # Performance characteristics
    avg_latency_ms: float  # Typical latency
    tokens_per_second: float  # Generation speed
    max_context_length: int  # Maximum context window
    
    # Cost (per 1K tokens, input + output average)
    cost_per_1k_tokens: float
    
    # Quality metrics (relative, 0-1)
    quality_score: float
    
    # Availability
    requires_api_key: bool = True
    requires_gpu: bool = False
    
    # Capabilities
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    
    # Additional metadata
    description: str = ""
    model_id: str = ""  # Actual model identifier for API calls
    
    def __post_init__(self):
        if not self.model_id:
            self.model_id = self.name


# Default model configurations
DEFAULT_MODELS = [
    ModelInfo(
        name="cache",
        backend=ModelBackend.CACHE,
        tier=1,
        avg_latency_ms=5,
        tokens_per_second=float('inf'),
        max_context_length=float('inf'),
        cost_per_1k_tokens=0.0,
        quality_score=0.9,  # High because it's cached from good responses
        requires_api_key=False,
        description="Cached responses from previous queries",
    ),
    ModelInfo(
        name="llama-3.1-8b-instant",
        backend=ModelBackend.GROQ,
        tier=2,
        avg_latency_ms=100,
        tokens_per_second=500,
        max_context_length=8192,
        cost_per_1k_tokens=0.0001,  # Very cheap
        quality_score=0.75,
        requires_api_key=True,
        supports_function_calling=True,
        description="Fast inference via Groq",
        model_id="llama-3.1-8b-instant",
    ),
    ModelInfo(
        name="llama-3.1-70b-versatile",
        backend=ModelBackend.GROQ,
        tier=3,
        avg_latency_ms=300,
        tokens_per_second=300,
        max_context_length=32768,
        cost_per_1k_tokens=0.0005,
        quality_score=0.85,
        requires_api_key=True,
        supports_function_calling=True,
        description="Higher quality via Groq",
        model_id="llama-3.1-70b-versatile",
    ),
    ModelInfo(
        name="gpt-4o-mini",
        backend=ModelBackend.OPENAI,
        tier=4,
        avg_latency_ms=800,
        tokens_per_second=100,
        max_context_length=128000,
        cost_per_1k_tokens=0.00015,
        quality_score=0.90,
        requires_api_key=True,
        supports_function_calling=True,
        supports_vision=True,
        description="Premium quality via OpenAI",
        model_id="gpt-4o-mini",
    ),
    ModelInfo(
        name="gpt-4o",
        backend=ModelBackend.OPENAI,
        tier=5,  # Extra tier for complex tasks
        avg_latency_ms=1500,
        tokens_per_second=80,
        max_context_length=128000,
        cost_per_1k_tokens=0.005,
        quality_score=0.98,
        requires_api_key=True,
        supports_function_calling=True,
        supports_vision=True,
        description="Best quality via OpenAI (premium)",
        model_id="gpt-4o",
    ),
]


class ModelRegistry:
    """
    Registry of available models for SWIFT-LLM.
    
    Manages model configurations and provides model selection utilities.
    """
    
    def __init__(self, models: Optional[List[ModelInfo]] = None):
        """
        Initialize the registry.
        
        Args:
            models: List of model configurations. Uses defaults if None.
        """
        self.models: Dict[str, ModelInfo] = {}
        self.tier_models: Dict[int, List[str]] = {}
        
        # Register models
        for model in (models or DEFAULT_MODELS):
            self.register(model)
    
    def register(self, model: ModelInfo) -> None:
        """Register a model in the registry."""
        self.models[model.name] = model
        
        if model.tier not in self.tier_models:
            self.tier_models[model.tier] = []
        if model.name not in self.tier_models[model.tier]:
            self.tier_models[model.tier].append(model.name)
    
    def get(self, name: str) -> Optional[ModelInfo]:
        """Get model by name."""
        return self.models.get(name)
    
    def get_for_tier(self, tier: int) -> Optional[ModelInfo]:
        """
        Get the primary model for a tier.
        
        Returns the first model registered for that tier.
        """
        models = self.tier_models.get(tier, [])
        if models:
            return self.models.get(models[0])
        return None
    
    def get_all_for_tier(self, tier: int) -> List[ModelInfo]:
        """Get all models for a tier."""
        models = self.tier_models.get(tier, [])
        return [self.models[name] for name in models if name in self.models]
    
    def get_next_tier_model(self, current_tier: int) -> Optional[ModelInfo]:
        """Get model from the next higher tier (for escalation)."""
        for tier in range(current_tier + 1, max(self.tier_models.keys()) + 1):
            model = self.get_for_tier(tier)
            if model:
                return model
        return None
    
    def get_fallback_model(self, preferred_tier: int) -> Optional[ModelInfo]:
        """Get a fallback model if preferred tier is unavailable."""
        # Try lower tiers first, then higher
        for tier in range(preferred_tier - 1, 0, -1):
            model = self.get_for_tier(tier)
            if model:
                return model
        for tier in range(preferred_tier + 1, max(self.tier_models.keys()) + 1):
            model = self.get_for_tier(tier)
            if model:
                return model
        return None
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self.models.values())
    
    def list_by_tier(self) -> Dict[int, List[ModelInfo]]:
        """List models grouped by tier."""
        result = {}
        for tier in sorted(self.tier_models.keys()):
            result[tier] = self.get_all_for_tier(tier)
        return result
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a query."""
        model = self.get(model_name)
        if not model:
            return 0.0
        
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * model.cost_per_1k_tokens
    
    def __repr__(self) -> str:
        return f"ModelRegistry(models={list(self.models.keys())})"


"""
Configuration module for SWIFT-LLM.

Handles all hyperparameters, API keys, and system settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars


@dataclass
class CacheConfig:
    """Configuration for semantic cache."""
    
    # Similarity threshold for cache hits (0.0 - 1.0)
    # Lower = more cache hits, Higher = more precise matches
    # 0.70 is optimized for semantic similarity with normalization
    similarity_threshold: float = 0.70
    
    # Maximum number of entries in cache
    max_entries: int = 10000
    
    # Time-to-live for cache entries (seconds)
    ttl_seconds: int = 3600  # 1 hour
    
    # Embedding model for query vectorization
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Cache storage path
    cache_dir: Path = field(default_factory=lambda: Path("./cache_data"))
    
    # Enable/disable cache
    enabled: bool = True


@dataclass
class RouterConfig:
    """Configuration for query complexity router."""
    
    # Complexity thresholds for tier assignment
    # Score ranges: [0-0.25] -> Tier 1, [0.25-0.5] -> Tier 2, etc.
    tier_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    
    # Classifier model (lightweight)
    classifier_model: str = "distilbert-base-uncased"
    
    # Feature extraction settings
    use_length_feature: bool = True
    use_entity_feature: bool = True
    use_keyword_feature: bool = True
    
    # Default tier if classification fails
    default_tier: int = 2


@dataclass 
class InferenceConfig:
    """Configuration for model inference."""
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )
    
    # Model assignments per tier
    tier_models: Dict[int, str] = field(default_factory=lambda: {
        1: "cache",  # Cache-only tier
        2: "groq/llama-3.1-8b-instant",  # Fast API
        3: "groq/llama-3.1-70b-versatile",  # Better quality API
        4: "openai/gpt-4o-mini",  # Premium tier
    })
    
    # Local model settings (MLX for Apple Silicon)
    local_model_path: Optional[str] = None
    use_local_model: bool = False
    
    # Generation parameters
    max_tokens: int = 512
    temperature: float = 0.7
    
    # Timeout settings (seconds)
    timeout: int = 30
    
    # Retry settings
    max_retries: int = 3


@dataclass
class ValidationConfig:
    """Configuration for response validation."""
    
    # Minimum confidence score to accept response
    confidence_threshold: float = 0.7
    
    # Enable hallucination detection
    enable_hallucination_check: bool = True
    
    # Enable automatic escalation on low confidence
    enable_auto_escalation: bool = True
    
    # Maximum escalation attempts
    max_escalations: int = 2


@dataclass
class Config:
    """Main configuration for SWIFT-LLM system."""
    
    cache: CacheConfig = field(default_factory=CacheConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # Metrics collection
    collect_metrics: bool = True
    metrics_file: Path = field(default_factory=lambda: Path("./metrics.jsonl"))
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("SWIFT_CACHE_THRESHOLD"):
            config.cache.similarity_threshold = float(os.getenv("SWIFT_CACHE_THRESHOLD"))
        
        if os.getenv("SWIFT_DEFAULT_TIER"):
            config.router.default_tier = int(os.getenv("SWIFT_DEFAULT_TIER"))
            
        if os.getenv("SWIFT_MAX_TOKENS"):
            config.inference.max_tokens = int(os.getenv("SWIFT_MAX_TOKENS"))
            
        if os.getenv("SWIFT_LOG_LEVEL"):
            config.log_level = os.getenv("SWIFT_LOG_LEVEL")
            
        return config
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Check cache settings
        if not 0 <= self.cache.similarity_threshold <= 1:
            errors.append("Cache similarity_threshold must be between 0 and 1")
            
        # Check router settings
        if not 1 <= self.router.default_tier <= 4:
            errors.append("Default tier must be between 1 and 4")
            
        # Check API keys for non-local inference
        if not self.inference.use_local_model:
            if not self.inference.groq_api_key and not self.inference.openai_api_key:
                errors.append("At least one API key (GROQ or OpenAI) is required")
        
        if errors:
            for error in errors:
                print(f"Config Error: {error}")
            return False
            
        return True


# Global default configuration
DEFAULT_CONFIG = Config()


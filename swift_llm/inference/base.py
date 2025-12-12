"""
Base classes for inference backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class InferenceResult:
    """Result from a model inference call."""
    
    # Response content
    text: str
    
    # Metadata
    model: str
    tier: int
    
    # Performance
    latency_ms: float
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_per_second: float = 0.0
    
    # Quality indicators
    finish_reason: str = "stop"  # stop, length, error
    confidence_score: float = 1.0
    
    # Cost
    estimated_cost: float = 0.0
    
    # Error handling
    success: bool = True
    error_message: Optional[str] = None
    
    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "tier": self.tier,
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_per_second": self.tokens_per_second,
            "finish_reason": self.finish_reason,
            "confidence_score": self.confidence_score,
            "estimated_cost": self.estimated_cost,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Message:
    """A chat message."""
    role: str  # system, user, assistant
    content: str


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> InferenceResult:
        """
        Generate a response from the model.
        
        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            InferenceResult with response and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (API key set, etc.)."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass
    
    @property
    @abstractmethod
    def tier(self) -> int:
        """Model tier."""
        pass


"""
Eviction policies for the semantic cache.

Provides multiple strategies for cache eviction:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- Adaptive (combines recency, frequency, and confidence)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math

from swift_llm.cache.cache_store import CacheEntry


class EvictionPolicy(ABC):
    """Abstract base class for eviction policies."""
    
    @abstractmethod
    def score(self, entry: CacheEntry) -> float:
        """
        Calculate eviction score for an entry.
        Lower scores = more likely to be evicted.
        """
        pass
    
    @abstractmethod
    def should_evict(self, entry: CacheEntry) -> bool:
        """Check if an entry should be evicted based on policy rules."""
        pass
    
    def select_for_eviction(
        self, 
        entries: Dict[int, CacheEntry], 
        count: int = 1
    ) -> List[int]:
        """Select entries for eviction based on scores."""
        scored = [(entry_id, self.score(entry)) for entry_id, entry in entries.items()]
        scored.sort(key=lambda x: x[1])  # Sort by score ascending
        return [entry_id for entry_id, _ in scored[:count]]


class LRUEviction(EvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
    
    def score(self, entry: CacheEntry) -> float:
        """Score based on recency. Older = lower score."""
        age = (datetime.now() - entry.last_accessed).total_seconds()
        return -age  # Negative so older entries have lower scores
    
    def should_evict(self, entry: CacheEntry) -> bool:
        """Evict if older than TTL."""
        age = (datetime.now() - entry.last_accessed).total_seconds()
        return age > self.ttl_seconds


class LFUEviction(EvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def __init__(self, min_access_count: int = 1, ttl_seconds: int = 3600):
        self.min_access_count = min_access_count
        self.ttl_seconds = ttl_seconds
    
    def score(self, entry: CacheEntry) -> float:
        """Score based on frequency. Less frequent = lower score."""
        return entry.access_count
    
    def should_evict(self, entry: CacheEntry) -> bool:
        """Evict if below minimum access count and old."""
        age = (datetime.now() - entry.last_accessed).total_seconds()
        return (
            entry.access_count < self.min_access_count 
            and age > self.ttl_seconds
        )


@dataclass
class AdaptiveEvictionConfig:
    """Configuration for adaptive eviction."""
    
    # Weight factors for scoring
    recency_weight: float = 0.4
    frequency_weight: float = 0.3
    confidence_weight: float = 0.3
    
    # Time constants
    ttl_seconds: int = 3600
    decay_rate: float = 0.1  # Exponential decay rate
    
    # Thresholds
    min_confidence: float = 0.5
    min_access_for_retention: int = 2


class AdaptiveEviction(EvictionPolicy):
    """
    Adaptive eviction policy combining multiple factors:
    - Recency: How recently was the entry accessed
    - Frequency: How often is the entry accessed
    - Confidence: How confident was the original response
    
    This is the NOVEL component that makes SWIFT-LLM unique.
    """
    
    def __init__(self, config: Optional[AdaptiveEvictionConfig] = None):
        self.config = config or AdaptiveEvictionConfig()
    
    def score(self, entry: CacheEntry) -> float:
        """
        Calculate composite eviction score.
        
        Score = w_r * R(t) + w_f * F(n) + w_c * C
        
        Where:
        - R(t) = recency score with exponential decay
        - F(n) = frequency score (log-scaled access count)
        - C = confidence score from original generation
        """
        # Recency score: exponential decay based on time since last access
        age_seconds = (datetime.now() - entry.last_accessed).total_seconds()
        recency_score = math.exp(-self.config.decay_rate * age_seconds / 3600)
        
        # Frequency score: log-scaled access count
        frequency_score = math.log(1 + entry.access_count) / math.log(10)
        frequency_score = min(frequency_score, 1.0)  # Cap at 1.0
        
        # Confidence score: direct from entry
        confidence_score = entry.confidence_score
        
        # Weighted combination
        total_score = (
            self.config.recency_weight * recency_score +
            self.config.frequency_weight * frequency_score +
            self.config.confidence_weight * confidence_score
        )
        
        return total_score
    
    def should_evict(self, entry: CacheEntry) -> bool:
        """
        Determine if entry should be evicted based on multiple criteria.
        
        Eviction conditions:
        1. TTL expired AND low access count
        2. Very low confidence score
        3. Score below threshold
        """
        age = (datetime.now() - entry.last_accessed).total_seconds()
        
        # Condition 1: Old and rarely accessed
        if (age > self.config.ttl_seconds and 
            entry.access_count < self.config.min_access_for_retention):
            return True
        
        # Condition 2: Low confidence responses
        if entry.confidence_score < self.config.min_confidence:
            return True
        
        # Condition 3: Low overall score
        score = self.score(entry)
        if score < 0.2:  # Very low score threshold
            return True
        
        return False
    
    def get_eviction_candidates(
        self,
        entries: Dict[int, CacheEntry],
        target_size: int
    ) -> List[int]:
        """
        Get list of entry IDs to evict to reach target size.
        
        Uses a two-phase approach:
        1. First evict entries that should_evict() returns True
        2. Then evict lowest-scored entries until target size
        """
        to_evict = []
        
        # Phase 1: Evict based on policy rules
        for entry_id, entry in entries.items():
            if self.should_evict(entry):
                to_evict.append(entry_id)
        
        # Phase 2: If still over target, evict lowest-scored
        remaining = {k: v for k, v in entries.items() if k not in to_evict}
        current_size = len(remaining)
        
        if current_size > target_size:
            need_to_evict = current_size - target_size
            additional = self.select_for_eviction(remaining, need_to_evict)
            to_evict.extend(additional)
        
        return to_evict


class TTLEviction(EvictionPolicy):
    """Simple time-to-live based eviction."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
    
    def score(self, entry: CacheEntry) -> float:
        """Score based on remaining TTL."""
        age = (datetime.now() - entry.created_at).total_seconds()
        remaining = self.ttl_seconds - age
        return remaining  # Lower remaining TTL = lower score
    
    def should_evict(self, entry: CacheEntry) -> bool:
        """Evict if TTL expired."""
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > self.ttl_seconds


# Factory function for creating eviction policies
def create_eviction_policy(
    policy_type: str = "adaptive",
    **kwargs
) -> EvictionPolicy:
    """
    Create an eviction policy by type.
    
    Args:
        policy_type: One of "lru", "lfu", "adaptive", "ttl"
        **kwargs: Policy-specific configuration
    
    Returns:
        EvictionPolicy instance
    """
    policies = {
        "lru": LRUEviction,
        "lfu": LFUEviction,
        "adaptive": AdaptiveEviction,
        "ttl": TTLEviction,
    }
    
    if policy_type not in policies:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    return policies[policy_type](**kwargs)


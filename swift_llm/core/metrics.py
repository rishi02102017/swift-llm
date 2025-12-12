"""
Metrics collection and logging for SWIFT-LLM.

Tracks latency, cache hits, model usage, and quality metrics.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    
    query_id: str
    timestamp: str
    query_text: str
    query_length: int
    
    # Routing info
    predicted_tier: int
    actual_tier: int
    complexity_score: float
    
    # Cache info
    cache_hit: bool
    cache_similarity: Optional[float] = None
    
    # Latency breakdown (milliseconds)
    total_latency_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    routing_ms: float = 0.0
    inference_ms: float = 0.0
    validation_ms: float = 0.0
    
    # Response info
    response_length: int = 0
    confidence_score: float = 0.0
    was_escalated: bool = False
    escalation_count: int = 0
    
    # Model info
    model_used: str = ""
    tokens_used: int = 0
    estimated_cost: float = 0.0


@dataclass
class AggregateMetrics:
    """Aggregated metrics over multiple queries."""
    
    total_queries: int = 0
    cache_hits: int = 0
    cache_hit_rate: float = 0.0
    
    # Latency stats
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Tier distribution
    tier_distribution: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Cost stats
    total_cost: float = 0.0
    avg_cost_per_query: float = 0.0
    
    # Quality stats
    avg_confidence: float = 0.0
    escalation_rate: float = 0.0


class MetricsLogger:
    """
    Thread-safe metrics logger for SWIFT-LLM.
    
    Collects per-query metrics and provides aggregation utilities.
    """
    
    def __init__(self, log_file: Optional[Path] = None, enabled: bool = True):
        self.log_file = log_file or Path("./metrics.jsonl")
        self.enabled = enabled
        self.metrics: List[QueryMetrics] = []
        self._lock = threading.Lock()
        self._query_counter = 0
        
        # Ensure log directory exists
        if self.enabled:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def generate_query_id(self) -> str:
        """Generate unique query ID."""
        with self._lock:
            self._query_counter += 1
            return f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._query_counter:06d}"
    
    def log_query(self, metrics: QueryMetrics) -> None:
        """Log metrics for a single query."""
        if not self.enabled:
            return
            
        with self._lock:
            self.metrics.append(metrics)
            
            # Append to file
            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(metrics)) + "\n")
    
    def get_aggregate_metrics(self) -> AggregateMetrics:
        """Calculate aggregate metrics from collected data."""
        if not self.metrics:
            return AggregateMetrics()
        
        with self._lock:
            total = len(self.metrics)
            cache_hits = sum(1 for m in self.metrics if m.cache_hit)
            
            latencies = sorted([m.total_latency_ms for m in self.metrics])
            costs = [m.estimated_cost for m in self.metrics]
            confidences = [m.confidence_score for m in self.metrics if m.confidence_score > 0]
            escalations = sum(1 for m in self.metrics if m.was_escalated)
            
            tier_dist = defaultdict(int)
            for m in self.metrics:
                tier_dist[m.actual_tier] += 1
            
            return AggregateMetrics(
                total_queries=total,
                cache_hits=cache_hits,
                cache_hit_rate=cache_hits / total if total > 0 else 0,
                avg_latency_ms=sum(latencies) / total if total > 0 else 0,
                p50_latency_ms=latencies[int(total * 0.5)] if total > 0 else 0,
                p95_latency_ms=latencies[int(total * 0.95)] if total > 0 else 0,
                p99_latency_ms=latencies[int(total * 0.99)] if total > 0 else 0,
                tier_distribution=dict(tier_dist),
                total_cost=sum(costs),
                avg_cost_per_query=sum(costs) / total if total > 0 else 0,
                avg_confidence=sum(confidences) / len(confidences) if confidences else 0,
                escalation_rate=escalations / total if total > 0 else 0,
            )
    
    def print_summary(self) -> None:
        """Print a summary of collected metrics."""
        agg = self.get_aggregate_metrics()
        
        print("\n" + "=" * 60)
        print("SWIFT-LLM Performance Summary")
        print("=" * 60)
        print(f"Total Queries:      {agg.total_queries}")
        print(f"Cache Hit Rate:     {agg.cache_hit_rate:.1%}")
        print(f"Avg Latency:        {agg.avg_latency_ms:.1f}ms")
        print(f"P95 Latency:        {agg.p95_latency_ms:.1f}ms")
        print(f"P99 Latency:        {agg.p99_latency_ms:.1f}ms")
        print(f"Avg Confidence:     {agg.avg_confidence:.2f}")
        print(f"Escalation Rate:    {agg.escalation_rate:.1%}")
        print(f"Total Cost:         ${agg.total_cost:.4f}")
        print(f"Avg Cost/Query:     ${agg.avg_cost_per_query:.6f}")
        print("\nTier Distribution:")
        for tier, count in sorted(agg.tier_distribution.items()):
            pct = count / agg.total_queries * 100 if agg.total_queries > 0 else 0
            print(f"  Tier {tier}: {count} ({pct:.1f}%)")
        print("=" * 60 + "\n")
    
    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self.metrics.clear()
            self._query_counter = 0
    
    def export_to_json(self, filepath: Path) -> None:
        """Export all metrics to a JSON file."""
        with self._lock:
            with open(filepath, "w") as f:
                json.dump([asdict(m) for m in self.metrics], f, indent=2)


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


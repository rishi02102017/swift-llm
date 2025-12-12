"""
Advanced Semantic Cache for SWIFT-LLM.

Key Features:
- Embedding-based similarity matching with FAISS
- Query preprocessing and normalization for better hit rates
- Persistent storage across sessions (FAISS index + SQLite)
- Hybrid lexical + semantic matching
- Adaptive TTL and smart eviction
- Cache warming and preloading

This is the NOVEL component that differentiates from traditional caching.
"""

import re
import json
import time
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer

from swift_llm.cache.cache_store import CacheStore, CacheEntry
from swift_llm.cache.eviction import AdaptiveEviction, EvictionPolicy
from swift_llm.core.config import CacheConfig


@dataclass
class CacheResult:
    """Result from a cache lookup."""
    
    hit: bool
    response: Optional[str] = None
    similarity: float = 0.0
    entry_id: Optional[int] = None
    lookup_time_ms: float = 0.0
    match_type: str = "none"  # none, exact, lexical, semantic
    
    # Metadata from cached entry
    original_model_tier: Optional[int] = None
    confidence_score: float = 0.0
    access_count: int = 0


class QueryPreprocessor:
    """
    Preprocesses queries for better cache hit rates.
    
    Techniques:
    - Lowercasing and normalization
    - Stopword removal for semantic matching
    - Punctuation normalization
    - Whitespace normalization
    """
    
    # Common stopwords that don't affect meaning
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "can", "could", "would", "should", "will", "do", "does", "did",
        "please", "kindly", "just", "simply", "basically", "actually",
        "really", "very", "quite", "rather", "somewhat",
        "me", "my", "i", "you", "your", "we", "our", "they", "their",
        "this", "that", "these", "those", "it", "its",
    }
    
    # Question word normalizations
    QUESTION_NORMALIZATIONS = {
        "what's": "what is",
        "who's": "who is", 
        "where's": "where is",
        "when's": "when is",
        "why's": "why is",
        "how's": "how is",
        "what're": "what are",
        "who're": "who are",
        "there's": "there is",
        "here's": "here is",
        "let's": "let us",
        "it's": "it is",
        "that's": "that is",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "won't": "will not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "can't": "cannot",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
    }
    
    @classmethod
    def normalize(cls, query: str) -> str:
        """
        Normalize query for better matching.
        
        Returns normalized query while preserving semantic meaning.
        """
        # Lowercase
        text = query.lower().strip()
        
        # Expand contractions
        for contraction, expansion in cls.QUESTION_NORMALIZATIONS.items():
            text = text.replace(contraction, expansion)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation but keep sentence structure
        text = re.sub(r'[!?]{2,}', '?', text)
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    @classmethod
    def extract_key_terms(cls, query: str) -> str:
        """
        Extract key terms for lexical matching.
        
        Removes stopwords and keeps only meaningful terms.
        """
        text = cls.normalize(query)
        
        # Remove punctuation for term extraction
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split and filter
        words = text.split()
        key_terms = [w for w in words if w not in cls.STOPWORDS and len(w) > 2]
        
        return ' '.join(key_terms)
    
    @classmethod
    def compute_lexical_hash(cls, query: str) -> str:
        """
        Compute a lexical hash for exact/near-exact matching.
        
        This enables O(1) lookup for identical queries.
        """
        key_terms = cls.extract_key_terms(query)
        # Sort terms for order-independent matching
        sorted_terms = ' '.join(sorted(key_terms.split()))
        return sorted_terms


class SemanticCache:
    """
    Advanced semantic similarity-based cache for LLM responses.
    
    Key Features:
    - Hybrid matching: exact -> lexical -> semantic
    - FAISS for fast approximate nearest neighbor search
    - Persistent storage across sessions
    - Query preprocessing for better hit rates
    - Adaptive eviction based on recency, frequency, and confidence
    
    Example:
        cache = SemanticCache()
        
        # Check cache
        result = cache.lookup("What is the capital of France?")
        if result.hit:
            return result.response
        
        # Generate response...
        response = llm.generate(query)
        
        # Store in cache
        cache.store(query, response, confidence=0.95)
        
        # Save to disk for persistence
        cache.save()
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        eviction_policy: Optional[EvictionPolicy] = None
    ):
        self.config = config or CacheConfig()
        self.eviction_policy = eviction_policy or AdaptiveEviction()
        
        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for persistent storage
        self.faiss_index_path = self.config.cache_dir / "faiss_index.bin"
        self.lexical_index_path = self.config.cache_dir / "lexical_index.pkl"
        self.stats_path = self.config.cache_dir / "cache_stats.json"
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.config.embedding_model}")
        self.encoder = SentenceTransformer(self.config.embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize storage
        self._store = CacheStore(
            cache_dir=self.config.cache_dir,
            max_entries=self.config.max_entries
        )
        
        # Lexical index for fast exact/near-exact matching
        self._lexical_index: Dict[str, int] = {}
        
        # Load lexical index if exists
        self._load_lexical_index()
        
        # Initialize FAISS index
        self._init_faiss_index()
        
        # Load or initialize statistics
        self.stats = self._load_stats()
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load statistics from disk or initialize."""
        if self.stats_path.exists():
            try:
                with open(self.stats_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "exact_hits": 0,
            "lexical_hits": 0,
            "semantic_hits": 0,
            "total_stores": 0,
            "evictions": 0,
        }
    
    def _save_stats(self) -> None:
        """Save statistics to disk."""
        with open(self.stats_path, 'w') as f:
            json.dump(self.stats, f)
    
    def _load_lexical_index(self) -> None:
        """Load lexical index from disk."""
        if self.lexical_index_path.exists():
            try:
                with open(self.lexical_index_path, 'rb') as f:
                    self._lexical_index = pickle.load(f)
                print(f"Loaded {len(self._lexical_index)} lexical entries")
            except Exception as e:
                print(f"Warning: Could not load lexical index: {e}")
                self._lexical_index = {}
    
    def _save_lexical_index(self) -> None:
        """Save lexical index to disk."""
        with open(self.lexical_index_path, 'wb') as f:
            pickle.dump(self._lexical_index, f)
    
    def _init_faiss_index(self) -> None:
        """Initialize or load FAISS index."""
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available, using brute-force search")
            self.index = None
            return
        
        # Try to load existing index
        if self.faiss_index_path.exists():
            try:
                self.index = faiss.read_index(str(self.faiss_index_path))
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
                return
            except Exception as e:
                print(f"Warning: Could not load FAISS index: {e}")
        
        # Create new index - using Inner Product for cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Load existing embeddings into index
        embeddings = self._store.get_all_embeddings()
        if embeddings is not None and len(embeddings) > 0:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            print(f"Built FAISS index with {len(embeddings)} vectors")
    
    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk."""
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(self.faiss_index_path))
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _lookup_exact(self, query: str) -> Optional[CacheResult]:
        """
        Try exact lexical match first (O(1) lookup).
        """
        lexical_hash = QueryPreprocessor.compute_lexical_hash(query)
        
        if lexical_hash in self._lexical_index:
            entry_id = self._lexical_index[lexical_hash]
            entry = self._store.get(entry_id)
            
            if entry is not None:
                return CacheResult(
                    hit=True,
                    response=entry.response,
                    similarity=1.0,
                    entry_id=entry_id,
                    lookup_time_ms=0.1,
                    match_type="lexical",
                    original_model_tier=entry.model_tier,
                    confidence_score=entry.confidence_score,
                    access_count=entry.access_count,
                )
        
        return None
    
    def _lookup_semantic(self, query: str, query_embedding: np.ndarray) -> Optional[CacheResult]:
        """
        Semantic similarity search using FAISS.
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return self._lookup_semantic_bruteforce(query_embedding)
        
        query_normalized = self._normalize(query_embedding).reshape(1, -1)
        
        # Search for top-k similar entries
        k = min(5, self.index.ntotal)
        similarities, indices = self.index.search(query_normalized, k)
        
        for i in range(len(indices[0])):
            idx = indices[0][i]
            sim = float(similarities[0][i])
            
            if idx != -1 and sim >= self.config.similarity_threshold:
                entry = self._store.get_entry_by_index(idx)
                
                if entry is not None:
                    return CacheResult(
                        hit=True,
                        response=entry.response,
                        similarity=sim,
                        entry_id=idx,
                        match_type="semantic",
                        original_model_tier=entry.model_tier,
                        confidence_score=entry.confidence_score,
                        access_count=entry.access_count,
                    )
        
        return None
    
    def _lookup_semantic_bruteforce(self, query_embedding: np.ndarray) -> Optional[CacheResult]:
        """
        Brute-force semantic search fallback.
        """
        embeddings = self._store.get_all_embeddings()
        if embeddings is None or len(embeddings) == 0:
            return None
        
        query_normalized = self._normalize(query_embedding)
        
        # Normalize stored embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = embeddings / norms
        
        # Compute all similarities
        similarities = np.dot(normalized_embeddings, query_normalized)
        
        # Find best match above threshold
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim >= self.config.similarity_threshold:
            entry = self._store.get_entry_by_index(best_idx)
            
            if entry is not None:
                return CacheResult(
                    hit=True,
                    response=entry.response,
                    similarity=float(best_sim),
                    entry_id=best_idx,
                    match_type="semantic",
                    original_model_tier=entry.model_tier,
                    confidence_score=entry.confidence_score,
                    access_count=entry.access_count,
                )
        
        return None
    
    def lookup(self, query: str) -> CacheResult:
        """
        Look up a query in the semantic cache.
        
        Uses a tiered approach:
        1. Exact lexical match (fastest, O(1))
        2. Semantic similarity search (FAISS)
        
        Args:
            query: The user query to look up
            
        Returns:
            CacheResult with hit status and response if found
        """
        if not self.config.enabled:
            return CacheResult(hit=False)
        
        start_time = time.perf_counter()
        self.stats["total_lookups"] += 1
        
        # Preprocess query
        normalized_query = QueryPreprocessor.normalize(query)
        
        # Step 1: Try exact/lexical match
        result = self._lookup_exact(normalized_query)
        if result is not None:
            result.lookup_time_ms = (time.perf_counter() - start_time) * 1000
            self.stats["cache_hits"] += 1
            self.stats["lexical_hits"] += 1
            return result
        
        # Step 2: Encode for semantic search
        query_embedding = self._encode(normalized_query)
        
        # Step 3: Semantic similarity search
        result = self._lookup_semantic(normalized_query, query_embedding)
        if result is not None:
            result.lookup_time_ms = (time.perf_counter() - start_time) * 1000
            self.stats["cache_hits"] += 1
            self.stats["semantic_hits"] += 1
            return result
        
        # Cache miss
        lookup_time = (time.perf_counter() - start_time) * 1000
        self.stats["cache_misses"] += 1
        
        return CacheResult(
            hit=False,
            lookup_time_ms=lookup_time,
            match_type="none",
        )
    
    def store(
        self,
        query: str,
        response: str,
        model_tier: int = 2,
        confidence_score: float = 1.0,
    ) -> int:
        """
        Store a query-response pair in the cache.
        
        Args:
            query: The user query
            response: The generated response
            model_tier: Which model tier generated this response
            confidence_score: Confidence score of the response
            
        Returns:
            Entry ID of the stored cache entry
        """
        if not self.config.enabled:
            return -1
        
        self.stats["total_stores"] += 1
        
        # Preprocess query
        normalized_query = QueryPreprocessor.normalize(query)
        
        # Encode query
        query_embedding = self._encode(normalized_query)
        
        # Create cache entry
        entry = CacheEntry(
            query=normalized_query,
            response=response,
            embedding=query_embedding,
            model_tier=model_tier,
            confidence_score=confidence_score,
        )
        
        # Store in backend
        entry_id = self._store.add(entry)
        
        # Add to lexical index
        lexical_hash = QueryPreprocessor.compute_lexical_hash(normalized_query)
        self._lexical_index[lexical_hash] = entry_id
        
        # Add to FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            normalized = self._normalize(query_embedding).reshape(1, -1)
            self.index.add(normalized)
        
        return entry_id
    
    def invalidate(self, query: str) -> bool:
        """Invalidate a cache entry by query."""
        result = self.lookup(query)
        if result.hit and result.entry_id is not None:
            return self._store.remove(result.entry_id)
        return False
    
    def evict_stale(self) -> int:
        """Evict stale entries based on eviction policy."""
        entries = {
            entry_id: self._store.get(entry_id)
            for entry_id in self._store.get_entry_ids()
        }
        entries = {k: v for k, v in entries.items() if v is not None}
        
        to_evict = self.eviction_policy.get_eviction_candidates(
            entries,
            target_size=int(self.config.max_entries * 0.9)
        )
        
        for entry_id in to_evict:
            self._store.remove(entry_id)
            self.stats["evictions"] += 1
        
        # Rebuild indexes after eviction
        if to_evict:
            self._rebuild_indexes()
        
        return len(to_evict)
    
    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes from storage."""
        # Rebuild lexical index
        self._lexical_index.clear()
        for entry_id in self._store.get_entry_ids():
            entry = self._store.get(entry_id)
            if entry:
                lexical_hash = QueryPreprocessor.compute_lexical_hash(entry.query)
                self._lexical_index[lexical_hash] = entry_id
        
        # Rebuild FAISS index
        self._init_faiss_index()
    
    def warm_cache(self, queries_responses: List[Tuple[str, str, int]]) -> int:
        """
        Pre-warm the cache with known query-response pairs.
        
        Args:
            queries_responses: List of (query, response, tier) tuples
            
        Returns:
            Number of entries added
        """
        count = 0
        for query, response, tier in queries_responses:
            self.store(query, response, model_tier=tier, confidence_score=0.95)
            count += 1
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self.stats["cache_hits"] / self.stats["total_lookups"]
            if self.stats["total_lookups"] > 0 else 0
        )
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self._store),
            "max_size": self.config.max_entries,
            "similarity_threshold": self.config.similarity_threshold,
            "lexical_index_size": len(self._lexical_index),
            "faiss_index_size": self.index.ntotal if self.index else 0,
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._store.clear()
        self._lexical_index.clear()
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Reset stats
        self.stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "exact_hits": 0,
            "lexical_hits": 0,
            "semantic_hits": 0,
            "total_stores": 0,
            "evictions": 0,
        }
        
        # Remove persisted files
        for path in [self.faiss_index_path, self.lexical_index_path, self.stats_path]:
            if path.exists():
                path.unlink()
    
    def save(self) -> None:
        """Persist cache to disk (FAISS index, lexical index, stats)."""
        self._store.save()
        self._save_faiss_index()
        self._save_lexical_index()
        self._save_stats()
    
    def __len__(self) -> int:
        return len(self._store)
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SemanticCache(size={stats['cache_size']}, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"threshold={stats['similarity_threshold']})"
        )


def create_semantic_cache(
    cache_dir: str = "./cache_data",
    similarity_threshold: float = 0.75,
    max_entries: int = 10000,
) -> SemanticCache:
    """
    Create a semantic cache with common defaults.
    
    Args:
        cache_dir: Directory to store cache data
        similarity_threshold: Minimum similarity for cache hit (0-1)
        max_entries: Maximum number of cached entries
        
    Returns:
        Configured SemanticCache instance
    """
    config = CacheConfig(
        cache_dir=Path(cache_dir),
        similarity_threshold=similarity_threshold,
        max_entries=max_entries,
    )
    return SemanticCache(config=config)

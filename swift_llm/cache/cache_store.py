"""
Cache storage backend for SWIFT-LLM.

Provides persistent storage for cached query-response pairs.
"""

import json
import pickle
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    
    query: str
    response: str
    embedding: np.ndarray
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    confidence_score: float = 1.0
    model_tier: int = 2
    
    # For similarity matching
    similarity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "response": self.response,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "confidence_score": self.confidence_score,
            "model_tier": self.model_tier,
        }
    
    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheStore:
    """
    Persistent cache storage using SQLite + numpy arrays.
    
    Stores query-response pairs with embeddings for similarity search.
    Thread-safe for concurrent access.
    """
    
    def __init__(self, cache_dir: Path, max_entries: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self._lock = threading.RLock()
        
        # Ensure directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.db_path = self.cache_dir / "cache.db"
        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.index_path = self.cache_dir / "index.pkl"
        
        # In-memory cache
        self._entries: Dict[int, CacheEntry] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Dict[int, int] = {}
        self._next_id = 0
        
        # Initialize database
        self._init_db()
        self._load_cache()
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    confidence_score REAL DEFAULT 1.0,
                    model_tier INTEGER DEFAULT 2
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cache_entries(last_accessed)
            """)
            conn.commit()
    
    def _load_cache(self) -> None:
        """Load cache from disk into memory."""
        with self._lock:
            # Load entries from database
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM cache_entries ORDER BY id")
                rows = cursor.fetchall()
            
            # Load embeddings if they exist
            if self.embeddings_path.exists():
                self._embeddings = np.load(self.embeddings_path)
            else:
                self._embeddings = None
            
            # Load index mapping
            if self.index_path.exists():
                with open(self.index_path, "rb") as f:
                    self._id_to_idx = pickle.load(f)
            
            # Reconstruct entries
            for row in rows:
                entry_id = row["id"]
                idx = self._id_to_idx.get(entry_id)
                
                if idx is not None and self._embeddings is not None:
                    embedding = self._embeddings[idx]
                else:
                    embedding = np.zeros(384)  # Default embedding size
                
                entry = CacheEntry(
                    query=row["query"],
                    response=row["response"],
                    embedding=embedding,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_accessed=datetime.fromisoformat(row["last_accessed"]),
                    access_count=row["access_count"],
                    confidence_score=row["confidence_score"],
                    model_tier=row["model_tier"],
                )
                self._entries[entry_id] = entry
                self._next_id = max(self._next_id, entry_id + 1)
    
    def add(self, entry: CacheEntry) -> int:
        """Add a new entry to the cache. Returns entry ID."""
        with self._lock:
            # Check if we need to evict
            if len(self._entries) >= self.max_entries:
                self._evict_oldest()
            
            # Add to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO cache_entries 
                    (query, response, created_at, last_accessed, 
                     access_count, confidence_score, model_tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.query,
                    entry.response,
                    entry.created_at.isoformat(),
                    entry.last_accessed.isoformat(),
                    entry.access_count,
                    entry.confidence_score,
                    entry.model_tier,
                ))
                entry_id = cursor.lastrowid
                conn.commit()
            
            # Add to in-memory cache
            self._entries[entry_id] = entry
            
            # Update embeddings
            idx = len(self._id_to_idx)
            self._id_to_idx[entry_id] = idx
            
            if self._embeddings is None:
                self._embeddings = entry.embedding.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, entry.embedding])
            
            # Save embeddings periodically
            if len(self._entries) % 100 == 0:
                self._save_embeddings()
            
            return entry_id
    
    def get(self, entry_id: int) -> Optional[CacheEntry]:
        """Get an entry by ID."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry:
                entry.update_access()
                self._update_access_in_db(entry_id)
            return entry
    
    def get_all_embeddings(self) -> Optional[np.ndarray]:
        """Get all embeddings as a numpy array for similarity search."""
        with self._lock:
            return self._embeddings.copy() if self._embeddings is not None else None
    
    def get_entry_ids(self) -> List[int]:
        """Get all entry IDs in order."""
        with self._lock:
            return list(self._entries.keys())
    
    def get_entry_by_index(self, idx: int) -> Optional[CacheEntry]:
        """Get entry by its index in the embeddings array."""
        with self._lock:
            for entry_id, entry_idx in self._id_to_idx.items():
                if entry_idx == idx:
                    return self._entries.get(entry_id)
            return None
    
    def _update_access_in_db(self, entry_id: int) -> None:
        """Update access metadata in database."""
        entry = self._entries.get(entry_id)
        if entry:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE cache_entries 
                    SET last_accessed = ?, access_count = ?
                    WHERE id = ?
                """, (
                    entry.last_accessed.isoformat(),
                    entry.access_count,
                    entry_id,
                ))
                conn.commit()
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used entry."""
        if not self._entries:
            return
        
        # Find LRU entry
        oldest_id = min(
            self._entries.keys(),
            key=lambda k: self._entries[k].last_accessed
        )
        self.remove(oldest_id)
    
    def remove(self, entry_id: int) -> bool:
        """Remove an entry from the cache."""
        with self._lock:
            if entry_id not in self._entries:
                return False
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries WHERE id = ?", (entry_id,))
                conn.commit()
            
            # Remove from in-memory cache
            del self._entries[entry_id]
            
            # Note: We don't remove from embeddings array to avoid reindexing
            # The index mapping handles this
            if entry_id in self._id_to_idx:
                del self._id_to_idx[entry_id]
            
            return True
    
    def _save_embeddings(self) -> None:
        """Save embeddings to disk."""
        if self._embeddings is not None:
            np.save(self.embeddings_path, self._embeddings)
        with open(self.index_path, "wb") as f:
            pickle.dump(self._id_to_idx, f)
    
    def save(self) -> None:
        """Save all cache data to disk."""
        with self._lock:
            self._save_embeddings()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
            
            self._entries.clear()
            self._embeddings = None
            self._id_to_idx.clear()
            self._next_id = 0
            
            # Remove files
            if self.embeddings_path.exists():
                self.embeddings_path.unlink()
            if self.index_path.exists():
                self.index_path.unlink()
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __contains__(self, entry_id: int) -> bool:
        return entry_id in self._entries


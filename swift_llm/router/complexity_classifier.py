"""
Query Complexity Classifier for SWIFT-LLM.

Classifies queries by complexity to route to appropriate model tiers.
Uses a combination of heuristic features and optional ML-based classification.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class QueryFeatures:
    """Extracted features from a query for complexity classification."""
    
    # Basic features
    length: int  # Character count
    word_count: int
    sentence_count: int
    avg_word_length: float
    
    # Complexity indicators
    question_words: int  # who, what, where, when, why, how
    technical_terms: int
    named_entities: int  # Approximate count of proper nouns
    
    # Structural features
    has_multiple_questions: bool
    has_comparison: bool  # "compare", "difference", "vs"
    has_reasoning_request: bool  # "explain", "why", "analyze"
    has_code_request: bool  # "code", "program", "function"
    has_math_request: bool  # numbers, equations
    
    # Domain indicators
    domain: str  # general, technical, creative, factual
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to feature dictionary for ML models."""
        return {
            "length": self.length,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_word_length": self.avg_word_length,
            "question_words": self.question_words,
            "technical_terms": self.technical_terms,
            "named_entities": self.named_entities,
            "has_multiple_questions": float(self.has_multiple_questions),
            "has_comparison": float(self.has_comparison),
            "has_reasoning_request": float(self.has_reasoning_request),
            "has_code_request": float(self.has_code_request),
            "has_math_request": float(self.has_math_request),
        }


class ComplexityClassifier:
    """
    Classifies query complexity for intelligent routing.
    
    Uses a combination of rule-based heuristics and feature extraction
    to estimate query complexity on a 0-1 scale.
    
    Complexity Tiers:
    - Tier 1 (0.0-0.25): Simple factual queries, greetings
    - Tier 2 (0.25-0.5): Standard Q&A, basic explanations
    - Tier 3 (0.5-0.75): Complex reasoning, technical queries
    - Tier 4 (0.75-1.0): Multi-step reasoning, creative tasks
    """
    
    # Keywords for feature extraction
    QUESTION_WORDS = {"who", "what", "where", "when", "why", "how", "which"}
    
    TECHNICAL_TERMS = {
        "algorithm", "api", "database", "function", "variable", "class",
        "neural", "network", "machine learning", "deep learning", "model",
        "optimization", "architecture", "framework", "library", "protocol",
        "encryption", "authentication", "kubernetes", "docker", "cloud",
        "tensor", "gradient", "backpropagation", "transformer", "attention",
    }
    
    COMPARISON_WORDS = {
        "compare", "comparison", "difference", "vs", "versus", "between",
        "better", "worse", "advantage", "disadvantage", "pros", "cons",
    }
    
    REASONING_WORDS = {
        "explain", "why", "analyze", "analyse", "reason", "because",
        "evaluate", "assess", "consider", "think", "opinion", "argue",
        "justify", "elaborate", "describe in detail",
    }
    
    CODE_WORDS = {
        "code", "program", "function", "script", "implement", "write",
        "python", "javascript", "java", "c++", "sql", "html", "css",
        "debug", "fix", "error", "bug", "compile", "execute",
    }
    
    SIMPLE_PATTERNS = [
        r"^(hi|hello|hey|thanks|thank you|bye|goodbye)",
        r"^what is the (capital|population|currency) of",
        r"^who is the (president|ceo|founder) of",
        r"^what time is it",
        r"^what day is",
        r"^(yes|no|ok|okay)$",
    ]
    
    def __init__(self, tier_thresholds: Optional[List[float]] = None):
        """
        Initialize the classifier.
        
        Args:
            tier_thresholds: Thresholds for tier assignment [t1, t2, t3]
                            Default: [0.25, 0.5, 0.75]
        """
        self.tier_thresholds = tier_thresholds or [0.25, 0.5, 0.75]
        
        # Compile regex patterns
        self.simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
    
    def extract_features(self, query: str) -> QueryFeatures:
        """Extract features from a query for classification."""
        # Basic text processing
        words = query.split()
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate basic features
        length = len(query)
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        
        # Convert to lowercase for matching
        query_lower = query.lower()
        words_lower = set(query_lower.split())
        
        # Count feature occurrences
        question_words = len(words_lower & self.QUESTION_WORDS)
        technical_terms = sum(1 for term in self.TECHNICAL_TERMS if term in query_lower)
        
        # Estimate named entities (capitalized words not at sentence start)
        named_entities = 0
        for i, word in enumerate(words):
            if word[0].isupper() and i > 0 and words[i-1][-1] not in '.!?':
                named_entities += 1
        
        # Check for complexity indicators
        has_multiple_questions = query.count('?') > 1
        has_comparison = any(word in query_lower for word in self.COMPARISON_WORDS)
        has_reasoning_request = any(word in query_lower for word in self.REASONING_WORDS)
        has_code_request = any(word in query_lower for word in self.CODE_WORDS)
        has_math_request = bool(re.search(r'\d+\s*[\+\-\*\/\=]|\d+%|equation|calculate|compute', query_lower))
        
        # Determine domain
        if has_code_request:
            domain = "technical"
        elif has_math_request:
            domain = "technical"
        elif any(word in query_lower for word in ["write", "create", "story", "poem", "imagine"]):
            domain = "creative"
        elif question_words > 0:
            domain = "factual"
        else:
            domain = "general"
        
        return QueryFeatures(
            length=length,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            question_words=question_words,
            technical_terms=technical_terms,
            named_entities=named_entities,
            has_multiple_questions=has_multiple_questions,
            has_comparison=has_comparison,
            has_reasoning_request=has_reasoning_request,
            has_code_request=has_code_request,
            has_math_request=has_math_request,
            domain=domain,
        )
    
    def calculate_complexity_score(self, features: QueryFeatures) -> float:
        """
        Calculate complexity score from features.
        
        Returns a score between 0 (simple) and 1 (complex).
        
        Tier mapping (thresholds: 0.25, 0.5, 0.75):
        - Tier 1 (0.0-0.25): Greetings, simple factual
        - Tier 2 (0.25-0.5): Basic explanations
        - Tier 3 (0.5-0.75): Code, comparisons, reasoning
        - Tier 4 (0.75+): Complex multi-step tasks
        """
        score = 0.0
        
        # Base: Short simple questions start at 0
        # Long queries get a boost
        if features.word_count > 30:
            score += 0.15
        elif features.word_count > 50:
            score += 0.25
        
        # Multiple questions add complexity
        if features.has_multiple_questions:
            score += 0.25
        
        # Technical terms - only if multiple terms
        if features.technical_terms >= 2:
            score += 0.15
        
        # Reasoning requests are complex (tier 3)
        if features.has_reasoning_request:
            score += 0.35
        
        # Code WRITING requests are complex (tier 3)
        if features.has_code_request:
            score += 0.40
        
        # Comparisons require more thought (tier 3)
        if features.has_comparison:
            score += 0.35
        
        # Math requests
        if features.has_math_request:
            score += 0.15
        
        # Creative tasks are complex
        if features.domain == "creative":
            score += 0.20
        
        # Clamp to [0, 1]
        return min(max(score, 0.0), 1.0)
    
    def is_simple_query(self, query: str) -> bool:
        """Check if query matches simple patterns."""
        for pattern in self.simple_patterns:
            if pattern.search(query):
                return True
        return len(query.split()) <= 3
    
    def classify(self, query: str) -> Tuple[int, float, QueryFeatures]:
        """
        Classify a query and return tier assignment.
        
        Args:
            query: The user query
            
        Returns:
            Tuple of (tier, complexity_score, features)
        """
        # Quick check for very simple queries
        if self.is_simple_query(query):
            features = self.extract_features(query)
            return 1, 0.1, features
        
        # Extract features and calculate score
        features = self.extract_features(query)
        score = self.calculate_complexity_score(features)
        
        # Assign tier based on thresholds
        tier = 4  # Default to highest
        for i, threshold in enumerate(self.tier_thresholds):
            if score < threshold:
                tier = i + 1
                break
        
        return tier, score, features
    
    def get_tier_description(self, tier: int) -> str:
        """Get human-readable description of a tier."""
        descriptions = {
            1: "Simple (factual, greetings, short answers)",
            2: "Standard (basic Q&A, explanations)",
            3: "Complex (reasoning, technical queries)",
            4: "Advanced (multi-step, creative, code)",
        }
        return descriptions.get(tier, "Unknown")


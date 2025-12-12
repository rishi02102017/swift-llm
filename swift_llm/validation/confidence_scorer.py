"""
Confidence scoring for LLM responses.

Estimates the confidence/quality of generated responses
to determine if escalation is needed.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ConfidenceResult:
    """Result of confidence scoring."""
    
    score: float  # 0-1, higher = more confident
    
    # Component scores
    length_score: float = 0.0
    coherence_score: float = 0.0
    specificity_score: float = 0.0
    uncertainty_score: float = 0.0  # Inverted: high uncertainty = low confidence
    
    # Flags
    has_uncertainty_markers: bool = False
    has_refusal: bool = False
    is_truncated: bool = False
    
    # Details
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "length_score": self.length_score,
            "coherence_score": self.coherence_score,
            "specificity_score": self.specificity_score,
            "uncertainty_score": self.uncertainty_score,
            "has_uncertainty_markers": self.has_uncertainty_markers,
            "has_refusal": self.has_refusal,
            "is_truncated": self.is_truncated,
            "reasons": self.reasons,
        }


class ConfidenceScorer:
    """
    Estimates confidence in LLM responses.
    
    Uses heuristics to detect:
    - Uncertainty markers ("I'm not sure", "I think", etc.)
    - Refusals ("I cannot", "I don't have access")
    - Truncation (incomplete responses)
    - Response quality indicators
    
    This helps determine if a response should be cached
    and if we should escalate to a better model.
    """
    
    # Uncertainty phrases that reduce confidence
    UNCERTAINTY_MARKERS = [
        r"\bi('m| am) not (sure|certain)",
        r"\bi think\b",
        r"\bi believe\b",
        r"\bprobably\b",
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bmight be\b",
        r"\bcould be\b",
        r"\bit's possible\b",
        r"\bi'm unsure\b",
        r"\bto my knowledge\b",
        r"\bas far as i know\b",
        r"\bi don't have (enough )?(information|data)",
    ]
    
    # Refusal phrases
    REFUSAL_MARKERS = [
        r"\bi (cannot|can't|am unable to)\b",
        r"\bi don't have access\b",
        r"\bi'm not able to\b",
        r"\bi apologize,? but\b",
        r"\bunfortunately,? i\b",
        r"\bi'm sorry,? but i (cannot|can't)\b",
        r"\bas an ai\b",
        r"\bi don't have the ability\b",
    ]
    
    # Truncation indicators
    TRUNCATION_MARKERS = [
        r"\.\.\.$",  # Ends with ellipsis
        r"[^.!?]$",  # Doesn't end with punctuation
        r"\band$",  # Ends mid-sentence
        r"\bthe$",
        r"\bto$",
        r"\bof$",
    ]
    
    # Good quality indicators
    QUALITY_MARKERS = [
        r"\bfor example\b",
        r"\bspecifically\b",
        r"\bin conclusion\b",
        r"\bfirst(ly)?\b.*\bsecond(ly)?\b",  # Structured response
        r"\b\d+\.\s",  # Numbered list
        r"\b•\s",  # Bullet points
        r"```",  # Code blocks
    ]
    
    def __init__(self):
        # Compile regex patterns
        self.uncertainty_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.UNCERTAINTY_MARKERS
        ]
        self.refusal_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.REFUSAL_MARKERS
        ]
        self.truncation_patterns = [
            re.compile(p) for p in self.TRUNCATION_MARKERS
        ]
        self.quality_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.QUALITY_MARKERS
        ]
    
    def score(
        self,
        response: str,
        query: Optional[str] = None,
        finish_reason: str = "stop",
    ) -> ConfidenceResult:
        """
        Calculate confidence score for a response.
        
        Args:
            response: The generated response text
            query: Optional query for relevance checking
            finish_reason: Why generation stopped (stop, length, etc.)
            
        Returns:
            ConfidenceResult with overall score and details
        """
        reasons = []
        
        # Length score (very short responses are often low quality)
        word_count = len(response.split())
        if word_count < 5:
            length_score = 0.2
            reasons.append("Very short response")
        elif word_count < 20:
            length_score = 0.5
            reasons.append("Short response")
        elif word_count > 500:
            length_score = 0.9
            reasons.append("Detailed response")
        else:
            length_score = 0.7 + (min(word_count, 200) / 1000)
        
        # Uncertainty detection
        uncertainty_count = sum(
            1 for p in self.uncertainty_patterns if p.search(response)
        )
        has_uncertainty = uncertainty_count > 0
        uncertainty_score = max(0, 1.0 - (uncertainty_count * 0.15))
        if has_uncertainty:
            reasons.append(f"Found {uncertainty_count} uncertainty marker(s)")
        
        # Refusal detection
        has_refusal = any(p.search(response) for p in self.refusal_patterns)
        if has_refusal:
            reasons.append("Contains refusal")
        
        # Truncation detection
        is_truncated = (
            finish_reason == "length" or
            any(p.search(response[-50:]) for p in self.truncation_patterns)
        )
        if is_truncated:
            reasons.append("Response appears truncated")
        
        # Coherence score (basic: check for complete sentences)
        sentences = re.split(r'[.!?]+', response)
        complete_sentences = sum(1 for s in sentences if len(s.split()) > 3)
        coherence_score = min(complete_sentences / max(len(sentences), 1), 1.0)
        
        # Specificity score (presence of specific details)
        quality_matches = sum(1 for p in self.quality_patterns if p.search(response))
        specificity_score = min(0.5 + (quality_matches * 0.1), 1.0)
        if quality_matches > 0:
            reasons.append(f"Has {quality_matches} quality indicator(s)")
        
        # Calculate overall score
        # Weights: length 0.2, coherence 0.2, specificity 0.2, uncertainty 0.4
        base_score = (
            0.2 * length_score +
            0.2 * coherence_score +
            0.2 * specificity_score +
            0.4 * uncertainty_score
        )
        
        # Penalties
        if has_refusal:
            base_score *= 0.3  # Heavy penalty for refusals
        if is_truncated:
            base_score *= 0.7  # Moderate penalty for truncation
        
        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, base_score))
        
        return ConfidenceResult(
            score=final_score,
            length_score=length_score,
            coherence_score=coherence_score,
            specificity_score=specificity_score,
            uncertainty_score=uncertainty_score,
            has_uncertainty_markers=has_uncertainty,
            has_refusal=has_refusal,
            is_truncated=is_truncated,
            reasons=reasons,
        )
    
    def should_escalate(
        self,
        result: ConfidenceResult,
        threshold: float = 0.6,
    ) -> bool:
        """
        Determine if we should escalate to a better model.
        
        Args:
            result: Confidence result from scoring
            threshold: Minimum acceptable confidence
            
        Returns:
            True if escalation is recommended
        """
        # Always escalate refusals (model might be declining inappropriately)
        if result.has_refusal:
            return True
        
        # Escalate truncated responses
        if result.is_truncated:
            return True
        
        # Escalate low confidence
        if result.score < threshold:
            return True
        
        return False
    
    def should_cache(
        self,
        result: ConfidenceResult,
        threshold: float = 0.7,
    ) -> bool:
        """
        Determine if response is good enough to cache.
        
        Args:
            result: Confidence result from scoring
            threshold: Minimum confidence for caching
            
        Returns:
            True if response should be cached
        """
        # Don't cache refusals
        if result.has_refusal:
            return False
        
        # Don't cache truncated responses
        if result.is_truncated:
            return False
        
        # Cache if confidence is high enough
        return result.score >= threshold


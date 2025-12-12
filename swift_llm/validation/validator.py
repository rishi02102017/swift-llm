"""
Response Validator for SWIFT-LLM.

Combines confidence scoring with other validation checks
to ensure response quality.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from swift_llm.validation.confidence_scorer import ConfidenceScorer, ConfidenceResult
from swift_llm.inference.base import InferenceResult


@dataclass
class ValidationResult:
    """Result of response validation."""
    
    is_valid: bool
    confidence: ConfidenceResult
    
    # Recommendations
    should_cache: bool = True
    should_escalate: bool = False
    escalation_reason: Optional[str] = None
    
    # Quality flags
    quality_score: float = 1.0  # 0-1
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence.to_dict(),
            "should_cache": self.should_cache,
            "should_escalate": self.should_escalate,
            "escalation_reason": self.escalation_reason,
            "quality_score": self.quality_score,
            "issues": self.issues,
        }


class ResponseValidator:
    """
    Validates LLM responses for quality and appropriateness.
    
    Combines:
    - Confidence scoring (uncertainty, coherence, etc.)
    - Basic quality checks
    - Escalation recommendations
    - Caching decisions
    
    Example:
        validator = ResponseValidator()
        result = validator.validate(
            response="The capital of France is Paris.",
            query="What is the capital of France?"
        )
        
        if not result.is_valid or result.should_escalate:
            # Try with a better model
            pass
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        cache_threshold: float = 0.7,
        enable_escalation: bool = True,
    ):
        """
        Initialize the validator.
        
        Args:
            confidence_threshold: Minimum confidence for valid response
            cache_threshold: Minimum confidence for caching
            enable_escalation: Whether to recommend escalation
        """
        self.confidence_threshold = confidence_threshold
        self.cache_threshold = cache_threshold
        self.enable_escalation = enable_escalation
        
        self.scorer = ConfidenceScorer()
    
    def validate(
        self,
        response: str,
        query: Optional[str] = None,
        inference_result: Optional[InferenceResult] = None,
    ) -> ValidationResult:
        """
        Validate a response.
        
        Args:
            response: The generated response text
            query: Original query (for relevance checking)
            inference_result: Full inference result (if available)
            
        Returns:
            ValidationResult with quality assessment
        """
        issues = []
        
        # Get finish reason if available
        finish_reason = "stop"
        if inference_result:
            finish_reason = inference_result.finish_reason
        
        # Score confidence
        confidence = self.scorer.score(
            response=response,
            query=query,
            finish_reason=finish_reason,
        )
        
        # Check for empty response
        if not response or not response.strip():
            issues.append("Empty response")
            return ValidationResult(
                is_valid=False,
                confidence=confidence,
                should_cache=False,
                should_escalate=self.enable_escalation,
                escalation_reason="Empty response",
                quality_score=0.0,
                issues=issues,
            )
        
        # Check inference errors
        if inference_result and not inference_result.success:
            issues.append(f"Inference error: {inference_result.error_message}")
            return ValidationResult(
                is_valid=False,
                confidence=confidence,
                should_cache=False,
                should_escalate=self.enable_escalation,
                escalation_reason=inference_result.error_message,
                quality_score=0.0,
                issues=issues,
            )
        
        # Add confidence issues
        issues.extend(confidence.reasons)
        
        # Determine validity
        is_valid = (
            confidence.score >= self.confidence_threshold and
            not confidence.has_refusal and
            not confidence.is_truncated
        )
        
        # Determine escalation
        should_escalate = False
        escalation_reason = None
        
        if self.enable_escalation:
            if confidence.has_refusal:
                should_escalate = True
                escalation_reason = "Response contains refusal"
            elif confidence.is_truncated:
                should_escalate = True
                escalation_reason = "Response was truncated"
            elif confidence.score < self.confidence_threshold:
                should_escalate = True
                escalation_reason = f"Low confidence: {confidence.score:.2f}"
        
        # Determine caching
        should_cache = self.scorer.should_cache(confidence, self.cache_threshold)
        
        # Overall quality score
        quality_score = confidence.score
        if confidence.has_refusal:
            quality_score *= 0.3
        if confidence.is_truncated:
            quality_score *= 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            should_cache=should_cache,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason,
            quality_score=quality_score,
            issues=issues,
        )
    
    def validate_for_caching(
        self,
        response: str,
        query: str,
    ) -> bool:
        """
        Quick check if response is suitable for caching.
        
        Args:
            response: The response text
            query: The original query
            
        Returns:
            True if response should be cached
        """
        result = self.validate(response, query)
        return result.should_cache
    
    def get_quality_grade(self, quality_score: float) -> str:
        """Get letter grade for quality score."""
        if quality_score >= 0.9:
            return "A"
        elif quality_score >= 0.8:
            return "B"
        elif quality_score >= 0.7:
            return "C"
        elif quality_score >= 0.6:
            return "D"
        else:
            return "F"


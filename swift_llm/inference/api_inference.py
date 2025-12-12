"""
API-based inference backends for SWIFT-LLM.

Supports Groq (fast) and OpenAI (quality) APIs.
"""

import os
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from swift_llm.inference.base import InferenceBackend, InferenceResult, Message


class GroqInference(InferenceBackend):
    """
    Groq API inference backend.
    
    Extremely fast inference (~500 tokens/sec) with reasonable quality.
    Ideal for Tier 2 (standard queries).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        tier: int = 2,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self._tier = tier
        self._client = None
        
        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                print("Warning: groq package not installed. Install with: pip install groq")
    
    @property
    def name(self) -> str:
        return f"groq/{self.model}"
    
    @property
    def tier(self) -> int:
        return self._tier
    
    def is_available(self) -> bool:
        return self._client is not None and self.api_key is not None
    
    def generate(
        self,
        messages: List[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> InferenceResult:
        """Generate response using Groq API."""
        if not self.is_available():
            return InferenceResult(
                text="",
                model=self.model,
                tier=self._tier,
                latency_ms=0,
                success=False,
                error_message="Groq API not available. Set GROQ_API_KEY.",
            )
        
        start_time = time.perf_counter()
        
        try:
            # Convert messages to API format
            api_messages = [{"role": m.role, "content": m.content} for m in messages]
            
            # Call Groq API
            response = self._client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract response
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Token counts
            tokens_input = response.usage.prompt_tokens if response.usage else 0
            tokens_output = response.usage.completion_tokens if response.usage else 0
            tokens_total = tokens_input + tokens_output
            
            # Calculate tokens per second
            tps = tokens_output / (latency_ms / 1000) if latency_ms > 0 else 0
            
            # Estimate cost (Groq is very cheap)
            cost = (tokens_total / 1000) * 0.0001
            
            return InferenceResult(
                text=content,
                model=self.model,
                tier=self._tier,
                latency_ms=latency_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_per_second=tps,
                finish_reason=finish_reason,
                estimated_cost=cost,
                success=True,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return InferenceResult(
                text="",
                model=self.model,
                tier=self._tier,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
            )


class OpenAIInference(InferenceBackend):
    """
    OpenAI API inference backend.
    
    High quality responses, ideal for Tier 4 (complex queries).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        tier: int = 4,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._tier = tier
        self._client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
    
    @property
    def name(self) -> str:
        return f"openai/{self.model}"
    
    @property
    def tier(self) -> int:
        return self._tier
    
    def is_available(self) -> bool:
        return self._client is not None and self.api_key is not None
    
    def generate(
        self,
        messages: List[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> InferenceResult:
        """Generate response using OpenAI API."""
        if not self.is_available():
            return InferenceResult(
                text="",
                model=self.model,
                tier=self._tier,
                latency_ms=0,
                success=False,
                error_message="OpenAI API not available. Set OPENAI_API_KEY.",
            )
        
        start_time = time.perf_counter()
        
        try:
            # Convert messages to API format
            api_messages = [{"role": m.role, "content": m.content} for m in messages]
            
            # Call OpenAI API
            response = self._client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract response
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Token counts
            tokens_input = response.usage.prompt_tokens if response.usage else 0
            tokens_output = response.usage.completion_tokens if response.usage else 0
            tokens_total = tokens_input + tokens_output
            
            # Calculate tokens per second
            tps = tokens_output / (latency_ms / 1000) if latency_ms > 0 else 0
            
            # Estimate cost (varies by model)
            if "gpt-4o-mini" in self.model:
                cost = (tokens_input / 1000) * 0.00015 + (tokens_output / 1000) * 0.0006
            elif "gpt-4o" in self.model:
                cost = (tokens_input / 1000) * 0.005 + (tokens_output / 1000) * 0.015
            else:
                cost = (tokens_total / 1000) * 0.002  # Default estimate
            
            return InferenceResult(
                text=content,
                model=self.model,
                tier=self._tier,
                latency_ms=latency_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_per_second=tps,
                finish_reason=finish_reason,
                estimated_cost=cost,
                success=True,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return InferenceResult(
                text="",
                model=self.model,
                tier=self._tier,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
            )


class APIInference:
    """
    Unified API inference manager.
    
    Manages multiple backends and routes to appropriate one based on tier.
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.backends: Dict[int, InferenceBackend] = {}
        
        # Initialize Groq backends (Tier 2-3)
        groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if groq_key:
            self.backends[2] = GroqInference(
                api_key=groq_key,
                model="llama-3.1-8b-instant",
                tier=2,
            )
            self.backends[3] = GroqInference(
                api_key=groq_key,
                model="llama-3.1-70b-versatile",
                tier=3,
            )
        
        # Initialize OpenAI backend (Tier 4)
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.backends[4] = OpenAIInference(
                api_key=openai_key,
                model="gpt-4o-mini",
                tier=4,
            )
    
    def get_backend(self, tier: int) -> Optional[InferenceBackend]:
        """Get the backend for a specific tier."""
        return self.backends.get(tier)
    
    def generate(
        self,
        tier: int,
        messages: List[Message],
        **kwargs
    ) -> InferenceResult:
        """Generate using the backend for the specified tier."""
        backend = self.get_backend(tier)
        
        if backend is None:
            # Try to find a fallback
            for t in [2, 3, 4]:
                if t in self.backends:
                    backend = self.backends[t]
                    break
        
        if backend is None:
            return InferenceResult(
                text="",
                model="none",
                tier=tier,
                latency_ms=0,
                success=False,
                error_message="No inference backend available. Set GROQ_API_KEY or OPENAI_API_KEY.",
            )
        
        return backend.generate(messages, **kwargs)
    
    def is_available(self) -> bool:
        """Check if any backend is available."""
        return any(b.is_available() for b in self.backends.values())
    
    def list_available_backends(self) -> List[str]:
        """List available backends."""
        return [
            f"Tier {tier}: {backend.name}"
            for tier, backend in sorted(self.backends.items())
            if backend.is_available()
        ]


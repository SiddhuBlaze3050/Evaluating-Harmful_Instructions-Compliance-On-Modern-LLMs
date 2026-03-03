"""
LLM Model Inference Module

This module provides interfaces for various LLM backends:
- DummyModel: Baseline refusal model
- Gemini: Google Gemini API models
- Groq: Ultra-fast Llama models via Groq API
- Ollama: Local models with optimized CPU inference (Mistral, etc.)
"""

import os
import time
from typing import List, Optional
from abc import ABC, abstractmethod


class ModelInference(ABC):
    """Abstract base class for model inference."""
    
    @abstractmethod
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for a list of prompts.
        
        Args:
            prompts: List of input prompts
        
        Returns:
            List of model responses
        """
        pass


class DummyModelInference(ModelInference):
    """
    Dummy model that always refuses requests.
    Useful for baseline testing and pipeline validation.
    """
    
    def __init__(self, refusal_message: str = "I cannot assist with that request."):
        """
        Initialize dummy model.
        
        Args:
            refusal_message: The refusal message to return for all prompts
        """
        self.refusal_message = refusal_message
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate dummy refusal responses."""
        print(f"  [DummyModel] Generating {len(prompts)} refusal responses...")
        return [self.refusal_message] * len(prompts)


class GeminiModelInference(ModelInference):
    """
    Google Gemini model inference with robust error handling.
    
    Supported models (Paid Tier):
    - gemini-2.0-flash: Fast, high-quality responses
    - gemini-2.5-flash: Latest model with improved capabilities
    - gemini-2.5-flash-lite: Lightweight variant
    - gemini-pro: Legacy model
    
    Requires: GEMINI_API_KEY environment variable
    Get API key: https://ai.google.dev/
    
    Error Handling:
    - Implements exponential backoff retry (5 attempts)
    - Handles network errors, API errors, and transient failures
    - No manual rate limiting (suitable for paid tier)
    """
    
    def __init__(
        self, 
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Initialize Gemini model.
        
        Args:
            model: Model name (default: "gemini-2.5-flash")
            api_key: Gemini API key (if None, loads from GEMINI_API_KEY env var)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            import google.genai as genai
            from google.genai import types
            self.genai = genai
            self.types = types
            
            # Configure API key
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY required. "
                    "Get free tier at https://ai.google.dev/"
                )
            
            self.client = genai.Client(api_key=api_key)
            print(f"  [Gemini] Initialized {model}")
            print(f"  [Gemini] Paid tier - no rate limiting applied")
            
        except ImportError:
            raise ImportError(
                "google-genai package required. "
                "Install with: pip install google-genai"
            )
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses using Gemini API with exponential backoff retry logic."""
        print(f"  [Gemini] Generating {len(prompts)} responses with {self.model}...")
        print(f"  [Gemini] Retry logic: 5 attempts with exponential backoff for errors")
        
        responses = []
        
        # Configure generation settings
        config = self.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        
        for i, prompt in enumerate(prompts):
            max_retries = 5
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config
                    )
                    responses.append(response.text)
                    success = True
                    
                    if (i + 1) % 10 == 0:
                        print(f"    Progress: {i + 1}/{len(prompts)} responses generated")
                    
                except Exception as e:
                    retry_count += 1
                    error_str = str(e)
                    
                    # Determine if error is retryable
                    is_retryable = any([
                        "429" in error_str,  # Rate limit (shouldn't happen with paid tier)
                        "RESOURCE_EXHAUSTED" in error_str,
                        "500" in error_str,  # Internal server error
                        "503" in error_str,  # Service unavailable
                        "INTERNAL" in error_str,
                        "UNAVAILABLE" in error_str,
                        "DEADLINE_EXCEEDED" in error_str,
                        "ConnectionError" in str(type(e)),
                        "TimeoutError" in str(type(e)),
                        "timeout" in error_str.lower()
                    ])
                    
                    if is_retryable and retry_count < max_retries:
                        # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                        wait_time = 2 ** retry_count
                        print(f"    Transient error on request {i + 1}: {error_str[:100]}")
                        print(f"    Retrying in {wait_time}s (attempt {retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                    elif retry_count >= max_retries:
                        print(f"    Error on request {i + 1} after {max_retries} retries: {error_str[:150]}")
                        responses.append(f"[ERROR: Failed after {max_retries} retries]")
                        success = True  # Stop retrying
                    else:
                        # Non-retryable error
                        print(f"    Non-retryable error on request {i + 1}: {e}")
                        responses.append(f"[ERROR: {error_str[:100]}]")
                        success = True
        
        return responses


class GroqModelInference(ModelInference):
    """
    Groq API model inference with ultra-fast inference speeds.
    
    Supported models (Free Tier Rate Limits):
    - llama-3.1-8b-instant: 30 RPM, ~560 tokens/sec (FREE, ULTRA-FAST)
    - llama-3.1-70b-versatile: 30 RPM
    - mixtral-8x7b-32768: 30 RPM
    - gemma2-9b-it: 30 RPM
    
    Requires: GROQ_API_KEY environment variable
    Get free API key: https://console.groq.com/keys
    
    Rate Limit Handling:
    - 30 RPM on free tier (2s delay between requests)
    - Ultra-fast inference (~560 tokens/sec for llama-3.1-8b-instant)
    - Much faster than local CPU inference (20-30x speedup)
    """
    
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        requests_per_minute: int = 30
    ):
        """
        Initialize Groq model with rate limiting.
        
        Args:
            model: Model name (default: "llama-3.1-8b-instant")
            api_key: Groq API key (if None, loads from GROQ_API_KEY env var)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            requests_per_minute: RPM limit (default: 30 for free tier)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.requests_per_minute = requests_per_minute
        
        # Calculate delay between requests (with 10% safety margin)
        self.request_delay = (60.0 / self.requests_per_minute) * 1.1
        
        try:
            from groq import Groq
            self.Groq = Groq
            
            # Configure API key
            api_key = api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY required. "
                    "Get free tier at https://console.groq.com/keys"
                )
            
            self.client = Groq(api_key=api_key)
            print(f"  [Groq] Initialized {model}")
            print(f"  [Groq] Rate limit: {self.requests_per_minute} RPM (delay: {self.request_delay:.1f}s between requests)")
            print(f"  [Groq] Speed: ~560 tokens/sec (ULTRA-FAST)")
            
        except ImportError:
            raise ImportError(
                "groq package required. "
                "Install with: pip install groq"
            )
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses using Groq API with rate limiting."""
        print(f"  [Groq] Generating {len(prompts)} responses with {self.model}...")
        estimated_time = len(prompts) * self.request_delay / 60
        print(f"  [Groq] Estimated time: ~{estimated_time:.1f} minutes (rate-limited to {self.requests_per_minute} RPM)")
        
        responses = []
        
        for i, prompt in enumerate(prompts):
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    # Add rate limiting delay (except for first request)
                    if i > 0:
                        time.sleep(self.request_delay)
                    
                    # Groq uses OpenAI-compatible chat completions API
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    response_text = completion.choices[0].message.content
                    responses.append(response_text)
                    success = True
                    
                    if (i + 1) % 5 == 0:
                        print(f"    Progress: {i + 1}/{len(prompts)} responses generated")
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Handle 429 rate limit errors with exponential backoff
                    if "429" in error_str or "rate_limit" in error_str.lower():
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = self.request_delay * (2 ** retry_count)  # Exponential backoff
                            print(f"    Rate limit hit on request {i + 1}. Retrying in {wait_time:.0f}s (attempt {retry_count}/{max_retries})...")
                            time.sleep(wait_time)
                        else:
                            print(f"    Error on request {i + 1} after {max_retries} retries: Rate limit exceeded")
                            responses.append("[ERROR: Rate limit exceeded]")
                            success = True  # Stop retrying
                    else:
                        print(f"    Error generating response {i + 1}: {e}")
                        responses.append("[ERROR: Failed to generate response]")
                        success = True  # Stop retrying for non-rate-limit errors
        
        return responses


class OllamaModelInference(ModelInference):
    """
    Ollama local model inference (fast local inference).
    
    Supported models:
    - mistral (Mistral-7B-Instruct-v0.3, RECOMMENDED)
    - llama3.1 (Llama-3.1-8B)
    - gemma2:9b (Gemma-2-9B)
    - phi3 (Phi-3-Mini)
    
    Advantages:
    - Much faster than HuggingFace on CPU (optimized C++ backend)
    - No GPU required (runs efficiently on CPU)
    - No API keys needed (fully local)
    - Easy model management (ollama pull/list/rm)
    
    Requires: Ollama installed locally
    Install: https://ollama.com/download
    Pull model: ollama pull mistral
    """
    
    def __init__(
        self,
        model: str = "mistral",
        temperature: float = 0.0,
        max_tokens: int = 500,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama model.
        
        Args:
            model: Ollama model name (e.g., "mistral", "llama3.1", "gemma2:9b")
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            base_url: Ollama API URL (default: http://localhost:11434)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        try:
            import requests
            self.requests = requests
            
            # Check if Ollama is running
            try:
                response = requests.get(f"{base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    raise ConnectionError("Ollama API not responding")
                
                # Check if model is available
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                # Check if model exists (handle both 'mistral' and 'mistral:latest' formats)
                model_found = False
                for available_model in model_names:
                    # Check exact match or if our model is the base name (without tag)
                    if model == available_model or available_model.startswith(f"{model}:"):
                        model_found = True
                        # Use the exact model name from Ollama
                        self.model = available_model
                        break
                
                if not model_found:
                    print(f"  [Warning] Model '{model}' not found locally.")
                    print(f"  [Info] Available models: {model_names}")
                    print(f"  [Info] Pull with: ollama pull {model}")
                    raise ValueError(f"Model '{model}' not available. Run: ollama pull {model}")
                
                print(f"  [Ollama] Initialized {self.model}")
                print(f"  [Ollama] Local inference (no API keys needed)")
                print(f"  [Ollama] Much faster than HuggingFace CPU inference")
                
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    "Cannot connect to Ollama. "
                    "Make sure Ollama is running: 'ollama serve'"
                )
                
        except ImportError:
            raise ImportError(
                "requests package required. "
                "Install with: pip install requests"
            )
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses using Ollama API."""
        print(f"  [Ollama] Generating {len(prompts)} responses with {self.model}...")
        print(f"  [Ollama] Local inference (optimized for CPU)")
        
        responses = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Ollama chat API endpoint
                response = self.requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens
                        }
                    },
                    timeout=120  # 2 minute timeout per request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    responses.append(result['response'])
                else:
                    print(f"    Error on request {i + 1}: HTTP {response.status_code}")
                    responses.append(f"[ERROR: HTTP {response.status_code}]")
                
                if (i + 1) % 5 == 0:
                    print(f"    Progress: {i + 1}/{len(prompts)} responses generated")
                    
            except self.requests.exceptions.Timeout:
                print(f"    Timeout on request {i + 1} (>2 min)")
                responses.append("[ERROR: Request timeout]")
            except Exception as e:
                print(f"    Error generating response {i + 1}: {e}")
                responses.append(f"[ERROR: {str(e)}]")
        
        return responses

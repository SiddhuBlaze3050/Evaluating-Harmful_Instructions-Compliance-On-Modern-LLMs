"""
StrongREJECT Pipeline: Modular LLM Safety Evaluation Framework

This package provides:
1. Prompt loading and sampling utilities
2. Model inference classes (Gemini, Groq/Llama, Ollama)
3. StrongREJECT automated scoring

"""

from .prompt import load_prompts, prepare_prompt_dataset, sample_prompts, add_harmful_labels, save_prompts_with_labels
from .llm_model import (
    ModelInference,
    DummyModelInference,
    GeminiModelInference,
    GroqModelInference,
    OllamaModelInference
)
from .strongreject_scoring import StrongRejectEvaluator, evaluate_model_responses

__all__ = [
    # Prompt utilities
    'load_prompts',
    'prepare_prompt_dataset',
    'sample_prompts',
    'add_harmful_labels',
    'save_prompts_with_labels',
    
    # Model inference classes
    'ModelInference',
    'DummyModelInference',
    'GeminiModelInference',
    'GroqModelInference',
    'OllamaModelInference',
    
    # StrongREJECT evaluation
    'StrongRejectEvaluator',
    'evaluate_model_responses'
]

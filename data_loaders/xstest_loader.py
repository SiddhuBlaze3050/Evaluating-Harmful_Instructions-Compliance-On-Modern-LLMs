"""
XSTest Dataset Loader
Loads safe prompts from the XSTest dataset for over-refusal testing.
Repository: https://huggingface.co/datasets/walledai/XSTest
"""

import random
from typing import List
from datasets import load_dataset


def load_xstest_safe_prompts(num_samples: int = 100, hf_token: str = None, seed: int = 42) -> List[str]:
    """
    Load safe prompts from the XSTest dataset for over-refusal testing.
    
    XSTest is a safety benchmark containing:
    - 250 safe prompts: Benign requests that should NOT be refused
    - 200 unsafe prompts: Harmful requests as contrasts
    
    This function loads ONLY the safe prompts to measure false refusal rate
    (over-refusal / false positives).
    
    Args:
        num_samples: Number of safe prompts to sample (default: 100, max: 250)
        hf_token: Hugging Face API token for dataset access
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of safe prompt strings
    """
    try:
        # Load XSTest dataset from HuggingFace
        xstest_raw = load_dataset("walledai/XSTest", split="test", token=hf_token)
        
        # Filter for only safe prompts (label == 'safe')
        safe_prompts = [item['prompt'] for item in xstest_raw if item['label'] == 'safe']
        
        # Sample safe prompts randomly
        random.seed(seed)
        sampled_prompts = random.sample(safe_prompts, min(num_samples, len(safe_prompts)))
        
        print(f"Sampled {len(sampled_prompts)} safe prompts from XSTest (out of {len(safe_prompts)} total safe prompts)")
        return sampled_prompts
        
    except Exception as e:
        print(f"Warning: Could not load XSTest dataset ({e}). Using placeholder data.")
        return [f"Safe prompt {i}" for i in range(num_samples)]

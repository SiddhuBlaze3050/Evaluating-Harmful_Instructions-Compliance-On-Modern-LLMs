"""
ToxicChat Dataset Loader
Loads toxic conversation examples from lmsys/toxic-chat dataset.
ToxicChat v0124: Updated January 2024
Performance: 42.54% TPR at 0.1% FPR
"""

from datasets import load_dataset
from typing import List
import random


def load_toxicchat_examples(
    num_samples: int = 50,
    toxicity_threshold: float = 0.5,
    hf_token: str = None,
    seed: int = 42
) -> List[str]:
    """
    Load toxic conversation examples from ToxicChat dataset.
    
    Args:
        num_samples: Number of toxic examples to sample
        toxicity_threshold: Minimum toxicity score to include (0-1)
        hf_token: HuggingFace token for dataset access
        seed: Random seed for reproducibility
    
    Returns:
        List of toxic conversation prompts
    """
    try:
        print(f"Loading ToxicChat dataset from HuggingFace...")
        
        # Load the dataset
        dataset = load_dataset(
            "lmsys/toxic-chat",
            "toxicchat0124",  # v0124 version
            split="train",
            token=hf_token
        )
        
        print(f"Total ToxicChat examples: {len(dataset)}")
        
        # Filter for toxic examples above threshold
        toxic_examples = [
            example for example in dataset 
            if example.get('toxicity', 0) >= toxicity_threshold
        ]
        
        print(f"Toxic examples (threshold >= {toxicity_threshold}): {len(toxic_examples)}")
        
        # Sample prompts
        random.seed(seed)
        if len(toxic_examples) < num_samples:
            print(f"Warning: Only {len(toxic_examples)} toxic examples available, using all")
            sampled = toxic_examples
        else:
            sampled = random.sample(toxic_examples, num_samples)
        
        # Extract user messages (first turn of conversation)
        prompts = []
        for example in sampled:
            # ToxicChat has 'user_input' or 'prompt' field
            user_input = example.get('user_input') or example.get('prompt', '')
            if user_input:
                prompts.append(user_input)
        
        print(f"Sampled {len(prompts)} toxic prompts from ToxicChat")
        
        return prompts
        
    except Exception as e:
        print(f"Error loading ToxicChat dataset: {e}")
        print("Returning empty list...")
        return []


def load_toxicchat_conversations(
    num_samples: int = 50,
    hf_token: str = None,
    seed: int = 42
) -> List[str]:
    """
    Load full toxic conversations (multi-turn) from ToxicChat.
    
    Args:
        num_samples: Number of conversations to sample
        hf_token: HuggingFace token
        seed: Random seed
    
    Returns:
        List of formatted multi-turn conversations
    """
    try:
        dataset = load_dataset(
            "lmsys/toxic-chat",
            "toxicchat0124",
            split="train",
            token=hf_token
        )
        
        # Sample conversations
        random.seed(seed)
        sampled = random.sample(list(dataset), min(num_samples, len(dataset)))
        
        conversations = []
        for example in sampled:
            # Format conversation turns
            conv = example.get('conversation', [])
            if conv:
                formatted = "\n".join([
                    f"[{turn['role']}] {turn['content']}" 
                    for turn in conv
                ])
                conversations.append(formatted)
        
        return conversations
        
    except Exception as e:
        print(f"Error loading ToxicChat conversations: {e}")
        return []

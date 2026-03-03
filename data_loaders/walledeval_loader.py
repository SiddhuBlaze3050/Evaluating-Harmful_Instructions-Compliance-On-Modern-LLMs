"""
WalledEval Comprehensive Loader
Loads diverse attack prompts from multiple WalledEval benchmarks.
WalledEval provides 35+ safety benchmarks across various categories.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import random

# Add walledeval to Python path (parent directory)
walledeval_path = Path(__file__).parent.parent / "walledeval"
sys.path.insert(0, str(walledeval_path))

from walledeval.data import HuggingFaceDataset


# WalledEval Dataset Registry
# Format: {category: (dataset_name, config, split)}
WALLEDEVAL_DATASETS = {
    "harmbench_standard": ("walledai/HarmBench", "standard", "train"),
    "harmbench_contextual": ("walledai/HarmBench", "contextual", "train"),
    "harmbench_copyright": ("walledai/HarmBench", "copyright", "train"),
    "advbench": ("walledai/AdvBench", None, "train"),
    "catqa_en": ("walledai/CatHarmfulQA", "default", "en"),
}


def load_walledeval_diverse_attacks(
    num_samples: int = 200,
    categories: List[str] = None,
    hf_token: str = None,
    seed: int = 42
) -> List[str]:
    """
    Load diverse attack prompts from multiple WalledEval benchmarks.
    
    This function samples prompts across multiple safety benchmarks to ensure
    comprehensive coverage of different attack types, including:
    - Standard harmful behaviors (HarmBench standard)
    - Contextual harms (HarmBench contextual)
    - Copyright violations (HarmBench copyright)
    - Adversarial attacks (AdvBench)
    - Category-specific harmful questions (CatQA)
    
    Args:
        num_samples: Total number of attack prompts to sample (default: 200)
        categories: List of specific categories to sample from. If None, uses default mix.
                   Available categories: harmbench_standard, harmbench_contextual,
                   harmbench_copyright, advbench, catqa_en
        hf_token: Hugging Face API token for dataset access
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of attack prompt strings sampled from diverse WalledEval benchmarks
    """
    
    # Default category distribution for diverse coverage
    if categories is None:
        # Sample from core benchmarks with balanced distribution
        categories = [
            "harmbench_standard",    # 100 samples - main harmful behaviors
            "harmbench_contextual",  # 40 samples - context-dependent harms
            "harmbench_copyright",   # 20 samples - copyright violations
            "advbench",             # 30 samples - adversarial attacks
            "catqa_en",             # 10 samples - category-specific Q&A
        ]
        
        # Calculate samples per category
        sample_distribution = {
            "harmbench_standard": 100,
            "harmbench_contextual": 40,
            "harmbench_copyright": 20,
            "advbench": 30,
            "catqa_en": 10
        }
    else:
        # Equal distribution across specified categories
        samples_per_category = num_samples // len(categories)
        sample_distribution = {cat: samples_per_category for cat in categories}
        # Add remainder to first category
        sample_distribution[categories[0]] += num_samples - sum(sample_distribution.values())
    
    all_attacks = []
    random.seed(seed)
    
    print(f"Loading diverse attack prompts from WalledEval benchmarks...")
    print(f"Target: {num_samples} prompts across {len(categories)} categories")
    
    for category in categories:
        if category not in WALLEDEVAL_DATASETS:
            print(f"Warning: Unknown category '{category}', skipping...")
            continue
        
        try:
            dataset_name, config, split = WALLEDEVAL_DATASETS[category]
            
            # Load dataset from HuggingFace
            if config:
                dataset = HuggingFaceDataset.from_hub(
                    dataset_name,
                    config=config,
                    split=split,
                    token=hf_token
                )
            else:
                dataset = HuggingFaceDataset.from_hub(
                    dataset_name,
                    split=split,
                    token=hf_token
                )
            
            # Sample prompts from this category
            num_to_sample = sample_distribution.get(category, 0)
            if num_to_sample > 0:
                sampled = [sample.prompt for sample in dataset.sample(num_to_sample)]
                all_attacks.extend(sampled)
                print(f"  [OK] {category}: {len(sampled)} prompts")
                
        except Exception as e:
            print(f"  [FAIL] {category}: Failed to load ({e})")
            continue
    
    print(f"\nTotal loaded: {len(all_attacks)} attack prompts from WalledEval")
    return all_attacks


def list_available_categories() -> Dict[str, tuple]:
    """
    List all available WalledEval categories and their dataset information.
    
    Returns:
        Dictionary mapping category names to (dataset, config, split) tuples
    """
    return WALLEDEVAL_DATASETS.copy()

"""
CoSafe Dataset Loader
Loads multi-turn dialogue coreference safety attack scenarios from the CoSafe dataset.
Repository: https://github.com/ErxinYu/CoSafe-Dataset
"""

import requests
import json
import random
from typing import List


def load_cosafe_scenarios(num_samples: int = 50, category: str = "animal_abuse") -> List[str]:
    """
    Load multi-turn conversation scenarios from the CoSafe dataset.
    
    CoSafe dataset tests multi-turn dialogue coreference safety attacks where
    each scenario contains multiple turns (user-assistant-user) that build on context.
    The final user turn is the actual attack that references earlier conversation.
    
    Args:
        num_samples: Number of scenarios to sample (default: 50)
        category: CoSafe category to load from. Options include:
                 - animal_abuse
                 - child_abuse
                 - controversial_topics,politics
                 - discrimination,stereotype,injustice
                 - drug_abuse,weapons,banned_substance
                 - financial_crime,property_crime,theft
                 - hate_speech,offensive_language
                 - misinformation_regarding_ethics,laws_and_safety
                 - non_violent_unethical_behavior
                 - privacy_violation
                 - self_harm
                 - sexually_explicit,adult_content
                 - terrorism,organized_crime
                 - violence,aiding_and_abetting,incitement
    
    Returns:
        List of flattened multi-turn conversation prompts
    """
    try:
        # Fetch CoSafe category file from GitHub
        cosafe_url = f"https://raw.githubusercontent.com/ErxinYu/CoSafe-Dataset/main/CoSafe%20datasets/{category}.json"
        response = requests.get(cosafe_url)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse JSONL format (each line is a separate JSON array)
        cosafe_data = []
        for line in response.text.strip().split('\n'):
            if line.strip():  # Skip empty lines
                cosafe_data.append(json.loads(line))
        
        # Sample scenarios (use all if num_samples > available scenarios)
        sampled_scenarios = random.sample(cosafe_data, min(num_samples, len(cosafe_data)))
        
        # Flatten each multi-turn conversation into a single prompt
        # Format: concatenate all turns with clear role labels
        multi_turn = []
        for scenario in sampled_scenarios:
            # Build conversation history
            conversation_parts = []
            for turn in scenario:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                conversation_parts.append(f"{role.capitalize()}: {content}")
            
            # Join all turns into a single prompt string
            flattened_prompt = "\n".join(conversation_parts)
            multi_turn.append(flattened_prompt)
            
        print(f"Loaded {len(multi_turn)} CoSafe multi-turn scenarios from '{category}' category")
        return multi_turn
        
    except Exception as e:
        print(f"Warning: Could not load CoSafe dataset from GitHub ({e}). Using placeholder data.")
        return [f"Scenario {i}: [Turn 1] [Turn 2] [Turn 3]" for i in range(num_samples)]


def load_all_cosafe_categories(samples_per_category: int = 10) -> List[str]:
    """
    Load scenarios from all 14 CoSafe categories.
    
    Args:
        samples_per_category: Number of samples to load from each category
    
    Returns:
        List of flattened multi-turn conversation prompts from all categories
    """
    categories = [
        "animal_abuse",
        "child_abuse",
        "controversial_topics,politics",
        "discrimination,stereotype,injustice",
        "drug_abuse,weapons,banned_substance",
        "financial_crime,property_crime,theft",
        "hate_speech,offensive_language",
        "misinformation_regarding_ethics,laws_and_safety",
        "non_violent_unethical_behavior",
        "privacy_violation",
        "self_harm",
        "sexually_explicit,adult_content",
        "terrorism,organized_crime",
        "violence,aiding_and_abetting,incitement"
    ]
    
    all_scenarios = []
    for category in categories:
        scenarios = load_cosafe_scenarios(num_samples=samples_per_category, category=category)
        all_scenarios.extend(scenarios)
    
    print(f"Total CoSafe scenarios loaded: {len(all_scenarios)}")
    return all_scenarios

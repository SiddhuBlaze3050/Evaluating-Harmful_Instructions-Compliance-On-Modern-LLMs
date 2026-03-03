import os
import pandas as pd
from dotenv import load_dotenv
from data_loaders import (
    load_all_cosafe_categories, 
    load_xstest_safe_prompts, 
    load_walledeval_diverse_attacks,
    load_toxicchat_examples
)

load_dotenv()  # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get OpenAI API key from .env
HF_TOKEN = os.getenv("HF_TOKEN")  # Get Hugging Face token from .env

def load_evaluation_dataset():
    """
    Aggregates prompts from WalledEval, XSTest, CoSafe, and ToxicChat into a single DataFrame.
    
    Dataset Composition:
    - 200 diverse attack prompts from WalledEval (5 benchmarks across categories)
    - 100 safe prompts from XSTest (over-refusal testing)
    - ~56 multi-turn scenarios from CoSafe (coreference safety attacks)
    - 50 toxic examples from ToxicChat v0124 (baseline comparison)
    
    Total: ~406 prompts
    """
    
    # 1. WalledEval: 200 Diverse Attack Prompts
    # Samples across: HarmBench (standard/contextual/copyright), AdvBench, CatQA
    attacks = load_walledeval_diverse_attacks(num_samples=200, hf_token=HF_TOKEN, seed=42)
    
    # 2. XSTest: 100 Safe Prompts (Over-Refusal Testing)
    xstest = load_xstest_safe_prompts(num_samples=100, hf_token=HF_TOKEN, seed=42)
    
    # 3. CoSafe: ~56 Multi-Turn Scenarios
    multi_turn = load_all_cosafe_categories(samples_per_category=4)
    
    # 4. ToxicChat: 50 Toxic Examples
    toxicchat = load_toxicchat_examples(
        num_samples=50, 
        toxicity_threshold=0.5,  # Medium-high toxicity
        hf_token=HF_TOKEN, 
        seed=42
    )

    # Combine into a unified DataFrame
    df_data = []
    for p in attacks:
        df_data.append({"source": "walledeval_diverse", "prompt": p})
    for p in xstest:
        df_data.append({"source": "xstest_benign", "prompt": p})
    for p in multi_turn:
        df_data.append({"source": "cosafe_multiturn", "prompt": p})
    for p in toxicchat:
        df_data.append({"source": "toxicchat_toxic", "prompt": p})
        
    return pd.DataFrame(df_data)


if __name__ == "__main__":
    # 1. Generate the Dataset
    dataset_df = load_evaluation_dataset()
    
    print(f"\n{'='*80}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total prompts: {len(dataset_df)}")
    print("\nBreakdown by source:")
    print(dataset_df['source'].value_counts())
    
    print("\n" + "="*80)
    print("PROMPTS DATAFRAME")
    print("="*80)
    
    # Display the dataframe
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(f"\nDataFrame Shape: {dataset_df.shape}")
    print(f"\nColumns: {list(dataset_df.columns)}")
    print(f"\nSource distribution:")
    print(dataset_df['source'].value_counts())
    print("\n" + "="*80)
    print("ALL PROMPTS:")
    print("="*80)
    print(dataset_df.to_string())
    
    # 2. Save to CSV in dataset/ directory
    output_path = "dataset/prompts.csv"
    dataset_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Dataset saved to: {output_path}")
    print(f"{'='*80}")
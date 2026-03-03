"""
Test Evaluation Pipeline with Ollama Mistral (Subset)

Tests all 3 conditions with 5 prompts to preview CSV output format:
1. Baseline (no defense)
2. SafeDecoding (inference-time safety)
3. Llama Guard 3 (pre/post-filtering)

This test generates CSV files in results/ folder for quick inspection.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strongreject_pipeline.prompt import load_prompts, sample_prompts, add_harmful_labels
from strongreject_pipeline.llm_model import OllamaModelInference
from safe_decoding import create_safe_decoder
from guardrails import create_llamaguard_wrapper
from evaluation_pipeline import run_evaluation_pipeline

# Load environment variables
load_dotenv()


def test_pipeline_mistral_subset():
    """Test evaluation pipeline with Ollama Mistral on 5-prompt subset."""
    
    print("=" * 80)
    print("PIPELINE TEST: OLLAMA MISTRAL - 5 PROMPT SUBSET")
    print("=" * 80)
    print("\nThis test runs the complete pipeline with 3 conditions:")
    print("  1. Baseline (no defense)")
    print("  2. SafeDecoding (safety_weight=0.7)")
    print("  3. Llama Guard 3 (pre/post-filtering)")
    print("\nPrerequisites:")
    print("  - Ollama running: ollama serve")
    print("  - Mistral pulled: ollama pull mistral")
    print("  - Llama Guard 3 pulled: ollama pull llama-guard3")
    print("")
    
    # ========================================================================
    # LOAD DATASET SUBSET
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING DATASET (5 prompts)")
    print("=" * 80)
    
    try:
        # Load full dataset
        prompts_df = load_prompts("dataset/prompts.csv")
        
        # Add harmful labels if not present
        if 'is_harmful' not in prompts_df.columns:
            prompts_df = add_harmful_labels(prompts_df)
        
        # Sample 5 prompts (balanced mix if possible)
        if len(prompts_df) > 5:
            # Try to get balanced sample of harmful/benign
            harmful_df = prompts_df[prompts_df['is_harmful'] == True]
            benign_df = prompts_df[prompts_df['is_harmful'] == False]
            
            n_harmful = min(3, len(harmful_df))
            n_benign = min(2, len(benign_df))
            
            sampled_df = pd.concat([
                harmful_df.sample(n=n_harmful, random_state=42),
                benign_df.sample(n=n_benign, random_state=42)
            ])
            prompts_subset = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        else:
            prompts_subset = prompts_df
        
        print(f"\n✓ Loaded {len(prompts_subset)} prompts")
        print(f"  Sources: {prompts_subset['source'].value_counts().to_dict()}")
        
        if 'is_harmful' in prompts_subset.columns:
            harmful_count = prompts_subset['is_harmful'].sum()
            benign_count = len(prompts_subset) - harmful_count
            print(f"  Harmful: {harmful_count} ({100*harmful_count/len(prompts_subset):.1f}%)")
            print(f"  Benign: {benign_count} ({100*benign_count/len(prompts_subset):.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Store results for summary
    all_results = {}
    
    # ========================================================================
    # TEST 1: BASELINE
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1/3: MISTRAL BASELINE (No Defense)")
    print("=" * 80)
    
    try:
        model = OllamaModelInference(
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        results = run_evaluation_pipeline(
            prompts_df=prompts_subset,
            model_inference=model,
            output_file="results/test_mistral_baseline.csv"
        )
        all_results['baseline'] = results
        
        print(f"\n✅ Baseline complete - {len(results)} responses evaluated")
        
    except Exception as e:
        print(f"\n❌ Error in baseline test: {e}")
        import traceback
        traceback.print_exc()
        all_results['baseline'] = None
    
    # ========================================================================
    # TEST 2: SAFEDECODING
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2/3: MISTRAL + SAFEDECODING (Safety Weight=0.7)")
    print("=" * 80)
    
    try:
        safe_decoder = create_safe_decoder(
            model_type="ollama",
            safety_weight=0.7,
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        results = run_evaluation_pipeline(
            prompts_df=prompts_subset,
            model_inference=safe_decoder,
            output_file="results/test_mistral_safedecoding.csv"
        )
        all_results['safedecoding'] = results
        
        print(f"\n✅ SafeDecoding complete - {len(results)} responses evaluated")
        
    except Exception as e:
        print(f"\n❌ Error in SafeDecoding test: {e}")
        import traceback
        traceback.print_exc()
        all_results['safedecoding'] = None
    
    # ========================================================================
    # TEST 3: LLAMA GUARD 3
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3/3: MISTRAL + LLAMA GUARD 3 (Pre/Post Filtering)")
    print("=" * 80)
    
    try:
        guardrail_wrapper = create_llamaguard_wrapper(
            model_type="ollama",
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        results = run_evaluation_pipeline(
            prompts_df=prompts_subset,
            model_inference=guardrail_wrapper,
            output_file="results/test_mistral_llamaguard3.csv"
        )
        all_results['llamaguard3'] = results
        
        print(f"\n✅ Llama Guard 3 complete - {len(results)} responses evaluated")
        
    except Exception as e:
        print(f"\n❌ Error in Llama Guard 3 test: {e}")
        import traceback
        traceback.print_exc()
        all_results['llamaguard3'] = None
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE TEST COMPLETE - SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for v in all_results.values() if v is not None)
    failed = sum(1 for v in all_results.values() if v is None)
    
    print(f"\nTotal Tests: {len(all_results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    print("\n" + "-" * 80)
    print("CSV FILES GENERATED:")
    print("-" * 80)
    
    test_files = {
        'baseline': 'results/test_mistral_baseline.csv',
        'safedecoding': 'results/test_mistral_safedecoding.csv',
        'llamaguard3': 'results/test_mistral_llamaguard3.csv'
    }
    
    for condition, filepath in test_files.items():
        status = "✅" if all_results.get(condition) is not None else "❌"
        exists = "📄" if os.path.exists(filepath) else "⚠️"
        print(f"  {status} {exists} {condition.ljust(15)} → {filepath}")
    
    # Show sample results if available
    print("\n" + "-" * 80)
    print("SAMPLE RESULTS (First 3 prompts)")
    print("-" * 80)
    
    for condition in ['baseline', 'safedecoding', 'llamaguard3']:
        if all_results.get(condition) is not None:
            print(f"\n{condition.upper()}:")
            df = all_results[condition]
            
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                print(f"\n  [{i+1}] Prompt: {row['prompt'][:60]}...")
                print(f"      Response: {row['response'][:80]}...")
                print(f"      StrongREJECT: {row.get('strongreject_score', 'N/A'):.3f}")
                print(f"      Simple Refusal: {row.get('simple_refusal', 'N/A')}")
                print(f"      Is Harmful: {row.get('is_harmful', 'N/A')}")
    
    # Compare metrics
    print("\n" + "=" * 80)
    print("COMPARATIVE METRICS (Mean Scores)")
    print("=" * 80)
    
    if all(all_results.get(c) is not None for c in ['baseline', 'safedecoding', 'llamaguard3']):
        print(f"\n{'Metric':<25} {'Baseline':<12} {'SafeDecoding':<15} {'LlamaGuard3':<15}")
        print("-" * 80)
        
        # StrongREJECT scores
        baseline_sr = all_results['baseline']['strongreject_score'].mean()
        safe_sr = all_results['safedecoding']['strongreject_score'].mean()
        guard_sr = all_results['llamaguard3']['strongreject_score'].mean()
        print(f"{'StrongREJECT Score':<25} {baseline_sr:<12.3f} {safe_sr:<15.3f} {guard_sr:<15.3f}")
        
        # Refusal rate
        baseline_ref = all_results['baseline']['simple_refusal'].sum() / len(all_results['baseline'])
        safe_ref = all_results['safedecoding']['simple_refusal'].sum() / len(all_results['safedecoding'])
        guard_ref = all_results['llamaguard3']['simple_refusal'].sum() / len(all_results['llamaguard3'])
        print(f"{'Refusal Rate':<25} {baseline_ref:<12.1%} {safe_ref:<15.1%} {guard_ref:<15.1%}")
        
        # Average response length
        baseline_len = all_results['baseline']['response_length'].mean()
        safe_len = all_results['safedecoding']['response_length'].mean()
        guard_len = all_results['llamaguard3']['response_length'].mean()
        print(f"{'Avg Response Length':<25} {baseline_len:<12.1f} {safe_len:<15.1f} {guard_len:<15.1f}")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Open CSV files in results/ folder to inspect full results")
    print("2. Review StrongREJECT scores, refusal patterns, and response quality")
    print("3. Compare defense mechanism effectiveness")
    print("4. If satisfied, run full pipeline on complete dataset")
    print("\n✓ Pipeline test complete!")
    
    return all_results


if __name__ == "__main__":
    print("\n🚀 Starting pipeline test with 5-prompt subset...\n")
    test_pipeline_mistral_subset()

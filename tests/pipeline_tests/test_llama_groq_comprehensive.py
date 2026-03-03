"""
Test Evaluation Pipeline with Llama 3.1 8B (Groq) - Comprehensive Mix

Tests all 3 defense conditions with a balanced 6-prompt dataset:
- 2 harmful prompts (walledeval_diverse)
- 2 benign prompts (xstest_benign)  
- 2 multi-turn prompts (cosafe_multiturn)

Generates two consolidated output files:
1. harmful_benign analysis (4 prompts × 3 conditions = 12 rows)
2. multi-turn degradation analysis (2 prompts × 3 conditions)

Conditions tested:
1. Baseline (no defense)
2. SafeDecoding (inference-time safety)
3. Llama Guard 3 (pre/post-filtering via Ollama)
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strongreject_pipeline.prompt import load_prompts, add_harmful_labels
from strongreject_pipeline.llm_model import GroqModelInference
from safe_decoding import create_safe_decoder
from guardrails import create_llamaguard_wrapper
from evaluation_pipeline import run_evaluation_pipeline
from strongreject_pipeline.multi_turn_degradation import run_multiturn_evaluation_pipeline

# Load environment variables
load_dotenv()


def test_llama_groq_comprehensive():
    """
    Test evaluation pipeline with Llama 3.1 8B (Groq) on balanced 6-prompt mix.
    
    Dataset composition:
    - 2 harmful prompts from walledeval_diverse
    - 2 benign prompts from xstest_benign
    - 2 multi-turn prompts from cosafe_multiturn
    
    Outputs:
    - llama_groq_comprehensive_harmful_benign.csv (4 prompts × 3 conditions)
    - llama_groq_comprehensive_multiturn.csv (2 prompts × 3 conditions)
    """
    
    print("=" * 80)
    print("COMPREHENSIVE PIPELINE TEST: LLAMA 3.1 8B (GROQ)")
    print("=" * 80)
    print("\nDataset: 6 prompts (2 harmful + 2 benign + 2 multi-turn)")
    print("Conditions: 3 (Baseline, SafeDecoding, Llama Guard 3)")
    print("Output: 2 CSV files (harmful/benign + multi-turn)")
    print("\nPrerequisites:")
    print("  - GROQ_API_KEY in .env file")
    print("  - Ollama running (for Llama Guard 3): ollama serve")
    print("  - Llama Guard 3 pulled: ollama pull llama-guard3")
    print("")
    
    # Verify Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("\n❌ Error: GROQ_API_KEY not found in environment variables")
        print("   Please add GROQ_API_KEY=your_key_here to your .env file")
        return None
    else:
        print("✓ GROQ_API_KEY found")
    
    # ========================================================================
    # LOAD AND PREPARE DATASET
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING BALANCED DATASET")
    print("=" * 80)
    
    try:
        # Load full dataset
        full_df = load_prompts("dataset/prompts.csv")
        
        # Add harmful labels if not present
        if 'is_harmful' not in full_df.columns:
            full_df = add_harmful_labels(full_df)
        
        # Select specific prompts by source
        harmful_df = full_df[full_df['source'] == 'walledeval_diverse']
        benign_df = full_df[full_df['source'] == 'xstest_benign']
        multiturn_df = full_df[full_df['source'] == 'cosafe_multiturn']
        
        print(f"\nAvailable prompts per source:")
        print(f"  - walledeval_diverse: {len(harmful_df)} prompts")
        print(f"  - xstest_benign: {len(benign_df)} prompts")
        print(f"  - cosafe_multiturn: {len(multiturn_df)} prompts")
        
        # Sample exactly what we need (using same random_state as other tests for consistency)
        harmful_benign_prompts = pd.concat([
            harmful_df.sample(n=2, random_state=42),     # 2 harmful
            benign_df.sample(n=2, random_state=42)       # 2 benign
        ]).reset_index(drop=True)
        
        multiturn_prompts = multiturn_df.sample(n=2, random_state=42).reset_index(drop=True)
        
        print(f"\n✓ Selected 6 prompts:")
        print(f"  - Harmful/Benign set: 4 prompts (2 harmful + 2 benign)")
        print(f"  - Multi-turn set: 2 prompts")
        
        # Display harmful/benign prompts
        print("\nHarmful/Benign Prompts:")
        for i, row in harmful_benign_prompts.iterrows():
            prompt_preview = row['prompt'][:80] + "..." if len(row['prompt']) > 80 else row['prompt']
            is_harmful = row.get('is_harmful', 'Unknown')
            print(f"  [{i+1}] [{row['source']}] [Harmful: {is_harmful}]")
            print(f"      {prompt_preview}")
        
        # Display multi-turn prompts
        print("\nMulti-turn Prompts:")
        for i, row in multiturn_prompts.iterrows():
            prompt_preview = row['prompt'][:80] + "..." if len(row['prompt']) > 80 else row['prompt']
            print(f"  [{i+1}] [{row['source']}]")
            print(f"      {prompt_preview}")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Store results for consolidation
    harmful_benign_results = []
    multiturn_results = []
    
    # ========================================================================
    # TEST 1: BASELINE
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1/3: LLAMA 3.1 8B BASELINE (No Defense)")
    print("=" * 80)
    
    try:
        model = GroqModelInference(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        print("\n[Baseline] Evaluating harmful/benign prompts...")
        results_hb = run_evaluation_pipeline(
            prompts_df=harmful_benign_prompts,
            model_inference=model
        )
        results_hb['model'] = 'Llama-3.1-8B'
        results_hb['evaluation_type'] = 'baseline'
        results_hb['prompt_type'] = results_hb['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results_hb)
        
        # Evaluate multi-turn prompts
        print("\n[Baseline] Evaluating multi-turn prompts...")
        results_mt = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_prompts,  # Pass only selected multi-turn prompts
            model_inference=model,
            model_name='Llama-3.1-8B',
            condition='baseline'
        )
        print(f"[DEBUG] Baseline multi-turn returned {len(results_mt)} rows")
        print(f"[DEBUG] Columns: {results_mt.columns.tolist() if len(results_mt) > 0 else 'empty'}")
        if len(results_mt) > 0:
            multiturn_results.append(results_mt)
            print(f"[DEBUG] Appended to multiturn_results. Total conditions so far: {len(multiturn_results)}")
        
        print(f"\n✅ Baseline complete")
        
    except Exception as e:
        print(f"\n❌ Error in baseline test: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # TEST 2: SAFEDECODING
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2/3: LLAMA 3.1 8B + SAFEDECODING (Safety Weight=0.7)")
    print("=" * 80)
    
    try:
        safe_decoder = create_safe_decoder(
            model_type="groq",
            safety_weight=0.7,
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        print("\n[SafeDecoding] Evaluating harmful/benign prompts...")
        results_hb = run_evaluation_pipeline(
            prompts_df=harmful_benign_prompts,
            model_inference=safe_decoder
        )
        results_hb['model'] = 'Llama-3.1-8B'
        results_hb['evaluation_type'] = 'safedecoding'
        results_hb['prompt_type'] = results_hb['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results_hb)
        
        # Evaluate multi-turn prompts
        print("\n[SafeDecoding] Evaluating multi-turn prompts...")
        results_mt = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_prompts,  # Pass only selected multi-turn prompts
            model_inference=safe_decoder,
            model_name='Llama-3.1-8B',
            condition='safedecoding'
        )
        print(f"[DEBUG] SafeDecoding multi-turn returned {len(results_mt)} rows")
        if len(results_mt) > 0:
            multiturn_results.append(results_mt)
            print(f"[DEBUG] Appended to multiturn_results. Total conditions so far: {len(multiturn_results)}")
        
        print(f"\n✅ SafeDecoding complete")
        
    except Exception as e:
        print(f"\n❌ Error in SafeDecoding test: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # TEST 3: LLAMA GUARD 3
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3/3: LLAMA 3.1 8B + LLAMA GUARD 3 (Pre/Post Filtering)")
    print("=" * 80)
    
    try:
        guardrail_wrapper = create_llamaguard_wrapper(
            model_type="groq",
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        print("\n[LlamaGuard3] Evaluating harmful/benign prompts...")
        results_hb = run_evaluation_pipeline(
            prompts_df=harmful_benign_prompts,
            model_inference=guardrail_wrapper
        )
        results_hb['model'] = 'Llama-3.1-8B'
        results_hb['evaluation_type'] = 'llamaguard3'
        results_hb['prompt_type'] = results_hb['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results_hb)
        
        # Evaluate multi-turn prompts
        print("\n[LlamaGuard3] Evaluating multi-turn prompts...")
        results_mt = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_prompts,  # Pass only selected multi-turn prompts
            model_inference=guardrail_wrapper,
            model_name='Llama-3.1-8B',
            condition='llamaguard3'
        )
        print(f"[DEBUG] LlamaGuard3 multi-turn returned {len(results_mt)} rows")
        if len(results_mt) > 0:
            multiturn_results.append(results_mt)
            print(f"[DEBUG] Appended to multiturn_results. Total conditions so far: {len(multiturn_results)}")
        
        print(f"\n✅ Llama Guard 3 complete")
        
    except Exception as e:
        print(f"\n❌ Error in Llama Guard 3 test: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # CONSOLIDATE AND SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONSOLIDATING RESULTS")
    print("=" * 80)
    
    output_files = {}
    
    # Consolidate harmful/benign results
    if harmful_benign_results:
        final_harmful_benign = pd.concat(harmful_benign_results, ignore_index=True)
        
        # Reorder columns with metadata first
        column_order = [
            'model', 'evaluation_type', 'prompt_type', 'source', 'prompt', 'is_harmful',
            'response', 'strongreject_score', 'simple_refusal', 'response_length'
        ]
        
        # Add any remaining columns
        remaining_cols = [col for col in final_harmful_benign.columns if col not in column_order]
        final_harmful_benign = final_harmful_benign[column_order + remaining_cols]
        
        output_file_hb = "results/llama_groq_comprehensive_harmful_benign.csv"
        final_harmful_benign.to_csv(output_file_hb, index=False)
        output_files['harmful_benign'] = output_file_hb
        
        print(f"\n✅ Saved harmful/benign analysis: {output_file_hb}")
        print(f"   Total rows: {len(final_harmful_benign)}")
        print(f"   Conditions: {final_harmful_benign['evaluation_type'].unique().tolist()}")
        print(f"   Prompt types: {final_harmful_benign['prompt_type'].unique().tolist()}")
    else:
        print("\n⚠️ No harmful/benign results to consolidate")
    
    # Consolidate multi-turn results
    print(f"\n[DEBUG] Total multi-turn result sets collected: {len(multiturn_results)}")
    if multiturn_results:
        print(f"[DEBUG] Concatenating {len(multiturn_results)} DataFrames...")
        final_multiturn = pd.concat(multiturn_results, ignore_index=True)
        print(f"[DEBUG] Final multi-turn DataFrame has {len(final_multiturn)} rows")
        
        # Reorder columns with metadata first
        column_order_mt = [
            'model', 'evaluation_type', 'prompt_type', 'conversation_id',
            'turn_number', 'turn_prompt', 'turn_response', 'previous_context',
            'cumulative_degradation', 'degradation_from_previous', 'is_jailbreak_attempt',
            'context_length', 'strongreject_score', 'simple_refusal', 'response_length'
        ]
        
        # Add any remaining columns
        remaining_cols_mt = [col for col in final_multiturn.columns if col not in column_order_mt]
        final_multiturn = final_multiturn[column_order_mt + remaining_cols_mt]
        
        output_file_mt = "results/llama_groq_comprehensive_multiturn.csv"
        final_multiturn.to_csv(output_file_mt, index=False)
        output_files['multiturn'] = output_file_mt
        
        print(f"\n✅ Saved multi-turn analysis: {output_file_mt}")
        print(f"   Total rows: {len(final_multiturn)}")
        print(f"   Conversations: {final_multiturn['conversation_id'].nunique()}")
        print(f"   Conditions: {final_multiturn['evaluation_type'].unique().tolist()}")
    else:
        print("\n⚠️ No multi-turn results to consolidate")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if harmful_benign_results:
        final_harmful_benign = pd.concat(harmful_benign_results, ignore_index=True)
        
        print("\n1. HARMFUL/BENIGN ANALYSIS:")
        print("-" * 80)
        
        for condition in ['baseline', 'safedecoding', 'llamaguard3']:
            cond_df = final_harmful_benign[final_harmful_benign['evaluation_type'] == condition]
            if len(cond_df) > 0:
                print(f"\n{condition.upper()}:")
                
                for prompt_type in ['harmful prompts', 'benign prompts']:
                    type_df = cond_df[cond_df['prompt_type'] == prompt_type]
                    if len(type_df) > 0:
                        avg_score = type_df['strongreject_score'].mean()
                        refusal_rate = type_df['simple_refusal'].sum() / len(type_df)
                        avg_length = type_df['response_length'].mean()
                        
                        print(f"  {prompt_type.capitalize()}:")
                        print(f"    Avg StrongREJECT Score: {avg_score:.3f}")
                        print(f"    Refusal Rate: {refusal_rate:.1%} ({type_df['simple_refusal'].sum()}/{len(type_df)})")
                        print(f"    Avg Response Length: {avg_length:.1f} words")
    
    if multiturn_results:
        final_multiturn = pd.concat(multiturn_results, ignore_index=True)
        
        print("\n2. MULTI-TURN DEGRADATION ANALYSIS:")
        print("-" * 80)
        
        for condition in ['baseline', 'safedecoding', 'llamaguard3']:
            cond_df = final_multiturn[final_multiturn['evaluation_type'] == condition]
            if len(cond_df) > 0:
                avg_degradation = cond_df['cumulative_degradation'].mean()
                avg_score = cond_df['strongreject_score'].mean()
                conversations = cond_df['conversation_id'].nunique()
                
                print(f"\n{condition.upper()}:")
                print(f"  Conversations: {conversations}")
                print(f"  Avg Cumulative Degradation: {avg_degradation:.3f}")
                print(f"  Avg StrongREJECT Score: {avg_score:.3f}")
    
    # ========================================================================
    # COMPARATIVE METRICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARATIVE METRICS ACROSS CONDITIONS")
    print("=" * 80)
    
    if harmful_benign_results:
        final_harmful_benign = pd.concat(harmful_benign_results, ignore_index=True)
        
        print(f"\n{'Metric':<30} {'Baseline':<15} {'SafeDecoding':<15} {'LlamaGuard3':<15}")
        print("-" * 80)
        
        # Overall metrics
        baseline_df = final_harmful_benign[final_harmful_benign['evaluation_type'] == 'baseline']
        safe_df = final_harmful_benign[final_harmful_benign['evaluation_type'] == 'safedecoding']
        guard_df = final_harmful_benign[final_harmful_benign['evaluation_type'] == 'llamaguard3']
        
        # StrongREJECT Score
        print(f"{'Overall StrongREJECT Score':<30} {baseline_df['strongreject_score'].mean():<15.3f} {safe_df['strongreject_score'].mean():<15.3f} {guard_df['strongreject_score'].mean():<15.3f}")
        
        # Refusal Rate
        baseline_ref = baseline_df['simple_refusal'].sum() / len(baseline_df) if len(baseline_df) > 0 else 0
        safe_ref = safe_df['simple_refusal'].sum() / len(safe_df) if len(safe_df) > 0 else 0
        guard_ref = guard_df['simple_refusal'].sum() / len(guard_df) if len(guard_df) > 0 else 0
        print(f"{'Refusal Rate':<30} {baseline_ref:<15.1%} {safe_ref:<15.1%} {guard_ref:<15.1%}")
        
        # Average Response Length
        print(f"{'Avg Response Length (words)':<30} {baseline_df['response_length'].mean():<15.1f} {safe_df['response_length'].mean():<15.1f} {guard_df['response_length'].mean():<15.1f}")
        
        print("\n" + "-" * 80)
        print("BY PROMPT TYPE:")
        print("-" * 80)
        
        # Metrics by prompt type
        for prompt_type in ['harmful prompts', 'benign prompts']:
            baseline_type = baseline_df[baseline_df['prompt_type'] == prompt_type]
            safe_type = safe_df[safe_df['prompt_type'] == prompt_type]
            guard_type = guard_df[guard_df['prompt_type'] == prompt_type]
            
            if len(baseline_type) > 0:
                print(f"\n{prompt_type.capitalize()}:")
                baseline_score = baseline_type['strongreject_score'].mean()
                safe_score = safe_type['strongreject_score'].mean()
                guard_score = guard_type['strongreject_score'].mean()
                print(f"  {'StrongREJECT Score':<28} {baseline_score:<15.3f} {safe_score:<15.3f} {guard_score:<15.3f}")
                
                baseline_ref = baseline_type['simple_refusal'].sum() / len(baseline_type)
                safe_ref = safe_type['simple_refusal'].sum() / len(safe_type)
                guard_ref = guard_type['simple_refusal'].sum() / len(guard_type)
                print(f"  {'Refusal Rate':<28} {baseline_ref:<15.1%} {safe_ref:<15.1%} {guard_ref:<15.1%}")
        
        print("-" * 80)
    
    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    if harmful_benign_results:
        final_harmful_benign = pd.concat(harmful_benign_results, ignore_index=True)
        
        baseline_df = final_harmful_benign[final_harmful_benign['evaluation_type'] == 'baseline']
        safe_df = final_harmful_benign[final_harmful_benign['evaluation_type'] == 'safedecoding']
        guard_df = final_harmful_benign[final_harmful_benign['evaluation_type'] == 'llamaguard3']
        
        # Compare harmful prompt handling
        harmful_baseline = baseline_df[baseline_df['prompt_type'] == 'harmful prompts']['strongreject_score'].mean()
        harmful_safe = safe_df[safe_df['prompt_type'] == 'harmful prompts']['strongreject_score'].mean()
        harmful_guard = guard_df[guard_df['prompt_type'] == 'harmful prompts']['strongreject_score'].mean()
        
        print("\n1. Harmful Prompt Defense Effectiveness:")
        print(f"   - Baseline: {harmful_baseline:.3f}")
        print(f"   - SafeDecoding: {harmful_safe:.3f} (Δ: {harmful_safe - harmful_baseline:+.3f})")
        print(f"   - LlamaGuard3: {harmful_guard:.3f} (Δ: {harmful_guard - harmful_baseline:+.3f})")
        
        # Compare benign prompt handling
        benign_baseline = baseline_df[baseline_df['prompt_type'] == 'benign prompts']['strongreject_score'].mean()
        benign_safe = safe_df[safe_df['prompt_type'] == 'benign prompts']['strongreject_score'].mean()
        benign_guard = guard_df[guard_df['prompt_type'] == 'benign prompts']['strongreject_score'].mean()
        
        print("\n2. Benign Prompt Over-filtering Risk:")
        print(f"   - Baseline: {benign_baseline:.3f}")
        print(f"   - SafeDecoding: {benign_safe:.3f} (Δ: {benign_safe - benign_baseline:+.3f})")
        print(f"   - LlamaGuard3: {benign_guard:.3f} (Δ: {benign_guard - benign_baseline:+.3f})")
        
        # Best defense
        best_defense = min(
            [('Baseline', harmful_baseline), ('SafeDecoding', harmful_safe), ('LlamaGuard3', harmful_guard)],
            key=lambda x: x[1]
        )
        print(f"\n3. Best Defense for Harmful Prompts: {best_defense[0]} (Score: {best_defense[1]:.3f})")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Review generated CSV files:")
    for file_type, filepath in output_files.items():
        print(f"   - {filepath}")
    print("\n2. Analysis recommendations:")
    print("   - Compare defense effectiveness on harmful vs benign prompts")
    print("   - Analyze multi-turn degradation patterns across conditions")
    print("   - Identify over-filtering (false positives on benign prompts)")
    print("   - Evaluate response quality and refusal rates")
    print("\n3. Compare with other models on same prompts:")
    print("   - Run test_gemini_comprehensive.py")
    print("   - Run test_mistral_comprehensive.py")
    print("   - Compare Llama 3.1 8B vs Gemini 2.5 Flash vs Mistral 7B performance")
    print("\n✓ Comprehensive pipeline test complete!")
    
    return output_files


if __name__ == "__main__":
    print("\n🚀 Starting comprehensive pipeline test with Llama 3.1 8B (Groq)...\n")
    output_files = test_llama_groq_comprehensive()
    
    if output_files:
        print("\n✅ Test execution successful!")
        print(f"   Generated files:")
        for file_type, filepath in output_files.items():
            print(f"   - {file_type}: {filepath}")
    else:
        print("\n❌ Test execution failed - check error messages above")

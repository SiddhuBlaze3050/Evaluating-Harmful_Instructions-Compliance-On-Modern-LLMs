"""
Test Multi-Turn Evaluation Pipeline with Groq Llama 3.1

Tests all 3 defense conditions on multi-turn conversations only:
- Baseline (no defense)
- SafeDecoding (inference-time safety)
- Llama Guard 3 (pre/post-filtering)

Uses Groq API with Llama 3.1 8B for fast cloud inference.

Output:
- llama_groq_multiturn.csv (all conversations × 3 conditions)
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
import gc
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strongreject_pipeline.prompt import load_prompts, add_harmful_labels
from strongreject_pipeline.llm_model import GroqModelInference
from safe_decoding import create_safe_decoder
from guardrails import create_llamaguard_wrapper
from strongreject_pipeline.multi_turn_degradation import run_multiturn_evaluation_pipeline

# Load environment variables
load_dotenv()


def free_memory():
    """Free up memory between evaluations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[MEMORY] Cleared cache")


def test_llama_groq_multiturn():
    """
    Test multi-turn evaluation pipeline with Groq Llama 3.1.
    
    Dataset composition:
    - 4 multi-turn conversations from cosafe_multiturn
    
    Outputs:
    - llama_groq_multiturn.csv (~12+ turns × 3 conditions)
    """
    
    print("=" * 80)
    print("MULTI-TURN PIPELINE TEST: GROQ LLAMA 3.1")
    print("=" * 80)
    print("\nDataset: 4 multi-turn conversations (cosafe_multiturn)")
    print("Conditions: 3 (Baseline, SafeDecoding, Llama Guard 3)")
    print("Output: 1 CSV file (multi-turn degradation analysis)")
    print("\nPrerequisites:")
    print("  - GROQ_API_KEY set in .env file")
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
    print("LOADING MULTI-TURN DATASET")
    print("=" * 80)
    
    try:
        # Load full dataset
        full_df = load_prompts("dataset/prompts.csv")
        
        # Add harmful labels if not present
        if 'is_harmful' not in full_df.columns:
            full_df = add_harmful_labels(full_df)
        
        # Select multi-turn conversations
        multiturn_df = full_df[full_df['source'] == 'cosafe_multiturn']
        
        print(f"\nAvailable multi-turn conversations: {len(multiturn_df)}")
        
        # Sample 4 conversations for testing
        multiturn_prompts = multiturn_df.sample(n=4, random_state=42).reset_index(drop=True)
        
        print(f"✓ Selected 4 multi-turn conversations")
        
        # Display selected conversations
        print("\nSelected Multi-Turn Conversations:")
        for i, row in multiturn_prompts.iterrows():
            # Count turns in conversation
            turns = row['prompt'].count('User:')
            prompt_preview = row['prompt'][:100] + "..." if len(row['prompt']) > 100 else row['prompt']
            print(f"  [{i+1}] [{row['source']}] ({turns} turns)")
            print(f"      {prompt_preview}")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Store results for consolidation
    multiturn_results = []
    
    # ========================================================================
    # TEST 1: BASELINE
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1/3: LLAMA 3.1 BASELINE (No Defense)")
    print("=" * 80)
    
    try:
        model = GroqModelInference(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        print("\n[Baseline] Evaluating multi-turn conversations...")
        results_mt = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_prompts,
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
        
        # Free memory
        del model
        free_memory()
        
    except Exception as e:
        print(f"\n❌ Error in baseline test: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # TEST 2: SAFEDECODING
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2/3: LLAMA 3.1 + SAFEDECODING (Safety Weight=0.7)")
    print("=" * 80)
    
    try:
        safe_decoder = create_safe_decoder(
            model_type="groq",
            safety_weight=0.7,
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        print("\n[SafeDecoding] Evaluating multi-turn conversations...")
        results_mt = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_prompts,
            model_inference=safe_decoder,
            model_name='Llama-3.1-8B',
            condition='safedecoding'
        )
        print(f"[DEBUG] SafeDecoding multi-turn returned {len(results_mt)} rows")
        if len(results_mt) > 0:
            multiturn_results.append(results_mt)
            print(f"[DEBUG] Appended to multiturn_results. Total conditions so far: {len(multiturn_results)}")
        
        print(f"\n✅ SafeDecoding complete")
        
        # Free memory
        del safe_decoder
        free_memory()
        
    except Exception as e:
        print(f"\n❌ Error in SafeDecoding test: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # TEST 3: LLAMA GUARD 3
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3/3: LLAMA 3.1 + LLAMA GUARD 3 (Pre/Post Filtering)")
    print("=" * 80)
    
    try:
        guardrail_wrapper = create_llamaguard_wrapper(
            model_type="groq",
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        print("\n[LlamaGuard3] Evaluating multi-turn conversations...")
        results_mt = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_prompts,
            model_inference=guardrail_wrapper,
            model_name='Llama-3.1-8B',
            condition='llamaguard3'
        )
        print(f"[DEBUG] LlamaGuard3 multi-turn returned {len(results_mt)} rows")
        if len(results_mt) > 0:
            multiturn_results.append(results_mt)
            print(f"[DEBUG] Appended to multiturn_results. Total conditions so far: {len(multiturn_results)}")
        
        print(f"\n✅ Llama Guard 3 complete")
        
        # Free memory
        del guardrail_wrapper
        free_memory()
        
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
        
        output_file_mt = "results/llama_groq_multiturn.csv"
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
    
    if multiturn_results:
        final_multiturn = pd.concat(multiturn_results, ignore_index=True)
        
        print("\nMULTI-TURN DEGRADATION ANALYSIS:")
        print("-" * 80)
        
        for condition in ['baseline', 'safedecoding', 'llamaguard3']:
            cond_df = final_multiturn[final_multiturn['evaluation_type'] == condition]
            if len(cond_df) > 0:
                avg_degradation = cond_df['cumulative_degradation'].mean()
                avg_score = cond_df['strongreject_score'].mean()
                conversations = cond_df['conversation_id'].nunique()
                total_turns = len(cond_df)
                refusal_rate = cond_df['simple_refusal'].sum() / len(cond_df)
                avg_length = cond_df['response_length'].mean()
                
                print(f"\n{condition.upper()}:")
                print(f"  Conversations: {conversations}")
                print(f"  Total Turns: {total_turns}")
                print(f"  Avg Cumulative Degradation: {avg_degradation:.3f}")
                print(f"  Avg StrongREJECT Score: {avg_score:.3f}")
                print(f"  Refusal Rate: {refusal_rate:.1%}")
                print(f"  Avg Response Length: {avg_length:.1f} words")
    
    # ========================================================================
    # COMPARATIVE METRICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARATIVE METRICS ACROSS CONDITIONS")
    print("=" * 80)
    
    if multiturn_results:
        final_multiturn = pd.concat(multiturn_results, ignore_index=True)
        
        print(f"\n{'Metric':<35} {'Baseline':<15} {'SafeDecoding':<15} {'LlamaGuard3':<15}")
        print("-" * 80)
        
        baseline_df = final_multiturn[final_multiturn['evaluation_type'] == 'baseline']
        safe_df = final_multiturn[final_multiturn['evaluation_type'] == 'safedecoding']
        guard_df = final_multiturn[final_multiturn['evaluation_type'] == 'llamaguard3']
        
        # Cumulative Degradation
        print(f"{'Avg Cumulative Degradation':<35} {baseline_df['cumulative_degradation'].mean():<15.3f} {safe_df['cumulative_degradation'].mean():<15.3f} {guard_df['cumulative_degradation'].mean():<15.3f}")
        
        # StrongREJECT Score
        print(f"{'Avg StrongREJECT Score':<35} {baseline_df['strongreject_score'].mean():<15.3f} {safe_df['strongreject_score'].mean():<15.3f} {guard_df['strongreject_score'].mean():<15.3f}")
        
        # Refusal Rate
        baseline_ref = baseline_df['simple_refusal'].sum() / len(baseline_df) if len(baseline_df) > 0 else 0
        safe_ref = safe_df['simple_refusal'].sum() / len(safe_df) if len(safe_df) > 0 else 0
        guard_ref = guard_df['simple_refusal'].sum() / len(guard_df) if len(guard_df) > 0 else 0
        print(f"{'Refusal Rate':<35} {baseline_ref:<15.1%} {safe_ref:<15.1%} {guard_ref:<15.1%}")
        
        # Average Response Length
        print(f"{'Avg Response Length (words)':<35} {baseline_df['response_length'].mean():<15.1f} {safe_df['response_length'].mean():<15.1f} {guard_df['response_length'].mean():<15.1f}")
        
        print("\n" + "-" * 80)
        
        # Turn-by-turn analysis
        print("\nDEGRADATION BY TURN NUMBER:")
        print("-" * 80)
        
        for condition in ['baseline', 'safedecoding', 'llamaguard3']:
            cond_df = final_multiturn[final_multiturn['evaluation_type'] == condition]
            if len(cond_df) > 0:
                turn_stats = cond_df.groupby('turn_number').agg({
                    'cumulative_degradation': 'mean',
                    'strongreject_score': 'mean'
                })
                
                print(f"\n{condition.upper()}:")
                for turn_num, row in turn_stats.iterrows():
                    print(f"  Turn {turn_num}: Degradation={row['cumulative_degradation']:+.3f}, Score={row['strongreject_score']:.3f}")
        
        print("-" * 80)
    
    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    if multiturn_results:
        final_multiturn = pd.concat(multiturn_results, ignore_index=True)
        
        baseline_df = final_multiturn[final_multiturn['evaluation_type'] == 'baseline']
        safe_df = final_multiturn[final_multiturn['evaluation_type'] == 'safedecoding']
        guard_df = final_multiturn[final_multiturn['evaluation_type'] == 'llamaguard3']
        
        # Compare degradation control
        baseline_deg = baseline_df['cumulative_degradation'].mean()
        safe_deg = safe_df['cumulative_degradation'].mean()
        guard_deg = guard_df['cumulative_degradation'].mean()
        
        print("\n1. Multi-Turn Degradation Control:")
        print(f"   - Baseline: {baseline_deg:+.3f}")
        print(f"   - SafeDecoding: {safe_deg:+.3f} (Δ: {safe_deg - baseline_deg:+.3f})")
        print(f"   - LlamaGuard3: {guard_deg:+.3f} (Δ: {guard_deg - baseline_deg:+.3f})")
        
        # Compare safety scores
        baseline_score = baseline_df['strongreject_score'].mean()
        safe_score = safe_df['strongreject_score'].mean()
        guard_score = guard_df['strongreject_score'].mean()
        
        print("\n2. Average Safety Score Across Turns:")
        print(f"   - Baseline: {baseline_score:.3f}")
        print(f"   - SafeDecoding: {safe_score:.3f} (Δ: {safe_score - baseline_score:+.3f})")
        print(f"   - LlamaGuard3: {guard_score:.3f} (Δ: {guard_score - baseline_score:+.3f})")
        
        # Best defense
        best_defense = min(
            [('Baseline', baseline_deg), ('SafeDecoding', safe_deg), ('LlamaGuard3', guard_deg)],
            key=lambda x: abs(x[1])  # Closest to 0 is best (no degradation)
        )
        print(f"\n3. Best Defense for Multi-Turn Degradation: {best_defense[0]} (Degradation: {best_defense[1]:+.3f})")
        
        # Jailbreak detection
        if 'is_jailbreak_attempt' in final_multiturn.columns:
            baseline_jb = baseline_df['is_jailbreak_attempt'].sum()
            safe_jb = safe_df['is_jailbreak_attempt'].sum()
            guard_jb = guard_df['is_jailbreak_attempt'].sum()
            
            print(f"\n4. Jailbreak Attempts Detected:")
            print(f"   - Baseline: {baseline_jb} attempts")
            print(f"   - SafeDecoding: {safe_jb} attempts")
            print(f"   - LlamaGuard3: {guard_jb} attempts")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Review generated CSV file:")
    for file_type, filepath in output_files.items():
        print(f"   - {filepath}")
    print("\n2. Analysis recommendations:")
    print("   - Analyze multi-turn degradation patterns across conditions")
    print("   - Compare how each defense handles conversation context")
    print("   - Identify turns where jailbreak attempts succeed/fail")
    print("   - Evaluate trade-offs between safety and response quality")
    print("\n3. Compare with Mistral results on same conversations")
    print("   - Run test_mistral_comprehensive.py with same prompts")
    print("   - Compare Groq Llama 3.1 vs Ollama Mistral 7B performance")
    print("\n✓ Multi-turn pipeline test complete!")
    
    return output_files


if __name__ == "__main__":
    print("\n🚀 Starting multi-turn pipeline test with Groq Llama 3.1...\n")
    output_files = test_llama_groq_multiturn()
    
    if output_files:
        print("\n✅ Test execution successful!")
        print(f"   Generated files:")
        for file_type, filepath in output_files.items():
            print(f"   - {file_type}: {filepath}")
    else:
        print("\n❌ Test execution failed - check error messages above")

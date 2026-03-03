"""
Test Full Evaluation Pipeline - 10% Sample (All Models × All Conditions)

Comprehensive validation test before running the full pipeline:
- Tests ALL 3 models: Gemini 2.5 Flash, Llama 3.1 (Groq), Mistral 7B (Ollama)
- Tests ALL 3 conditions: Baseline, SafeDecoding, Llama Guard 3
- Uses 10% sample: ~20 harmful/benign + ~6 multi-turn prompts
- Includes memory cleanup between models to prevent OOM
- Estimates total runtime for full pipeline

Output:
- sample_harmful_benign_analysis.csv (3 models × 3 conditions × ~20 prompts)
- sample_multi_turn_analysis.csv (3 models × 3 conditions × ~6 conversations)

Prerequisites:
- GEMINI_API_KEY in .env
- GROQ_API_KEY in .env
- Ollama running: ollama serve
- Models pulled: ollama pull mistral && ollama pull llama-guard3
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
import gc
import torch
import time
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from strongreject_pipeline.prompt import load_prompts, add_harmful_labels
from strongreject_pipeline.llm_model import (
    GeminiModelInference,
    GroqModelInference,
    OllamaModelInference
)
from safe_decoding import create_safe_decoder
from guardrails import create_llamaguard_wrapper
from evaluation_pipeline import run_evaluation_pipeline
from strongreject_pipeline.multi_turn_degradation import run_multiturn_evaluation_pipeline

# Load environment variables
load_dotenv()


def free_memory():
    """Free up memory between model evaluations to prevent OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[MEMORY] Cleared cache")


def test_full_pipeline_sample():
    """
    Test complete evaluation pipeline with 10% sample across all models and conditions.
    
    Dataset composition (10% sample):
    - ~20 harmful/benign prompts (walledeval_diverse + xstest_benign)
    - ~6 multi-turn conversations (cosafe_multiturn)
    
    Models × Conditions:
    - Gemini 2.5 Flash: Baseline, SafeDecoding, Llama Guard 3
    - Llama 3.1 (Groq): Baseline, SafeDecoding, Llama Guard 3
    - Mistral 7B (Ollama): Baseline, SafeDecoding, Llama Guard 3
    
    Outputs:
    - sample_harmful_benign_analysis.csv (~180 rows: 3 models × 3 conditions × 20 prompts)
    - sample_multi_turn_analysis.csv (~540 rows: 3 models × 3 conditions × 6 conversations × 3 turns avg)
    """
    
    start_time = time.time()
    
    print("=" * 80)
    print("FULL PIPELINE TEST: 10% SAMPLE (ALL MODELS × ALL CONDITIONS)")
    print("=" * 80)
    print("\nDataset: 10% sample (~26 total prompts)")
    print("  - ~20 harmful/benign prompts")
    print("  - ~6 multi-turn conversations")
    print("\nModels: 3 (Gemini, Llama 3.1, Mistral)")
    print("Conditions per model: 3 (Baseline, SafeDecoding, Llama Guard 3)")
    print("Total evaluations: 9 (3 models × 3 conditions)")
    print("\nOutput: 2 consolidated CSV files")
    print("\nPrerequisites:")
    print("  - GEMINI_API_KEY in .env")
    print("  - GROQ_API_KEY in .env")
    print("  - Ollama running: ollama serve")
    print("  - Models pulled: ollama pull mistral && ollama pull llama-guard3")
    print("")
    
    # Verify API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not gemini_key:
        print("\n⚠️ Warning: GEMINI_API_KEY not found - Gemini tests will fail")
    else:
        print("✓ GEMINI_API_KEY found")
    
    if not groq_key:
        print("⚠️ Warning: GROQ_API_KEY not found - Groq tests will fail")
    else:
        print("✓ GROQ_API_KEY found")
    
    # ========================================================================
    # LOAD AND PREPARE 10% SAMPLE DATASET
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING 10% SAMPLE DATASET")
    print("=" * 80)
    
    try:
        # Load full dataset
        full_df = load_prompts("dataset/prompts.csv")
        
        # Add harmful labels if not present
        if 'is_harmful' not in full_df.columns:
            full_df = add_harmful_labels(full_df)
        
        # Separate by source
        harmful_df = full_df[full_df['source'] == 'walledeval_diverse']
        benign_df = full_df[full_df['source'] == 'xstest_benign']
        multiturn_df = full_df[full_df['source'] == 'cosafe_multiturn']
        
        print(f"\nFull dataset sizes:")
        print(f"  - walledeval_diverse: {len(harmful_df)} prompts")
        print(f"  - xstest_benign: {len(benign_df)} prompts")
        print(f"  - cosafe_multiturn: {len(multiturn_df)} prompts")
        print(f"  - Total: {len(full_df)} prompts")
        
        # Sample 10% from each source
        sample_size_harmful = max(10, int(len(harmful_df) * 0.1))  # At least 10
        sample_size_benign = max(10, int(len(benign_df) * 0.1))   # At least 10
        sample_size_multiturn = max(5, int(len(multiturn_df) * 0.1))  # At least 5
        
        harmful_sample = harmful_df.sample(n=sample_size_harmful, random_state=42)
        benign_sample = benign_df.sample(n=sample_size_benign, random_state=42)
        multiturn_sample = multiturn_df.sample(n=sample_size_multiturn, random_state=42)
        
        # Combine harmful and benign for evaluation
        harmful_benign_sample = pd.concat([harmful_sample, benign_sample]).reset_index(drop=True)
        
        print(f"\n✓ 10% Sample sizes:")
        print(f"  - Harmful: {len(harmful_sample)} prompts")
        print(f"  - Benign: {len(benign_sample)} prompts")
        print(f"  - Multi-turn: {len(multiturn_sample)} conversations")
        print(f"  - Total harmful/benign: {len(harmful_benign_sample)} prompts")
        
        # Estimate full pipeline size
        estimated_harmful_benign_rows = len(harmful_benign_sample) * 3 * 3  # 3 models × 3 conditions
        estimated_multiturn_rows = len(multiturn_sample) * 3 * 3 * 3  # 3 models × 3 conditions × ~3 turns avg
        
        print(f"\nEstimated output sizes:")
        print(f"  - Harmful/benign rows: ~{estimated_harmful_benign_rows}")
        print(f"  - Multi-turn rows: ~{estimated_multiturn_rows}")
        
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
    
    # Track timing for each model
    model_timings = {}
    
    # ========================================================================
    # GEMINI 2.5 FLASH EVALUATIONS
    # ========================================================================
    if gemini_key:
        model_name = "Gemini-2.5 Flash"
        model_start_time = time.time()
        
        print("\n" + "=" * 80)
        print(f"TESTING {model_name.upper()}")
        print("=" * 80)
        
        # --- BASELINE ---
        print("\n" + "-" * 80)
        print(f"{model_name} - BASELINE")
        print("-" * 80)
        try:
            model = GeminiModelInference(
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=500
            )
            
            # Harmful/benign
            results = run_evaluation_pipeline(
                prompts_df=harmful_benign_sample,
                model_inference=model,
                output_file="results/temp_sample_gemini_baseline.csv"
            )
            results['model'] = model_name
            results['evaluation_type'] = 'baseline'
            results['prompt_type'] = results['source'].map({
                'walledeval_diverse': 'harmful prompts',
                'xstest_benign': 'benign prompts'
            })
            harmful_benign_results.append(results)
            
            # Multi-turn
            multiturn_df = run_multiturn_evaluation_pipeline(
                prompts_df=multiturn_sample,
                model_inference=model,
                model_name=model_name,
                condition='baseline'
            )
            if len(multiturn_df) > 0:
                multiturn_results.append(multiturn_df)
            
            print(f"✅ Baseline complete")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # --- SAFEDECODING ---
        print("\n" + "-" * 80)
        print(f"{model_name} - SAFEDECODING")
        print("-" * 80)
        try:
            safe_decoder = create_safe_decoder(
                model_type="gemini",
                safety_weight=0.7,
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=500
            )
            
            # Harmful/benign
            results = run_evaluation_pipeline(
                prompts_df=harmful_benign_sample,
                model_inference=safe_decoder,
                output_file="results/temp_sample_gemini_safedecoding.csv"
            )
            results['model'] = model_name
            results['evaluation_type'] = 'safedecoding'
            results['prompt_type'] = results['source'].map({
                'walledeval_diverse': 'harmful prompts',
                'xstest_benign': 'benign prompts'
            })
            harmful_benign_results.append(results)
            
            # Multi-turn
            multiturn_df = run_multiturn_evaluation_pipeline(
                prompts_df=multiturn_sample,
                model_inference=safe_decoder,
                model_name=model_name,
                condition='safedecoding'
            )
            if len(multiturn_df) > 0:
                multiturn_results.append(multiturn_df)
            
            print(f"✅ SafeDecoding complete")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # --- LLAMA GUARD 3 ---
        print("\n" + "-" * 80)
        print(f"{model_name} - LLAMA GUARD 3")
        print("-" * 80)
        try:
            guardrail_wrapper = create_llamaguard_wrapper(
                model_type="gemini",
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=500
            )
            
            # Harmful/benign
            results = run_evaluation_pipeline(
                prompts_df=harmful_benign_sample,
                model_inference=guardrail_wrapper,
                output_file="results/temp_sample_gemini_llamaguard3.csv"
            )
            results['model'] = model_name
            results['evaluation_type'] = 'llamaguard3'
            results['prompt_type'] = results['source'].map({
                'walledeval_diverse': 'harmful prompts',
                'xstest_benign': 'benign prompts'
            })
            harmful_benign_results.append(results)
            
            # Multi-turn
            multiturn_df = run_multiturn_evaluation_pipeline(
                prompts_df=multiturn_sample,
                model_inference=guardrail_wrapper,
                model_name=model_name,
                condition='llamaguard3'
            )
            if len(multiturn_df) > 0:
                multiturn_results.append(multiturn_df)
            
            print(f"✅ Llama Guard 3 complete")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Free memory after Gemini
        print("\n" + "-" * 80)
        print(f"MEMORY CLEANUP: {model_name} COMPLETE")
        print("-" * 80)
        try:
            del model
        except NameError:
            pass
        try:
            del safe_decoder
        except NameError:
            pass
        try:
            del guardrail_wrapper
        except NameError:
            pass
        free_memory()
        
        model_elapsed = time.time() - model_start_time
        model_timings[model_name] = model_elapsed
        print(f"⏱️ {model_name} total time: {model_elapsed/60:.1f} minutes")
    
    # ========================================================================
    # LLAMA 3.1 (GROQ) EVALUATIONS
    # ========================================================================
    if groq_key:
        model_name = "Llama-3.1-8B"
        model_start_time = time.time()
        
        print("\n" + "=" * 80)
        print(f"TESTING {model_name.upper()} (GROQ)")
        print("=" * 80)
        
        # --- BASELINE ---
        print("\n" + "-" * 80)
        print(f"{model_name} - BASELINE")
        print("-" * 80)
        try:
            model = GroqModelInference(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=500
            )
            
            # Harmful/benign
            results = run_evaluation_pipeline(
                prompts_df=harmful_benign_sample,
                model_inference=model,
                output_file="results/temp_sample_llama31_baseline.csv"
            )
            results['model'] = model_name
            results['evaluation_type'] = 'baseline'
            results['prompt_type'] = results['source'].map({
                'walledeval_diverse': 'harmful prompts',
                'xstest_benign': 'benign prompts'
            })
            harmful_benign_results.append(results)
            
            # Multi-turn
            multiturn_df = run_multiturn_evaluation_pipeline(
                prompts_df=multiturn_sample,
                model_inference=model,
                model_name=model_name,
                condition='baseline'
            )
            if len(multiturn_df) > 0:
                multiturn_results.append(multiturn_df)
            
            print(f"✅ Baseline complete")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # --- SAFEDECODING ---
        print("\n" + "-" * 80)
        print(f"{model_name} - SAFEDECODING")
        print("-" * 80)
        try:
            safe_decoder = create_safe_decoder(
                model_type="groq",
                safety_weight=0.7,
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=500
            )
            
            # Harmful/benign
            results = run_evaluation_pipeline(
                prompts_df=harmful_benign_sample,
                model_inference=safe_decoder,
                output_file="results/temp_sample_llama31_safedecoding.csv"
            )
            results['model'] = model_name
            results['evaluation_type'] = 'safedecoding'
            results['prompt_type'] = results['source'].map({
                'walledeval_diverse': 'harmful prompts',
                'xstest_benign': 'benign prompts'
            })
            harmful_benign_results.append(results)
            
            # Multi-turn
            multiturn_df = run_multiturn_evaluation_pipeline(
                prompts_df=multiturn_sample,
                model_inference=safe_decoder,
                model_name=model_name,
                condition='safedecoding'
            )
            if len(multiturn_df) > 0:
                multiturn_results.append(multiturn_df)
            
            print(f"✅ SafeDecoding complete")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # --- LLAMA GUARD 3 ---
        print("\n" + "-" * 80)
        print(f"{model_name} - LLAMA GUARD 3")
        print("-" * 80)
        try:
            guardrail_wrapper = create_llamaguard_wrapper(
                model_type="groq",
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=500
            )
            
            # Harmful/benign
            results = run_evaluation_pipeline(
                prompts_df=harmful_benign_sample,
                model_inference=guardrail_wrapper,
                output_file="results/temp_sample_llama31_llamaguard3.csv"
            )
            results['model'] = model_name
            results['evaluation_type'] = 'llamaguard3'
            results['prompt_type'] = results['source'].map({
                'walledeval_diverse': 'harmful prompts',
                'xstest_benign': 'benign prompts'
            })
            harmful_benign_results.append(results)
            
            # Multi-turn
            multiturn_df = run_multiturn_evaluation_pipeline(
                prompts_df=multiturn_sample,
                model_inference=guardrail_wrapper,
                model_name=model_name,
                condition='llamaguard3'
            )
            if len(multiturn_df) > 0:
                multiturn_results.append(multiturn_df)
            
            print(f"✅ Llama Guard 3 complete")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Free memory after Llama 3.1
        print("\n" + "-" * 80)
        print(f"MEMORY CLEANUP: {model_name} COMPLETE")
        print("-" * 80)
        try:
            del model
        except NameError:
            pass
        try:
            del safe_decoder
        except NameError:
            pass
        try:
            del guardrail_wrapper
        except NameError:
            pass
        free_memory()
        
        model_elapsed = time.time() - model_start_time
        model_timings[model_name] = model_elapsed
        print(f"⏱️ {model_name} total time: {model_elapsed/60:.1f} minutes")
    
    # ========================================================================
    # MISTRAL 7B (OLLAMA) EVALUATIONS
    # ========================================================================
    model_name = "Mistral-7B"
    model_start_time = time.time()
    
    print("\n" + "=" * 80)
    print(f"TESTING {model_name.upper()} (OLLAMA)")
    print("=" * 80)
    
    # --- BASELINE ---
    print("\n" + "-" * 80)
    print(f"{model_name} - BASELINE")
    print("-" * 80)
    try:
        model = OllamaModelInference(
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        # Harmful/benign
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_sample,
            model_inference=model,
            output_file="results/temp_sample_mistral_baseline.csv"
        )
        results['model'] = model_name
        results['evaluation_type'] = 'baseline'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_sample,
            model_inference=model,
            model_name=model_name,
            condition='baseline'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
        print(f"✅ Baseline complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # --- SAFEDECODING ---
    print("\n" + "-" * 80)
    print(f"{model_name} - SAFEDECODING")
    print("-" * 80)
    try:
        safe_decoder = create_safe_decoder(
            model_type="ollama",
            safety_weight=0.7,
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        # Harmful/benign
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_sample,
            model_inference=safe_decoder,
            output_file="results/temp_sample_mistral_safedecoding.csv"
        )
        results['model'] = model_name
        results['evaluation_type'] = 'safedecoding'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_sample,
            model_inference=safe_decoder,
            model_name=model_name,
            condition='safedecoding'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
        print(f"✅ SafeDecoding complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # --- LLAMA GUARD 3 ---
    print("\n" + "-" * 80)
    print(f"{model_name} - LLAMA GUARD 3")
    print("-" * 80)
    try:
        guardrail_wrapper = create_llamaguard_wrapper(
            model_type="ollama",
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        # Harmful/benign
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_sample,
            model_inference=guardrail_wrapper,
            output_file="results/temp_sample_mistral_llamaguard3.csv"
        )
        results['model'] = model_name
        results['evaluation_type'] = 'llamaguard3'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=multiturn_sample,
            model_inference=guardrail_wrapper,
            model_name=model_name,
            condition='llamaguard3'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
        print(f"✅ Llama Guard 3 complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Free memory after Mistral
    print("\n" + "-" * 80)
    print(f"MEMORY CLEANUP: {model_name} COMPLETE")
    print("-" * 80)
    try:
        del model
    except NameError:
        pass
    try:
        del safe_decoder
    except NameError:
        pass
    try:
        del guardrail_wrapper
    except NameError:
        pass
    free_memory()
    
    model_elapsed = time.time() - model_start_time
    model_timings[model_name] = model_elapsed
    print(f"⏱️ {model_name} total time: {model_elapsed/60:.1f} minutes")
    
    # ========================================================================
    # CONSOLIDATE AND SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONSOLIDATING RESULTS")
    print("=" * 80)
    
    output_files = {}
    
    # Merge harmful/benign results
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
        
        output_file_hb = "results/sample_harmful_benign_analysis.csv"
        final_harmful_benign.to_csv(output_file_hb, index=False)
        output_files['harmful_benign'] = output_file_hb
        
        print(f"\n✅ Saved harmful/benign analysis: {output_file_hb}")
        print(f"   Total rows: {len(final_harmful_benign)}")
        print(f"   Models: {final_harmful_benign['model'].unique().tolist()}")
        print(f"   Conditions: {final_harmful_benign['evaluation_type'].unique().tolist()}")
        print(f"   Prompt types: {final_harmful_benign['prompt_type'].unique().tolist()}")
    else:
        print("\n⚠️ No harmful/benign results to consolidate")
    
    # Merge multi-turn results
    if multiturn_results:
        final_multiturn = pd.concat(multiturn_results, ignore_index=True)
        
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
        
        output_file_mt = "results/sample_multi_turn_analysis.csv"
        final_multiturn.to_csv(output_file_mt, index=False)
        output_files['multiturn'] = output_file_mt
        
        print(f"\n✅ Saved multi-turn analysis: {output_file_mt}")
        print(f"   Total rows: {len(final_multiturn)}")
        print(f"   Conversations: {final_multiturn['conversation_id'].nunique()}")
        print(f"   Models: {final_multiturn['model'].unique().tolist()}")
        print(f"   Conditions: {final_multiturn['evaluation_type'].unique().tolist()}")
    else:
        print("\n⚠️ No multi-turn results to consolidate")
    
    # ========================================================================
    # TIMING ANALYSIS & FULL PIPELINE ESTIMATE
    # ========================================================================
    total_elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("TIMING ANALYSIS & FULL PIPELINE ESTIMATE")
    print("=" * 80)
    
    print(f"\n⏱️ Sample Test Timing (10% dataset):")
    print(f"   Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    
    if model_timings:
        print(f"\n   Per-Model Timing:")
        for model, elapsed in model_timings.items():
            print(f"   - {model}: {elapsed/60:.1f} minutes")
        
        avg_model_time = sum(model_timings.values()) / len(model_timings)
        print(f"\n   Average per model: {avg_model_time/60:.1f} minutes")
    
    # Estimate full pipeline
    print(f"\n📊 Full Pipeline Estimate (100% dataset):")
    estimated_total_time = total_elapsed * 10  # 10x for 100% dataset
    print(f"   Estimated total time: {estimated_total_time/60:.1f} minutes ({estimated_total_time/3600:.2f} hours)")
    
    if model_timings:
        print(f"\n   Estimated per-model time:")
        for model, elapsed in model_timings.items():
            estimated_model_time = elapsed * 10
            print(f"   - {model}: {estimated_model_time/60:.1f} minutes ({estimated_model_time/3600:.2f} hours)")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if harmful_benign_results:
        final_harmful_benign = pd.concat(harmful_benign_results, ignore_index=True)
        
        print("\n1. HARMFUL/BENIGN ANALYSIS BY MODEL:")
        print("-" * 80)
        
        for model in final_harmful_benign['model'].unique():
            model_df = final_harmful_benign[final_harmful_benign['model'] == model]
            print(f"\n{model}:")
            
            for condition in ['baseline', 'safedecoding', 'llamaguard3']:
                cond_df = model_df[model_df['evaluation_type'] == condition]
                if len(cond_df) > 0:
                    avg_score = cond_df['strongreject_score'].mean()
                    refusal_rate = cond_df['simple_refusal'].sum() / len(cond_df)
                    avg_length = cond_df['response_length'].mean()
                    
                    print(f"  {condition.capitalize()}: Score={avg_score:.3f}, Refusal={refusal_rate:.1%}, Len={avg_length:.1f}")
    
    if multiturn_results:
        final_multiturn = pd.concat(multiturn_results, ignore_index=True)
        
        print("\n2. MULTI-TURN DEGRADATION BY MODEL:")
        print("-" * 80)
        
        for model in final_multiturn['model'].unique():
            model_df = final_multiturn[final_multiturn['model'] == model]
            print(f"\n{model}:")
            
            for condition in ['baseline', 'safedecoding', 'llamaguard3']:
                cond_df = model_df[model_df['evaluation_type'] == condition]
                if len(cond_df) > 0:
                    avg_degradation = cond_df['cumulative_degradation'].mean()
                    avg_score = cond_df['strongreject_score'].mean()
                    
                    print(f"  {condition.capitalize()}: Degradation={avg_degradation:+.3f}, Score={avg_score:.3f}")
    
    # ========================================================================
    # VALIDATION & NEXT STEPS
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION & NEXT STEPS")
    print("=" * 80)
    
    # Check for errors
    errors_found = False
    
    expected_hb_count = len(harmful_benign_sample) * len(model_timings) * 3
    expected_mt_count = len(multiturn_sample) * len(model_timings) * 3
    
    if harmful_benign_results:
        actual_hb_count = len(pd.concat(harmful_benign_results, ignore_index=True))
        if actual_hb_count < expected_hb_count * 0.9:  # Allow 10% tolerance
            print(f"\n⚠️ Warning: Expected ~{expected_hb_count} harmful/benign rows, got {actual_hb_count}")
            errors_found = True
        else:
            print(f"\n✅ Harmful/benign row count: {actual_hb_count} (expected ~{expected_hb_count})")
    
    if multiturn_results:
        actual_mt_count = len(pd.concat(multiturn_results, ignore_index=True))
        print(f"✅ Multi-turn row count: {actual_mt_count} (expected ~{expected_mt_count})")
    
    if not errors_found:
        print("\n✅ All validations passed!")
        print("\n📋 Next Steps:")
        print("   1. Review sample results in:")
        for file_type, filepath in output_files.items():
            print(f"      - {filepath}")
        print("\n   2. If results look good, run full pipeline:")
        print("      python evaluation_pipeline.py")
        print(f"\n   3. Full pipeline estimated to take {estimated_total_time/3600:.1f} hours")
        print("      Plan accordingly (run overnight or during work hours)")
    else:
        print("\n⚠️ Some validations failed - review errors above before proceeding")
    
    print("\n✓ Sample pipeline test complete!")
    
    return output_files


if __name__ == "__main__":
    print("\n🚀 Starting 10% sample test (all models × all conditions)...\n")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    output_files = test_full_pipeline_sample()
    
    if output_files:
        print("\n✅ Test execution successful!")
        print(f"   Generated files:")
        for file_type, filepath in output_files.items():
            print(f"   - {file_type}: {filepath}")
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n❌ Test execution failed - check error messages above")

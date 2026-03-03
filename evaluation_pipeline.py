"""
Evaluation Pipeline: Automated Scoring Script
Complete end-to-end pipeline: prompt → model → StrongREJECT

This module provides a comprehensive evaluation pipeline for testing LLM safety:
1. Load prompts from dataset
2. Generate model responses (supports multiple backends)
3. Evaluate with StrongREJECT automated scoring
4. Calculate metrics by source and overall performance

Supported Models:
- Gemini 2.0/2.5 Flash (GEMINI_API_KEY - Free tier)
- Llama-3.1-8B-Instant via Groq API (GROQ_API_KEY - Free tier, ULTRA-FAST)
- Mistral-7B-Instruct-v0.3 via Ollama (Ollama - Free tier, local)

Supported Defense Mechanisms:
- SafeDecoding: Inference-time safety-aware decoding
- Llama Guard 3: Pre/post-generation content filtering via Meta's safety classifier
"""

import pandas as pd
from typing import List, Callable, Union
from dotenv import load_dotenv
import gc
import torch

# Load environment variables
load_dotenv()


def free_memory():
    """Free up memory between model evaluations to prevent OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[MEMORY] Cleared cache")

# Import modular components from strongreject_pipeline
from strongreject_pipeline.prompt import load_prompts, sample_prompts, add_harmful_labels
from strongreject_pipeline.llm_model import (
    ModelInference,
    DummyModelInference,
    GeminiModelInference,
    GroqModelInference,
    OllamaModelInference
)
from strongreject_pipeline.strongreject_scoring import (
    StrongRejectEvaluator,
    evaluate_model_responses
)

# Import defense mechanisms
from safe_decoding import SafeDecoder, create_safe_decoder
from guardrails import LlamaGuard3Wrapper, create_llamaguard_wrapper

# Import multi-turn degradation analysis
from strongreject_pipeline.multi_turn_degradation import run_multiturn_evaluation_pipeline


def run_evaluation_pipeline(
    prompts_df: pd.DataFrame,
    model_inference: Union[ModelInference, Callable[[List[str]], List[str]]],
    output_file: str = "dataset/evaluation_results.csv",
    prompt_column: str = "prompt",
    source_column: str = "source",
    strongreject_evaluator: str = "strongreject_finetuned"
) -> pd.DataFrame:
    """
    Complete evaluation pipeline: prompt → model → StrongREJECT
    
    Args:
        prompts_df: DataFrame containing prompts and sources
        model_inference: ModelInference instance or callable that takes prompts and returns responses
                        Can also be a defense mechanism wrapper (SafeDecoder, LlamaGuard3Wrapper)
        output_file: Path to save evaluation results
        prompt_column: Name of prompt column in DataFrame
        source_column: Name of source column in DataFrame
        strongreject_evaluator: StrongREJECT evaluator type
    
    Returns:
        DataFrame with evaluation results
    
    Example:
        >>> # Using built-in model
        >>> model = GeminiModelInference(model="gemini-2.5-flash")
        >>> results = run_evaluation_pipeline(
        ...     prompts_df=pd.read_csv("dataset/prompts_dataset.csv"),
        ...     model_inference=model
        ... )
        
        >>> # Using SafeDecoding defense
        >>> safe_decoder = create_safe_decoder("ollama", safety_weight=0.7, model="mistral")
        >>> results = run_evaluation_pipeline(
        ...     prompts_df=pd.read_csv("dataset/prompts_dataset.csv"),
        ...     model_inference=safe_decoder
        ... )
        
        >>> # Using Llama Guard 3 guardrails
        >>> guardrail = create_llamaguard_wrapper("ollama", model="mistral")
        >>> results = run_evaluation_pipeline(
        ...     prompts_df=pd.read_csv("dataset/prompts_dataset.csv"),
        ...     model_inference=guardrail
        ... )
        
        >>> # Using custom function
        >>> def my_model(prompts):
        ...     return [generate_response(p) for p in prompts]
        >>> results = run_evaluation_pipeline(
        ...     prompts_df=pd.read_csv("dataset/prompts_dataset.csv"),
        ...     model_inference=my_model
        ... )
    """
    print("\n" + "="*80)
    print("EVALUATION PIPELINE: PROMPT → MODEL → EVALUATION")
    print("="*80)
    
    # Step 1: Extract prompts
    prompts = prompts_df[prompt_column].tolist()
    print(f"\nStep 1/3: Loaded {len(prompts)} prompts")
    print(f"  Sources: {prompts_df[source_column].value_counts().to_dict()}")
    
    # Check for is_harmful column
    if 'is_harmful' in prompts_df.columns:
        harmful_count = prompts_df['is_harmful'].sum()
        print(f"  Harmful prompts: {harmful_count} ({harmful_count/len(prompts):.1%})")
        print(f"  Benign prompts: {len(prompts) - harmful_count} ({(len(prompts)-harmful_count)/len(prompts):.1%})")
    
    # Step 2: Generate model responses
    print(f"\nStep 2/3: Generating model responses...")
    try:
        # Check if object has generate_responses method (works for ModelInference and wrappers)
        if hasattr(model_inference, 'generate_responses'):
            model_responses = model_inference.generate_responses(prompts)
        else:
            # Assume it's a callable function
            model_responses = model_inference(prompts)
            
        if len(model_responses) != len(prompts):
            raise ValueError(f"Model returned {len(model_responses)} responses for {len(prompts)} prompts")
        print(f"  ✓ Generated {len(model_responses)} responses successfully")
        
    except Exception as e:
        print(f"  ✗ Error during model inference: {e}")
        raise
    
    # Step 3: Evaluate with StrongREJECT
    print(f"\nStep 3/3: Evaluating with StrongREJECT...")
    
    results = evaluate_model_responses(
        prompts_df=prompts_df,
        model_responses=model_responses,
        output_file=output_file,
        prompt_column=prompt_column,
        source_column=source_column,
        strongreject_evaluator=strongreject_evaluator
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"✓ Evaluated {len(results)} responses")
    print(f"✓ Results saved to: {output_file}")
    print(f"\nResult columns:")
    print(f"  - {prompt_column}, {source_column}, response, is_harmful")
    print(f"  - strongreject_score (0=Safe, 1=Harmful)")
    print(f"  - simple_refusal (keyword-based), response_length (word count)")
    
    return results


if __name__ == "__main__":

    import os
    os.makedirs("results", exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE EVALUATION: 3 MODELS × 3 CONDITIONS")
    print("="*80)
    
    # Load evaluation dataset using modular prompt loader
    prompts_df = load_prompts("dataset/prompts.csv")
    
    # Add harmful labels if not present
    if 'is_harmful' not in prompts_df.columns:
        prompts_df = add_harmful_labels(prompts_df)
    
    print("\nMODELS TO EVALUATE:")
    print("  1. Gemini 2.5 Flash (GEMINI_API_KEY required)")
    print("  2. Llama-3.1-8B-Instant (GROQ_API_KEY required)")
    print("  3. Mistral-7B-Instruct-v0.3 (Ollama required)")
    print("\nCONDITIONS FOR EACH MODEL:")
    print("  - Baseline (no defense)")
    print("  - SafeDecoding (inference-time safety)")
    print("  - Llama Guard 3 (pre/post-filtering)")
    print("\nDATASET SOURCES:")
    print("  - walledeval_diverse: Harmful prompts (~200)")
    print("  - xstest_benign: Benign prompts (~100)")
    print("  - cosafe_multiturn: Multi-turn conversations (~50)")
    print("\nPREREQUISITES:")
    print("  - GEMINI_API_KEY in .env")
    print("  - GROQ_API_KEY in .env")
    print("  - Ollama running: ollama serve")
    print("  - Models pulled: ollama pull mistral && ollama pull llama-guard3")
    
    # Storage for consolidated results
    harmful_benign_results = []  # For harmful & benign prompts
    multiturn_results = []       # For multi-turn degradation analysis
    
    # ========================================================================
    # GEMINI 2.5 FLASH EVALUATIONS
    # ========================================================================
    model_name = "Gemini-2.5 Flash"
    
    # --- BASELINE ---
    print("\n" + "="*80)
    print("GEMINI 2.5 FLASH - BASELINE")
    print("="*80)
    try:
        model = GeminiModelInference(
            model="gemini-2.5-flash",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=model,
            output_file="results/temp_gemini_baseline.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'baseline'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=model,
            model_name=model_name,
            condition='baseline'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # --- SAFEDECODING ---
    print("\n" + "="*80)
    print("GEMINI 2.5 FLASH - SAFEDECODING")
    print("="*80)
    try:
        safe_decoder = create_safe_decoder(
            model_type="gemini",
            safety_weight=0.7,
            model="gemini-2.5-flash",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=safe_decoder,
            output_file="results/temp_gemini_safedecoding.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'safedecoding'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=safe_decoder,
            model_name=model_name,
            condition='safedecoding'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # --- LLAMA GUARD 3 ---
    print("\n" + "="*80)
    print("GEMINI 2.5 FLASH - LLAMA GUARD 3")
    print("="*80)
    try:
        guardrail_wrapper = create_llamaguard_wrapper(
            model_type="gemini",
            model="gemini-2.5-flash",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=guardrail_wrapper,
            output_file="results/temp_gemini_llamaguard3.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'llamaguard3'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=guardrail_wrapper,
            model_name=model_name,
            condition='llamaguard3'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Free memory after Gemini evaluations
    print("\n" + "="*80)
    print("MEMORY CLEANUP: GEMINI EVALUATIONS COMPLETE")
    print("="*80)
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
    
    # ========================================================================
    # LLAMA 3.1 (GROQ) EVALUATIONS
    # ========================================================================
    model_name = "Llama-3.1-8B"
    
    # --- BASELINE ---
    print("\n" + "="*80)
    print("LLAMA 3.1-8B-INSTANT (GROQ) - BASELINE")
    print("="*80)
    try:
        model = GroqModelInference(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=model,
            output_file="results/temp_llama31_baseline.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'baseline'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=model,
            model_name=model_name,
            condition='baseline'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # --- SAFEDECODING ---
    print("\n" + "="*80)
    print("LLAMA 3.1-8B-INSTANT (GROQ) - SAFEDECODING")
    print("="*80)
    try:
        safe_decoder = create_safe_decoder(
            model_type="groq",
            safety_weight=0.7,
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=safe_decoder,
            output_file="results/temp_llama31_safedecoding.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'safedecoding'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=safe_decoder,
            model_name=model_name,
            condition='safedecoding'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # --- LLAMA GUARD 3 ---
    print("\n" + "="*80)
    print("LLAMA 3.1-8B-INSTANT (GROQ) - LLAMA GUARD 3")
    print("="*80)
    try:
        guardrail_wrapper = create_llamaguard_wrapper(
            model_type="groq",
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=guardrail_wrapper,
            output_file="results/temp_llama31_llamaguard3.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'llamaguard3'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=guardrail_wrapper,
            model_name=model_name,
            condition='llamaguard3'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Free memory after Llama 3.1 evaluations
    print("\n" + "="*80)
    print("MEMORY CLEANUP: LLAMA 3.1 EVALUATIONS COMPLETE")
    print("="*80)
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
    
    # ========================================================================
    # MISTRAL 7B (OLLAMA) EVALUATIONS
    # ========================================================================
    model_name = "Mistral-7B"
    
    # --- BASELINE ---
    print("\n" + "="*80)
    print("MISTRAL 7B-INSTRUCT (OLLAMA) - BASELINE")
    print("="*80)
    try:
        model = OllamaModelInference(
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=model,
            output_file="results/temp_mistral_baseline.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'baseline'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=model,
            model_name=model_name,
            condition='baseline'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # --- SAFEDECODING ---
    print("\n" + "="*80)
    print("MISTRAL 7B-INSTRUCT (OLLAMA) - SAFEDECODING")
    print("="*80)
    try:
        safe_decoder = create_safe_decoder(
            model_type="ollama",
            safety_weight=0.7,
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=safe_decoder,
            output_file="results/temp_mistral_safedecoding.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'safedecoding'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=safe_decoder,
            model_name=model_name,
            condition='safedecoding'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # --- LLAMA GUARD 3 ---
    print("\n" + "="*80)
    print("MISTRAL 7B-INSTRUCT (OLLAMA) - LLAMA GUARD 3")
    print("="*80)
    try:
        guardrail_wrapper = create_llamaguard_wrapper(
            model_type="ollama",
            model="mistral",
            temperature=0.0,
            max_tokens=500
        )
        
        # Evaluate harmful & benign prompts
        harmful_benign_df = prompts_df[prompts_df['source'].isin(['walledeval_diverse', 'xstest_benign'])].copy()
        results = run_evaluation_pipeline(
            prompts_df=harmful_benign_df,
            model_inference=guardrail_wrapper,
            output_file="results/temp_mistral_llamaguard3.csv"
        )
        
        # Add metadata
        results['model'] = model_name
        results['evaluation_type'] = 'llamaguard3'
        results['prompt_type'] = results['source'].map({
            'walledeval_diverse': 'harmful prompts',
            'xstest_benign': 'benign prompts'
        })
        harmful_benign_results.append(results)
        
        # Multi-turn degradation analysis
        multiturn_df = run_multiturn_evaluation_pipeline(
            prompts_df=prompts_df,
            model_inference=guardrail_wrapper,
            model_name=model_name,
            condition='llamaguard3'
        )
        if len(multiturn_df) > 0:
            multiturn_results.append(multiturn_df)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Free memory after Mistral evaluations
    print("\n" + "="*80)
    print("MEMORY CLEANUP: MISTRAL EVALUATIONS COMPLETE")
    print("="*80)
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
    
    # ========================================================================
    # CONSOLIDATE AND SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("CONSOLIDATING RESULTS")
    print("="*80)
    
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
        
        output_file_1 = "results/evaluation_harmful_benign_analysis.csv"
        final_harmful_benign.to_csv(output_file_1, index=False)
        print(f"✅ Saved harmful/benign analysis: {output_file_1}")
        print(f"   Total rows: {len(final_harmful_benign)}")
        print(f"   Models: {final_harmful_benign['model'].unique()}")
        print(f"   Conditions: {final_harmful_benign['evaluation_type'].unique()}")
        print(f"   Prompt types: {final_harmful_benign['prompt_type'].unique()}")
    else:
        print("⚠️ No harmful/benign results to consolidate")
    
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
        
        output_file_2 = "results/evaluation_multi_turn_analysis.csv"
        final_multiturn.to_csv(output_file_2, index=False)
        print(f"\n✅ Saved multi-turn analysis: {output_file_2}")
        print(f"   Total rows: {len(final_multiturn)}")
        print(f"   Conversations: {final_multiturn['conversation_id'].nunique()}")
        print(f"   Models: {final_multiturn['model'].unique()}")
        print(f"   Conditions: {final_multiturn['evaluation_type'].unique()}")
    else:
        print("\n⚠️ No multi-turn results to consolidate")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("✓ Two consolidated files generated:")
    print(f"  1. {output_file_1 if harmful_benign_results else 'N/A'}")
    print(f"  2. {output_file_2 if multiturn_results else 'N/A'}")
    print("\n📊 Ready for analysis and visualization!")
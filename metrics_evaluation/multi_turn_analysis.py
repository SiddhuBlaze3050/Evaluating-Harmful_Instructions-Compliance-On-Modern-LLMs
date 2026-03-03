"""
Multi-Turn Degradation Analysis

Analyzes how safety degrades across conversation turns in multi-turn attack scenarios.
Implements Research-Roadmap.md Section 7.3 specification.

Key Metrics:
- Safety degradation slope (negative = degradation over turns)
- R-squared (how linear the degradation is)
- Per-turn statistics (mean, std, count)
- Comparison across models and conditions

Reference: Research-Roadmap.md Section 7.3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def analyze_multiturn_degradation(multiturn_results: pd.DataFrame, 
                                   output_dir: str = "metrics_output",
                                   metric_column: str = "strongreject_score") -> Dict:
    """
    Measure how safety degrades with conversation length.
    
    Follows Research-Roadmap.md Section 7.3 specification.
    
    Args:
        multiturn_results: DataFrame with multi-turn conversation results
            Required columns: turn_number, strongreject_score (or safety)
        output_dir: Directory for saving plots
        metric_column: Column to analyze ('strongreject_score' or 'safety')
    
    Returns:
        Dictionary with:
            - slope: Degradation slope (negative = safety decreases)
            - r_squared: How linear the degradation is
            - p_value: Statistical significance
            - by_turn_stats: Per-turn statistics
    """
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by turn number
    by_turn = multiturn_results.groupby('turn_number').agg({
        metric_column: ['mean', 'std', 'count']
    })
    
    # Plot safety degradation
    plt.figure(figsize=(10, 6))
    
    turns = by_turn.index
    safety_mean = by_turn[(metric_column, 'mean')]
    safety_std = by_turn[(metric_column, 'std')]
    
    plt.plot(turns, safety_mean, marker='o', linewidth=2,
             label='Mean Safety Score')
    plt.fill_between(turns,
                     safety_mean - safety_std,
                     safety_mean + safety_std,
                     alpha=0.3)
    
    plt.xlabel("Conversation Turn", fontsize=12)
    plt.ylabel(f"{metric_column.replace('_', ' ').title()}", fontsize=12)
    plt.title("Safety Degradation Across Turns", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add linear regression line
    slope, intercept, r_value, p_value, std_err = linregress(turns, safety_mean)
    
    plt.plot(turns, slope * turns + intercept,
             '--', color='red',
             label=f'Linear fit (R²={r_value**2:.3f})')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "multiturn_degradation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "slope": slope,  # Negative = degradation
        "r_squared": r_value**2,
        "p_value": p_value,
        "std_err": std_err,
        "by_turn_stats": by_turn
    }


def analyze_by_model_condition(multiturn_results: pd.DataFrame,
                               output_dir: str = "metrics_output",
                               metric_column: str = "strongreject_score") -> pd.DataFrame:
    """
    Analyze degradation separately for each model-condition combination.
    
    Args:
        multiturn_results: DataFrame with multi-turn results
            Required columns: model, evaluation_type, turn_number, metric_column
        output_dir: Directory for saving plots
        metric_column: Column to analyze
    
    Returns:
        DataFrame with degradation statistics per model-condition
    """
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: By Model
    ax1 = axes[0]
    for model in multiturn_results['model'].unique():
        model_data = multiturn_results[multiturn_results['model'] == model]
        by_turn = model_data.groupby('turn_number')[metric_column].mean()
        
        ax1.plot(by_turn.index, by_turn.values, marker='o', 
                linewidth=2, label=model)
        
        # Compute regression
        if len(by_turn) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(
                by_turn.index, by_turn.values
            )
            results.append({
                'model': model,
                'condition': 'all',
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'initial_safety': by_turn.iloc[0],
                'final_safety': by_turn.iloc[-1],
                'total_degradation': by_turn.iloc[-1] - by_turn.iloc[0]
            })
    
    ax1.set_xlabel("Conversation Turn", fontsize=12)
    ax1.set_ylabel(f"{metric_column.replace('_', ' ').title()}", fontsize=12)
    ax1.set_title("Safety Degradation by Model", fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: By Condition
    ax2 = axes[1]
    for condition in multiturn_results['evaluation_type'].unique():
        cond_data = multiturn_results[multiturn_results['evaluation_type'] == condition]
        by_turn = cond_data.groupby('turn_number')[metric_column].mean()
        
        ax2.plot(by_turn.index, by_turn.values, marker='s',
                linewidth=2, label=condition)
        
        # Compute regression
        if len(by_turn) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(
                by_turn.index, by_turn.values
            )
            results.append({
                'model': 'all',
                'condition': condition,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'initial_safety': by_turn.iloc[0],
                'final_safety': by_turn.iloc[-1],
                'total_degradation': by_turn.iloc[-1] - by_turn.iloc[0]
            })
    
    ax2.set_xlabel("Conversation Turn", fontsize=12)
    ax2.set_ylabel(f"{metric_column.replace('_', ' ').title()}", fontsize=12)
    ax2.set_title("Safety Degradation by Condition", fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "multiturn_degradation_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detailed analysis: Each model-condition combination
    fig, axes = plt.subplots(
        len(multiturn_results['model'].unique()),
        len(multiturn_results['evaluation_type'].unique()),
        figsize=(15, 12)
    )
    
    if len(multiturn_results['model'].unique()) == 1:
        axes = axes.reshape(1, -1)
    if len(multiturn_results['evaluation_type'].unique()) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, model in enumerate(sorted(multiturn_results['model'].unique())):
        for j, condition in enumerate(sorted(multiturn_results['evaluation_type'].unique())):
            ax = axes[i, j]
            
            subset = multiturn_results[
                (multiturn_results['model'] == model) &
                (multiturn_results['evaluation_type'] == condition)
            ]
            
            if len(subset) > 0:
                by_turn = subset.groupby('turn_number').agg({
                    metric_column: ['mean', 'std', 'count']
                })
                
                turns = by_turn.index
                mean_vals = by_turn[(metric_column, 'mean')]
                std_vals = by_turn[(metric_column, 'std')]
                
                ax.plot(turns, mean_vals, marker='o', linewidth=2, color='blue')
                ax.fill_between(turns,
                               mean_vals - std_vals,
                               mean_vals + std_vals,
                               alpha=0.3, color='blue')
                
                # Linear fit
                if len(turns) > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        turns, mean_vals
                    )
                    ax.plot(turns, slope * turns + intercept,
                           '--', color='red', alpha=0.7,
                           label=f'R²={r_value**2:.3f}')
                    
                    results.append({
                        'model': model,
                        'condition': condition,
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_err': std_err,
                        'initial_safety': mean_vals.iloc[0],
                        'final_safety': mean_vals.iloc[-1],
                        'total_degradation': mean_vals.iloc[-1] - mean_vals.iloc[0]
                    })
                
                ax.set_title(f"{model}\n{condition}", fontsize=10)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            
            if i == len(multiturn_results['model'].unique()) - 1:
                ax.set_xlabel("Turn", fontsize=10)
            if j == 0:
                ax.set_ylabel("Safety Score", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "multiturn_degradation_detailed.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    
    results_df = pd.DataFrame(results)
    return results_df


def compute_degradation_metrics(multiturn_results: pd.DataFrame,
                                metric_column: str = "strongreject_score") -> pd.DataFrame:
    """
    Compute comprehensive degradation metrics for all model-condition combinations.
    
    Args:
        multiturn_results: DataFrame with multi-turn results
        metric_column: Column to analyze
    
    Returns:
        DataFrame with degradation metrics
    """
    results = []
    
    for (model, condition), group in multiturn_results.groupby(['model', 'evaluation_type']):
        by_turn = group.groupby('turn_number')[metric_column].agg(['mean', 'std', 'count'])
        
        if len(by_turn) > 1:
            turns = by_turn.index.values
            means = by_turn['mean'].values
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(turns, means)
            
            # Degradation metrics
            initial_safety = means[0]
            final_safety = means[-1]
            total_degradation = final_safety - initial_safety
            degradation_percent = (total_degradation / initial_safety * 100) if initial_safety > 0 else 0
            
            # Per-turn average degradation
            avg_degradation_per_turn = total_degradation / (len(turns) - 1) if len(turns) > 1 else 0
            
            results.append({
                'model': model,
                'condition': condition,
                'n_conversations': len(group['conversation_id'].unique()),
                'n_turns': len(by_turn),
                'initial_safety': initial_safety,
                'final_safety': final_safety,
                'total_degradation': total_degradation,
                'degradation_percent': degradation_percent,
                'avg_degradation_per_turn': avg_degradation_per_turn,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'is_significant': p_value < 0.05
            })
    
    return pd.DataFrame(results)


def print_degradation_summary(degradation_df: pd.DataFrame):
    """
    Print formatted summary of multi-turn degradation analysis.
    
    Args:
        degradation_df: DataFrame from compute_degradation_metrics
    """
    print("\n" + "=" * 100)
    print("MULTI-TURN DEGRADATION ANALYSIS")
    print("=" * 100 + "\n")
    
    print("Key Findings:")
    print("-" * 100)
    
    # Overall statistics
    avg_slope = degradation_df['slope'].mean()
    avg_r_squared = degradation_df['r_squared'].mean()
    
    print(f"\nOverall Degradation:")
    print(f"  Average slope: {avg_slope:.4f} (negative = safety decreases per turn)")
    print(f"  Average R²: {avg_r_squared:.3f} (linearity of degradation)")
    print(f"  Significant degradation: {(degradation_df['is_significant'] == True).sum()}/{len(degradation_df)} configurations")
    
    # Worst degradation
    worst = degradation_df.loc[degradation_df['slope'].idxmin()]
    print(f"\nWorst Degradation:")
    print(f"  {worst['model']} ({worst['condition']})")
    print(f"  Slope: {worst['slope']:.4f} per turn")
    print(f"  Total degradation: {worst['total_degradation']:.4f} ({worst['degradation_percent']:.1f}%)")
    
    # Best resistance
    best = degradation_df.loc[degradation_df['slope'].idxmax()]
    print(f"\nBest Resistance:")
    print(f"  {best['model']} ({best['condition']})")
    print(f"  Slope: {best['slope']:.4f} per turn")
    print(f"  Total degradation: {best['total_degradation']:.4f} ({best['degradation_percent']:.1f}%)")
    
    # Detailed table
    print("\n" + "=" * 100)
    print("DETAILED DEGRADATION METRICS")
    print("=" * 100)
    
    display_df = degradation_df.copy()
    display_df['slope'] = display_df['slope'].apply(lambda x: f"{x:.4f}")
    display_df['r_squared'] = display_df['r_squared'].apply(lambda x: f"{x:.3f}")
    display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}")
    display_df['degradation_percent'] = display_df['degradation_percent'].apply(lambda x: f"{x:.1f}%")
    
    cols_to_print = ['model', 'condition', 'slope', 'r_squared', 'p_value', 
                    'degradation_percent', 'is_significant']
    print(display_df[cols_to_print].to_string(index=False))
    print("=" * 100 + "\n")


def generate_multiturn_report(multiturn_results: pd.DataFrame,
                              output_dir: str = "metrics_output") -> Dict:
    """
    Generate comprehensive multi-turn degradation report.
    
    Combines all analysis functions and generates visualizations and tables.
    
    Args:
        multiturn_results: DataFrame with multi-turn conversation results
        output_dir: Directory for saving outputs
    
    Returns:
        Dictionary with all analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "tables").mkdir(exist_ok=True)
    (output_path / "figures").mkdir(exist_ok=True)
    
    print("\n" + "=" * 100)
    print("🔄 GENERATING MULTI-TURN DEGRADATION REPORT")
    print("=" * 100)
    
    # 1. Overall degradation analysis
    print("\n1. Analyzing overall degradation pattern...")
    overall_results = analyze_multiturn_degradation(
        multiturn_results, 
        output_dir=output_dir,
        metric_column="strongreject_score"
    )
    
    # 2. By model and condition
    print("\n2. Analyzing degradation by model and condition...")
    comparison_df = analyze_by_model_condition(
        multiturn_results,
        output_dir=output_dir,
        metric_column="strongreject_score"
    )
    
    # 3. Compute comprehensive metrics
    print("\n3. Computing degradation metrics...")
    degradation_metrics = compute_degradation_metrics(
        multiturn_results,
        metric_column="strongreject_score"
    )
    
    # 4. Save results
    degradation_metrics.to_csv(
        output_path / "tables" / "multiturn_degradation_metrics.csv",
        index=False
    )
    
    comparison_df.to_csv(
        output_path / "tables" / "multiturn_degradation_comparison.csv",
        index=False
    )
    
    # 5. Print summary
    print_degradation_summary(degradation_metrics)
    
    print(f"\n✓ Multi-turn degradation analysis complete")
    print(f"  - Degradation metrics: {output_path / 'tables' / 'multiturn_degradation_metrics.csv'}")
    print(f"  - Overall plot: {output_path / 'figures' / 'multiturn_degradation.png'}")
    print(f"  - Comparison plots: {output_path / 'figures' / 'multiturn_degradation_comparison.png'}")
    print(f"  - Detailed plots: {output_path / 'figures' / 'multiturn_degradation_detailed.png'}\n")
    
    return {
        'overall_results': overall_results,
        'comparison_df': comparison_df,
        'degradation_metrics': degradation_metrics
    }


def main():
    """
    Main execution function for standalone use.
    
    Usage:
        python multi_turn_analysis.py
    """
    # Default paths
    multiturn_csv = "results/sample_multi_turn_analysis.csv"
    output_dir = "metrics_output"
    
    # Check if results file exists
    import os
    if not os.path.exists(multiturn_csv):
        print(f"⚠ Error: {multiturn_csv} not found")
        print("Please run multi-turn evaluation first or specify correct path")
        return None
    
    # Load data
    print(f"\n📊 Loading multi-turn results from: {multiturn_csv}")
    df = pd.read_csv(multiturn_csv)
    print(f"✓ Loaded {len(df)} turn records from {len(df['conversation_id'].unique())} conversations")
    
    # Generate report
    results = generate_multiturn_report(df, output_dir)
    
    return results


if __name__ == "__main__":
    main()

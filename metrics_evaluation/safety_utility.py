"""
Safety-Utility Pareto Frontier Analysis

Implements Pareto frontier visualization and analysis for LLM safety evaluation.
Measures the tradeoff between safety (refusal of harmful content) and utility 
(preservation of model capabilities on benign tasks).

Key Insight: Perfect safety with zero utility is useless. Model deployment 
requires balancing both dimensions.

Reference: Research-Roadmap.md Section 7.2.3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def compute_safety_utility_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute safety and utility scores for each model-condition combination.
    
    Args:
        df: Evaluation results DataFrame with columns:
            - model: Model name
            - evaluation_type: Condition (baseline, safedecoding, llamaguard3)
            - is_harmful: Binary label (1=harmful, 0=benign)
            - safety: Pre-calculated safety score (1 - strongreject_score)
            - utility: Utility preservation score (0-1, higher=better)
    
    Returns:
        DataFrame with columns:
            - model: Model name
            - condition: Evaluation condition
            - safety: mean(safety) on harmful prompts
            - utility: mean(utility) on benign prompts
            - pareto_score: (safety + utility) / 2
            - n_samples, n_harmful, n_benign: Sample counts
    """
    results = []
    
    for (model, condition), group in df.groupby(['model', 'evaluation_type']):
        # Filter by prompt type
        harmful = group[group['is_harmful'] == 1]
        benign = group[group['is_harmful'] == 0]
        
        # Safety: Use pre-calculated safety column (1 - strongreject_score) on harmful prompts
        # Higher safety score = safer model behavior
        safety_score = harmful['safety'].mean() if len(harmful) > 0 else 0
        
        # Utility: Average utility on benign prompts (higher = better)
        utility_score = benign['utility'].mean() if len(benign) > 0 else 0
        
        # Pareto score: Balance of safety and utility
        pareto_score = (safety_score + utility_score) / 2
        
        results.append({
            'model': model,
            'condition': condition,
            'safety': safety_score,
            'utility': utility_score,
            'pareto_score': pareto_score,
            'n_samples': len(group),
            'n_harmful': len(harmful),
            'n_benign': len(benign)
        })
    
    return pd.DataFrame(results)


def identify_pareto_optimal(df: pd.DataFrame, 
                            safety_col: str = 'safety',
                            utility_col: str = 'utility') -> pd.DataFrame:
    """
    Identify Pareto optimal points (not dominated by any other point).
    
    A point is Pareto optimal if no other point has both higher safety AND higher utility.
    
    Args:
        df: DataFrame with safety and utility columns
        safety_col: Column name for safety metric
        utility_col: Column name for utility metric
    
    Returns:
        DataFrame with additional 'is_pareto_optimal' boolean column
    """
    df = df.copy()
    df['is_pareto_optimal'] = False
    
    for idx, row in df.iterrows():
        # Check if any other point dominates this one
        dominated = False
        for _, other_row in df.iterrows():
            if (other_row[safety_col] >= row[safety_col] and 
                other_row[utility_col] >= row[utility_col] and
                (other_row[safety_col] > row[safety_col] or 
                 other_row[utility_col] > row[utility_col])):
                dominated = True
                break
        
        if not dominated:
            df.at[idx, 'is_pareto_optimal'] = True
    
    return df


def plot_safety_utility_frontier(df: pd.DataFrame,
                                 safety_col: str = 'safety',
                                 utility_col: str = 'utility',
                                 group_by: str = 'condition',
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create Pareto frontier plot: safety (y) vs utility (x).
    
    Args:
        df: DataFrame from compute_safety_utility_metrics()
        safety_col: Column to use for y-axis (safety)
        utility_col: Column to use for x-axis (utility)
        group_by: Column to group by for coloring ('condition' or 'model')
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color schemes
    condition_colors = {
        'baseline': '#E74C3C',      # Red
        'safedecoding': '#3498DB',  # Blue
        'llamaguard3': '#2ECC71',   # Green
        'guardrail': '#2ECC71'      # Green (alias)
    }
    
    condition_markers = {
        'baseline': 'o',
        'safedecoding': 's',
        'llamaguard3': '^',
        'guardrail': '^'
    }
    
    # Identify Pareto optimal points
    df = identify_pareto_optimal(df, safety_col, utility_col)
    
    # Plot each group
    groups = df[group_by].unique()
    
    for group_val in groups:
        subset = df[df[group_by] == group_val]
        
        color = condition_colors.get(group_val, 'gray')
        marker = condition_markers.get(group_val, 'o')
        
        # Plot points
        ax.scatter(
            subset[utility_col],
            subset[safety_col],
            label=group_val.upper(),
            color=color,
            marker=marker,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )
        
        # Add model labels
        for _, row in subset.iterrows():
            # Offset label slightly
            offset_x = 0.02
            offset_y = 0.02
            
            label_text = row['model']
            if row['is_pareto_optimal']:
                label_text = f"★ {label_text}"  # Star for Pareto optimal
            
            ax.annotate(
                label_text,
                (row[utility_col], row[safety_col]),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold' if row['is_pareto_optimal'] else 'normal',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='yellow' if row['is_pareto_optimal'] else 'white',
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            )
    
    # Highlight Pareto frontier
    pareto_points = df[df['is_pareto_optimal']].sort_values(utility_col)
    if len(pareto_points) > 1:
        ax.plot(
            pareto_points[utility_col],
            pareto_points[safety_col],
            'k--',
            linewidth=2,
            alpha=0.3,
            label='Pareto Frontier'
        )
    
    # Target region lines (ideal: high safety AND high utility)
    target_safety = 0.8
    target_utility = 0.8
    
    ax.axhline(y=target_safety, color='green', linestyle='--', 
               alpha=0.3, linewidth=2, label=f'Target Safety ({target_safety:.0%})')
    ax.axvline(x=target_utility, color='blue', linestyle='--', 
               alpha=0.3, linewidth=2, label=f'Target Utility ({target_utility:.0%})')
    
    # Shade ideal region
    ax.fill_between([target_utility, 1.0], target_safety, 1.0, 
                     alpha=0.1, color='green', label='Ideal Region')
    
    # Labels and formatting
    ax.set_xlabel("Utility (Benign Prompts) →", fontsize=14, fontweight='bold')
    ax.set_ylabel("Safety (1 - StrongREJECT on Harmful) →", fontsize=14, fontweight='bold')
    ax.set_title("Safety-Utility Pareto Frontier\nBalancing Safety and Model Capability", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits with padding
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # Legend
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    
    # Add interpretation text
    interpretation = (
        "★ = Pareto Optimal (not dominated by any other point)\n"
        "Ideal models: Top-right (high safety + high utility)\n"
        f"Best tradeoff: Pareto frontier points"
    )
    ax.text(0.02, 0.98, interpretation,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved safety-utility frontier to: {save_path}")
    
    return fig


def plot_condition_comparison(df: pd.DataFrame,
                              safety_col: str = 'safety',
                              utility_col: str = 'utility',
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create faceted plot comparing conditions across models.
    
    Args:
        df: DataFrame from compute_safety_utility_metrics()
        safety_col: Safety metric column
        utility_col: Utility metric column
        save_path: Optional save path
    
    Returns:
        matplotlib Figure
    """
    models = df['model'].unique()
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5), sharey=True)
    
    if n_models == 1:
        axes = [axes]
    
    condition_colors = {
        'baseline': '#E74C3C',
        'safedecoding': '#3498DB',
        'llamaguard3': '#2ECC71'
    }
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = df[df['model'] == model]
        
        for _, row in model_data.iterrows():
            color = condition_colors.get(row['condition'], 'gray')
            
            ax.scatter(
                row[utility_col],
                row[safety_col],
                color=color,
                s=200,
                alpha=0.7,
                edgecolors='black',
                linewidths=2,
                label=row['condition'].upper()
            )
            
            ax.annotate(
                row['condition'],
                (row[utility_col], row[safety_col]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
        
        ax.set_xlabel("Utility", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Safety", fontsize=12)
        ax.set_title(model, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=9)
    
    plt.suptitle("Safety-Utility Tradeoff by Model", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved condition comparison to: {save_path}")
    
    return fig


def print_pareto_analysis(df: pd.DataFrame,
                         safety_col: str = 'safety',
                         utility_col: str = 'utility'):
    """
    Print detailed Pareto frontier analysis.
    
    Args:
        df: DataFrame with safety-utility metrics
        safety_col: Safety metric column
        utility_col: Utility metric column
    """
    print("\n" + "=" * 100)
    print("SAFETY-UTILITY PARETO FRONTIER ANALYSIS")
    print("=" * 100 + "\n")
    
    # Identify Pareto optimal points
    df = identify_pareto_optimal(df, safety_col, utility_col)
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"  Total configurations: {len(df)}")
    print(f"  Pareto optimal points: {df['is_pareto_optimal'].sum()}")
    print(f"  Safety range: [{df[safety_col].min():.1%}, {df[safety_col].max():.1%}]")
    print(f"  Utility range: [{df[utility_col].min():.1%}, {df[utility_col].max():.1%}]")
    
    # Pareto optimal points
    print("\n" + "-" * 100)
    print("★ PARETO OPTIMAL CONFIGURATIONS (Not dominated by any other)")
    print("-" * 100)
    
    pareto_df = df[df['is_pareto_optimal']].sort_values(safety_col, ascending=False)
    
    for _, row in pareto_df.iterrows():
        print(f"\n{row['model']} - {row['condition'].upper()}")
        print(f"  Safety (1 - StrongREJECT): {row[safety_col]:.1%}")
        print(f"  Utility (Benign):          {row[utility_col]:.1%}")
        print(f"  Pareto Score:              {row['pareto_score']:.3f}")
    
    # Best by metric
    print("\n" + "-" * 100)
    print("BEST BY METRIC")
    print("-" * 100)
    
    best_safety = df.loc[df[safety_col].idxmax()]
    print(f"\nHighest Safety: {best_safety['model']} - {best_safety['condition'].upper()}")
    print(f"  Safety: {best_safety[safety_col]:.1%}, Utility: {best_safety[utility_col]:.1%}")
    
    best_utility = df.loc[df[utility_col].idxmax()]
    print(f"\nHighest Utility: {best_utility['model']} - {best_utility['condition'].upper()}")
    print(f"  Safety: {best_utility[safety_col]:.1%}, Utility: {best_utility[utility_col]:.1%}")
    
    best_pareto = df.loc[df['pareto_score'].idxmax()]
    print(f"\nBest Balance (Pareto Score): {best_pareto['model']} - {best_pareto['condition'].upper()}")
    print(f"  Safety: {best_pareto[safety_col]:.1%}, Utility: {best_pareto[utility_col]:.1%}")
    print(f"  Pareto Score: {best_pareto['pareto_score']:.3f}")
    
    # Condition comparison
    print("\n" + "-" * 100)
    print("AVERAGE BY CONDITION")
    print("-" * 100)
    
    by_condition = df.groupby('condition').agg({
        safety_col: 'mean',
        utility_col: 'mean',
        'pareto_score': 'mean'
    }).sort_values('pareto_score', ascending=False)
    
    for condition, row in by_condition.iterrows():
        print(f"\n{condition.upper()}:")
        print(f"  Avg Safety:  {row[safety_col]:.1%}")
        print(f"  Avg Utility: {row[utility_col]:.1%}")
        print(f"  Avg Pareto:  {row['pareto_score']:.3f}")
    
    print("\n" + "=" * 100 + "\n")


def generate_safety_utility_report(df: pd.DataFrame,
                                   output_dir: Path) -> Dict:
    """
    Generate complete safety-utility analysis report.
    
    Args:
        df: Evaluation results DataFrame
        output_dir: Directory for saving outputs (Path object or string)
    
    Returns:
        Dictionary with computed metrics and figure paths
    """
    print("\n" + "=" * 100)
    print("GENERATING SAFETY-UTILITY ANALYSIS")
    print("=" * 100 + "\n")
    
    # Convert output_dir to Path if it's a string
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Compute metrics
    metrics_df = compute_safety_utility_metrics(df)
    
    # Save metrics table
    metrics_path = output_dir / "tables" / "safety_utility_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Saved safety-utility metrics to: {metrics_path}")
    
    # Identify Pareto optimal points
    pareto_df = identify_pareto_optimal(metrics_df)
    
    # Create visualizations
    figures_dir = output_dir / "figures"
    
    # 1. Main Pareto frontier
    frontier_path = figures_dir / "safety_utility_pareto_frontier.png"
    fig1 = plot_safety_utility_frontier(pareto_df, save_path=str(frontier_path))
    plt.close(fig1)
    
    # 2. Condition comparison
    comparison_path = figures_dir / "safety_utility_by_model.png"
    fig2 = plot_condition_comparison(pareto_df, save_path=str(comparison_path))
    plt.close(fig2)
    
    # Print analysis
    print_pareto_analysis(pareto_df)
    
    return {
        'metrics_df': metrics_df,
        'pareto_df': pareto_df,
        'pareto_frontier_fig': str(frontier_path),
        'comparison_fig': str(comparison_path)
    }


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("Safety-Utility Analysis Module")
    print("=" * 100)
    
    print("\nThis module provides:")
    print("  1. Pareto frontier visualization")
    print("  2. Safety-utility tradeoff analysis")
    print("  3. Identification of Pareto optimal configurations")
    print("  4. Comparative analysis across models and conditions")
    
    print("\nUsage:")
    print("  from safety_utility import generate_safety_utility_report")
    print("  report = generate_safety_utility_report(df, output_dir)")
    
    print("\n✓ Safety-utility module ready for use\n")

"""
False Refusal Rate (FRR) Analysis with Wilson Confidence Intervals

Measures over-refusal: how often models incorrectly refuse benign/safe prompts.
Critical for assessing safety-utility tradeoff and excessive caution.

Key Metrics:
- False Refusal Rate: % of benign prompts incorrectly refused
- Wilson Confidence Intervals: Statistical precision for small samples
- Safety-FRR Tradeoff: Visualization of over-refusal vs harmful detection

Reference: Research-Roadmap.md Section 7.2.2 (Wilson CI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from adjustText import adjust_text as _adjust_text
except ImportError:
    _adjust_text = None

try:
    from plot_style import (
        style_axes, format_legend,
        condition_color, condition_marker, condition_label, short_model_name,
        FIGURE_DPI, FIG_SINGLE, FIG_WIDE, LINE_WIDTH, FONT_SIZE_ANNOT, FONT_SIZE_LEGEND,
    )
except ImportError:
    from metrics_evaluation.plot_style import (
        style_axes, format_legend,
        condition_color, condition_marker, condition_label, short_model_name,
        FIGURE_DPI, FIG_SINGLE, FIG_WIDE, LINE_WIDTH, FONT_SIZE_ANNOT, FONT_SIZE_LEGEND,
    )

# Import Wilson CI functions
try:
    from confidence_intervals import wilson_ci
except ImportError:
    from metrics_evaluation.confidence_intervals import wilson_ci


def compute_false_refusal_rate(df: pd.DataFrame, 
                               confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute False Refusal Rate with Wilson confidence intervals.
    
    FRR = (Number of benign prompts refused) / (Total benign prompts)
    
    Args:
        df: DataFrame with evaluation results
            Required columns: model, evaluation_type, is_harmful, simple_refusal
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        DataFrame with FRR metrics per model-condition:
            - model, condition: Identifiers
            - frr: False Refusal Rate (proportion)
            - ci_lower, ci_upper: Wilson confidence interval bounds
            - n_benign: Total benign prompts
            - n_false_refusals: Number of benign prompts refused
    """
    results = []
    
    for (model, condition), group in df.groupby(['model', 'evaluation_type']):
        # Filter to benign prompts only
        benign = group[group['is_harmful'] == 0]
        
        if len(benign) == 0:
            continue
        
        # Count false refusals (refused benign prompts)
        false_refusals = benign['simple_refusal'].sum()
        total_benign = len(benign)
        
        # Compute Wilson CI
        ci_result = wilson_ci(
            successes=int(false_refusals),
            trials=int(total_benign),
            confidence=confidence
        )
        
        results.append({
            'model': model,
            'condition': condition,
            'frr': ci_result['proportion'],
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
            'ci_margin': ci_result['margin'],
            'n_benign': total_benign,
            'n_false_refusals': int(false_refusals),
            'confidence_level': confidence
        })
    
    return pd.DataFrame(results)


def compute_frr_by_condition(df: pd.DataFrame,
                             confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute FRR aggregated by condition (averaged across models).
    
    Args:
        df: DataFrame with evaluation results
        confidence: Confidence level for Wilson CI
    
    Returns:
        DataFrame with FRR metrics per condition
    """
    results = []
    
    for condition in df['evaluation_type'].unique():
        # Filter to this condition, benign prompts only
        subset = df[
            (df['evaluation_type'] == condition) &
            (df['is_harmful'] == 0)
        ]
        
        if len(subset) == 0:
            continue
        
        # Count false refusals across all models in this condition
        false_refusals = subset['simple_refusal'].sum()
        total_benign = len(subset)
        
        # Compute Wilson CI
        ci_result = wilson_ci(
            successes=int(false_refusals),
            trials=int(total_benign),
            confidence=confidence
        )
        
        # Also compute per-model statistics
        models = subset['model'].unique()
        n_models = len(models)
        
        results.append({
            'condition': condition,
            'frr': ci_result['proportion'],
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper'],
            'ci_margin': ci_result['margin'],
            'n_benign': total_benign,
            'n_false_refusals': int(false_refusals),
            'n_models': n_models,
            'models_tested': ', '.join(models),
            'confidence_level': confidence
        })
    
    return pd.DataFrame(results)


def plot_safety_frr_tradeoff(df: pd.DataFrame,
                             frr_df: pd.DataFrame,
                             output_dir: str = "metrics_output") -> None:
    """
    Generate Safety vs FRR tradeoff scatter plot.
    
    X-axis: False Refusal Rate (over-refusal)
    Y-axis: True Positive Rate (harmful detection)
    Ideal: Top-left (high TPR, low FRR)
    
    Args:
        df: Full evaluation DataFrame (for computing TPR)
        frr_df: FRR results from compute_false_refusal_rate()
        output_dir: Directory for saving plot
    """
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute TPR (harmful refusal rate) for each model-condition
    tpr_data = []
    for (model, condition), group in df.groupby(['model', 'evaluation_type']):
        harmful = group[group['is_harmful'] == 1]
        if len(harmful) > 0:
            tpr = harmful['simple_refusal'].mean()
            tpr_data.append({
                'model': model,
                'condition': condition,
                'tpr': tpr
            })
    
    tpr_df = pd.DataFrame(tpr_data)
    
    # Merge TPR with FRR
    plot_df = frr_df.merge(tpr_df, on=['model', 'condition'], how='inner')
    
    # Create figure (fig/ax interface for publication-quality control)
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # Color / marker by condition — from shared style module
    conditions = plot_df['condition'].unique()

    all_texts = []
    all_x, all_y = [], []

    for condition in conditions:
        subset = plot_df[plot_df['condition'] == condition]

        ax.scatter(
            subset['frr'],
            subset['tpr'],
            c=condition_color(condition),
            marker=condition_marker(condition),
            s=100,
            alpha=0.85,
            label=condition_label(condition),
            edgecolors='white',
            linewidths=0.6,
            zorder=3,
        )

        for _, row in subset.iterrows():
            all_x.append(row['frr'])
            all_y.append(row['tpr'])
            all_texts.append(
                ax.text(row['frr'], row['tpr'],
                        short_model_name(row['model']),
                        fontsize=FONT_SIZE_ANNOT, alpha=0.85)
            )

    # Reference lines
    ax.axhline(y=0.9, color='#2CA02C', linestyle='--', lw=0.9, alpha=0.4,
               label='Target Safety (90% TPR)')
    ax.axvline(x=0.1, color='#FF7F0E', linestyle='--', lw=0.9, alpha=0.4,
               label='Acceptable FRR (10%)')

    # Ideal region shading
    ax.fill_between([0, 0.1], 0.9, 1.0, alpha=0.08, color='#2CA02C',
                    label='Ideal Region')

    ax.set_xlabel("False Refusal Rate (Over-Refusal on Benign)")
    ax.set_ylabel("True Positive Rate (Harmful Detection)")
    ax.set_title("Safety\u2013Utility Tradeoff: Harmful Detection vs Over-Refusal")

    # Axis limits with padding
    ax.set_xlim(-0.02, plot_df['frr'].max() + 0.1)
    ax.set_ylim(max(0, plot_df['tpr'].min() - 0.1), 1.05)

    style_axes(ax, grid='both')
    format_legend(ax, loc='lower left')

    # Auto-repel labels so they don't overlap
    if _adjust_text is not None and all_texts:
        _adjust_text(all_texts, x=all_x, y=all_y, ax=ax,
                     arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5),
                     expand=(1.5, 1.5), force_text=(0.8, 0.8))

    plt.tight_layout()
    fig.savefig(output_path / "safety_frr_tradeoff.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_frr_by_condition(frr_condition_df: pd.DataFrame,
                          output_dir: str = "metrics_output") -> None:
    """
    Generate bar chart of FRR by condition with error bars.
    
    Args:
        frr_condition_df: FRR aggregated by condition
        output_dir: Directory for saving plot
    """
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sort by FRR for better visualization
    plot_df = frr_condition_df.sort_values('frr')
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    
    # Color palette — shared style module
    bar_colors = [condition_color(c) for c in plot_df['condition']]

    # Create bars
    x_pos = np.arange(len(plot_df))
    bars = ax.bar(x_pos, plot_df['frr'],
                   color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add error bars (Wilson CI)
    errors_lower = plot_df['frr'] - plot_df['ci_lower']
    errors_upper = plot_df['ci_upper'] - plot_df['frr']
    ax.errorbar(x_pos, plot_df['frr'],
                yerr=[errors_lower, errors_upper],
                fmt='none', ecolor='#333333', capsize=5, capthick=1.2, alpha=0.9)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        height = row['frr']
        ax.text(i, height + row['ci_margin'] + 0.02,
                f"{height:.1%}\n[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]",
                ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel("Defense Condition", fontweight='bold')
    ax.set_ylabel("False Refusal Rate", fontweight='bold')
    ax.set_title("")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([condition_label(c) for c in plot_df['condition']])
    ax.set_ylim(0, 0.24)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Reference threshold line
    ax.axhline(y=0.1, color='#FF7F0E', linestyle='--', linewidth=1.2, alpha=0.6,
               label='Acceptable Threshold (10%)')

    style_axes(ax, grid='y')
    format_legend(ax)

    plt.tight_layout()
    fig.savefig(output_path / "frr_by_condition.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_frr_by_model_condition(frr_df: pd.DataFrame,
                                output_dir: str = "metrics_output") -> None:
    """
    Generate grouped bar chart: FRR for each model across conditions.
    
    Args:
        frr_df: FRR results per model-condition
        output_dir: Directory for saving plot
    """
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pivot for grouped bar chart
    pivot_df = frr_df.pivot(index='model', columns='condition', values='frr')
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    
    # Plot grouped bars — use shared condition colours in column order
    ordered_cols = list(pivot_df.columns)
    bar_palette = [condition_color(c) for c in ordered_cols]
    pivot_df.plot(kind='bar', ax=ax, width=0.8,
                  color=bar_palette,
                  edgecolor='white', linewidth=0.5, alpha=0.85)
    
    # Customize
    ax.set_xlabel("Model", fontweight='bold')
    ax.set_ylabel("False Refusal Rate", fontweight='bold')
    ax.set_title("Over-Refusal Across Models and Conditions")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(title='Condition', fontsize=FONT_SIZE_LEGEND,
              title_fontsize=FONT_SIZE_LEGEND)
    style_axes(ax, grid='y')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    fig.savefig(output_path / "frr_by_model_condition.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def print_frr_summary(frr_df: pd.DataFrame, 
                     frr_condition_df: pd.DataFrame) -> None:
    """
    Print formatted FRR summary to console.
    
    Args:
        frr_df: Per-model-condition FRR results
        frr_condition_df: Per-condition aggregated FRR results
    """
    print("\n" + "=" * 100)
    print("FALSE REFUSAL RATE ANALYSIS")
    print("=" * 100 + "\n")
    
    print("Definition: FRR = % of benign prompts incorrectly refused (over-refusal)")
    print("Lower FRR = Better utility preservation\n")
    
    # Overall statistics
    avg_frr = frr_df['frr'].mean()
    min_frr = frr_df['frr'].min()
    max_frr = frr_df['frr'].max()
    
    print("Overall Statistics:")
    print(f"  Average FRR: {avg_frr:.1%}")
    print(f"  Range: {min_frr:.1%} to {max_frr:.1%}")
    print(f"  Configurations analyzed: {len(frr_df)}\n")
    
    # Best and worst configurations
    best = frr_df.loc[frr_df['frr'].idxmin()]
    worst = frr_df.loc[frr_df['frr'].idxmax()]
    
    print("Best Configuration (Lowest Over-Refusal):")
    print(f"  {best['model']} ({best['condition']})")
    print(f"  FRR: {best['frr']:.1%} [{best['ci_lower']:.1%}, {best['ci_upper']:.1%}]")
    print(f"  False refusals: {best['n_false_refusals']}/{best['n_benign']} benign prompts\n")
    
    print("Worst Configuration (Highest Over-Refusal):")
    print(f"  {worst['model']} ({worst['condition']})")
    print(f"  FRR: {worst['frr']:.1%} [{worst['ci_lower']:.1%}, {worst['ci_upper']:.1%}]")
    print(f"  False refusals: {worst['n_false_refusals']}/{worst['n_benign']} benign prompts\n")
    
    # Per-condition comparison
    print("=" * 100)
    print("FRR BY CONDITION (Aggregated Across Models)")
    print("=" * 100)
    
    display_cond = frr_condition_df.copy()
    display_cond = display_cond.sort_values('frr')
    
    for _, row in display_cond.iterrows():
        print(f"\n{row['condition'].upper()}:")
        print(f"  FRR: {row['frr']:.1%} (95% CI: [{row['ci_lower']:.1%}, {row['ci_upper']:.1%}])")
        print(f"  False refusals: {row['n_false_refusals']}/{row['n_benign']} benign prompts")
        print(f"  Models tested: {row['n_models']} ({row['models_tested']})")
    
    # Detailed per-model-condition table
    print("\n" + "=" * 100)
    print("DETAILED FRR BY MODEL AND CONDITION")
    print("=" * 100)
    
    display_df = frr_df.copy()
    display_df['frr_formatted'] = display_df.apply(
        lambda row: f"{row['frr']:.1%} [{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]",
        axis=1
    )
    display_df['refusals'] = display_df.apply(
        lambda row: f"{row['n_false_refusals']}/{row['n_benign']}",
        axis=1
    )
    
    print(display_df[['model', 'condition', 'frr_formatted', 'refusals']].to_string(index=False))
    print("=" * 100 + "\n")


def generate_frr_report(df: pd.DataFrame,
                       output_dir: str = "metrics_output",
                       confidence: float = 0.95) -> Dict:
    """
    Generate comprehensive False Refusal Rate report.
    
    Computes FRR with Wilson CIs and generates all visualizations.
    
    Args:
        df: Evaluation results DataFrame
        output_dir: Directory for saving outputs
        confidence: Confidence level for CIs
    
    Returns:
        Dictionary with:
            - frr_df: Per-model-condition FRR metrics
            - frr_condition_df: Per-condition aggregated FRR
            - figures: Generated visualizations
    """
    print("\n" + "=" * 100)
    print("🔍 GENERATING FALSE REFUSAL RATE REPORT")
    print("=" * 100)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "tables").mkdir(exist_ok=True)
    (output_path / "figures").mkdir(exist_ok=True)
    
    # 1. Compute FRR per model-condition
    print("\n1. Computing FRR with Wilson confidence intervals...")
    frr_df = compute_false_refusal_rate(df, confidence=confidence)
    
    # 2. Compute FRR aggregated by condition
    print("2. Aggregating FRR by condition...")
    frr_condition_df = compute_frr_by_condition(df, confidence=confidence)
    
    # 3. Save tables
    print("3. Saving FRR tables...")
    frr_df.to_csv(output_path / "tables" / "false_refusal_rate.csv", index=False)
    frr_condition_df.to_csv(output_path / "tables" / "false_refusal_rate_by_condition.csv", index=False)
    
    # 4. Generate visualizations
    print("4. Generating visualizations...")
    plot_safety_frr_tradeoff(df, frr_df, output_dir)
    plot_frr_by_condition(frr_condition_df, output_dir)
    plot_frr_by_model_condition(frr_df, output_dir)
    
    # 5. Print summary
    print_frr_summary(frr_df, frr_condition_df)
    
    print(f"\n✓ False Refusal Rate analysis complete")
    print(f"  - FRR metrics: {output_path / 'tables' / 'false_refusal_rate.csv'}")
    print(f"  - Condition aggregation: {output_path / 'tables' / 'false_refusal_rate_by_condition.csv'}")
    print(f"  - Safety-FRR tradeoff: {output_path / 'figures' / 'safety_frr_tradeoff.png'}")
    print(f"  - FRR by condition: {output_path / 'figures' / 'frr_by_condition.png'}")
    print(f"  - FRR by model-condition: {output_path / 'figures' / 'frr_by_model_condition.png'}\n")
    
    return {
        'frr_df': frr_df,
        'frr_condition_df': frr_condition_df,
        'metrics': {
            'avg_frr': frr_df['frr'].mean(),
            'min_frr': frr_df['frr'].min(),
            'max_frr': frr_df['frr'].max(),
            'n_configurations': len(frr_df)
        }
    }


def main():
    """
    Main execution function for standalone use.
    
    Usage:
        python false_refusal_rate.py
    """
    import os
    
    # Default path
    results_csv = "results/sample_harmful_benign_analysis.csv"
    output_dir = "metrics_output"
    
    # Check if results file exists
    if not os.path.exists(results_csv):
        print(f"\n❌ Error: Results file not found: {results_csv}")
        return None
    
    # Load data
    print(f"\n📊 Loading evaluation results from: {results_csv}")
    df = pd.read_csv(results_csv)
    
    # Convert boolean columns if needed
    if df['is_harmful'].dtype == 'object':
        df['is_harmful'] = df['is_harmful'].map({'TRUE': 1, 'True': 1, True: 1, 
                                                   'FALSE': 0, 'False': 0, False: 0})
    if df['simple_refusal'].dtype == 'object':
        df['simple_refusal'] = df['simple_refusal'].map({'TRUE': 1, 'True': 1, True: 1,
                                                           'FALSE': 0, 'False': 0, False: 0})
    
    print(f"✓ Loaded {len(df)} evaluation results")
    print(f"  Benign prompts: {(df['is_harmful'] == 0).sum()}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Conditions: {df['evaluation_type'].unique().tolist()}")
    
    # Generate report
    results = generate_frr_report(df, output_dir)
    
    return results


if __name__ == "__main__":
    main()

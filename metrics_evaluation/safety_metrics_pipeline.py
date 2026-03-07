"""
Safety Metrics Pipeline

End-to-end pipeline for computing modern safety metrics including:
- TPR @ 1% FPR (capability-aware detection)
- Wilson confidence intervals
- Safety-utility Pareto analysis
- Comparative analysis across models and conditions

Input: CSV files from evaluation pipeline (harmful/benign analysis)
Output: Comprehensive metrics, tables, and visualizations

Reference: Research-Roadmap.md Milestone 4 (Weeks 7-8)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    from plot_style import apply_publication_style
except ImportError:
    from metrics_evaluation.plot_style import apply_publication_style

# Import TPR/FPR calculation functions
try:
    from tpr_fpr import (
        compute_tpr_at_fpr,
        compute_tpr_at_multiple_fprs,
        plot_roc_curve,
        compare_conditions_roc,
        print_tpr_fpr_summary
    )
except ImportError:
    from metrics_evaluation.tpr_fpr import (
        compute_tpr_at_fpr,
        compute_tpr_at_multiple_fprs,
        plot_roc_curve,
        compare_conditions_roc,
        print_tpr_fpr_summary
    )

# Import confidence interval functions
try:
    from confidence_intervals import (
        wilson_ci,
        compute_metric_ci,
        compute_all_cis,
        compute_model_condition_cis,
        print_ci_summary,
        format_ci_string
    )
except ImportError:
    from metrics_evaluation.confidence_intervals import (
        wilson_ci,
        compute_metric_ci,
        compute_all_cis,
        compute_model_condition_cis,
        print_ci_summary,
        format_ci_string
    )

# Import safety-utility frontier functions
try:
    from safety_utility import (
        compute_safety_utility_metrics,
        identify_pareto_optimal,
        plot_safety_utility_frontier,
        plot_condition_comparison,
        print_pareto_analysis,
        generate_safety_utility_report
    )
except ImportError:
    from metrics_evaluation.safety_utility import (
        compute_safety_utility_metrics,
        identify_pareto_optimal,
        plot_safety_utility_frontier,
        plot_condition_comparison,
        print_pareto_analysis,
        generate_safety_utility_report
    )

# Import multi-turn degradation analysis functions
try:
    from multi_turn_analysis import (
        analyze_multiturn_degradation,
        analyze_by_model_condition,
        compute_degradation_metrics,
        print_degradation_summary,
        generate_multiturn_report
    )
except ImportError:
    from metrics_evaluation.multi_turn_analysis import (
        analyze_multiturn_degradation,
        analyze_by_model_condition,
        compute_degradation_metrics,
        print_degradation_summary,
        generate_multiturn_report
    )

# Import false refusal rate analysis functions
try:
    from false_refusal_rate import (
        compute_false_refusal_rate,
        compute_frr_by_condition,
        plot_safety_frr_tradeoff,
        plot_frr_by_condition,
        plot_frr_by_model_condition,
        print_frr_summary,
        generate_frr_report
    )
except ImportError:
    from metrics_evaluation.false_refusal_rate import (
        compute_false_refusal_rate,
        compute_frr_by_condition,
        plot_safety_frr_tradeoff,
        plot_frr_by_condition,
        plot_frr_by_model_condition,
        print_frr_summary,
        generate_frr_report
    )


class SafetyMetricsPipeline:
    """
    Comprehensive safety metrics analysis pipeline.
    
    Processes evaluation results and computes modern safety metrics
    following StrongREJECT and WalledEval best practices.
    """
    
    def __init__(self, results_csv: str, output_dir: str = "metrics_output", 
                 multiturn_csv: Optional[str] = None):
        """
        Initialize pipeline.
        
        Args:
            results_csv: Path to harmful/benign analysis CSV
            output_dir: Directory for saving outputs
            multiturn_csv: Optional path to multi-turn analysis CSV
        """
        self.results_csv = results_csv
        self.multiturn_csv = multiturn_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        
        # Load data
        print(f"\n📊 Loading results from: {results_csv}")
        self.df = pd.read_csv(results_csv)
        
        # Load multi-turn data if provided
        self.multiturn_df = None
        if multiturn_csv and os.path.exists(multiturn_csv):
            print(f"📊 Loading multi-turn results from: {multiturn_csv}")
            self.multiturn_df = pd.read_csv(multiturn_csv)
            print(f"✓ Loaded {len(self.multiturn_df)} turn records from {len(self.multiturn_df['conversation_id'].unique())} conversations")
        
        # Preprocess
        self._preprocess_data()
        
        print(f"✓ Loaded {len(self.df)} evaluation results")
        print(f"  Models: {self.df['model'].unique().tolist()}")
        print(f"  Conditions: {self.df['evaluation_type'].unique().tolist()}")
        print(f"  Prompt types: {self.df['prompt_type'].unique().tolist()}\n")
    
    def _preprocess_data(self):
        """Preprocess data for analysis."""
        # Convert boolean columns
        if self.df['is_harmful'].dtype == 'object':
            self.df['is_harmful'] = self.df['is_harmful'].map({
                'TRUE': 1, 'True': 1, 'true': 1, True: 1,
                'FALSE': 0, 'False': 0, 'false': 0, False: 0
            })
        
        if self.df['simple_refusal'].dtype == 'object':
            self.df['simple_refusal'] = self.df['simple_refusal'].map({
                'TRUE': 1, 'True': 1, 'true': 1, True: 1,
                'FALSE': 0, 'False': 0, 'false': 0, False: 0
            })
        
        # Ensure numeric columns
        numeric_cols = ['strongreject_score', 'safety', 'utility', 'response_length']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def compute_tpr_fpr_metrics(self, target_fprs: List[float] = [0.01, 0.05, 0.10]) -> pd.DataFrame:
        """
        Compute TPR @ FPR metrics for all model-condition combinations.
        
        Args:
            target_fprs: List of FPR thresholds to evaluate
        
        Returns:
            DataFrame with TPR@FPR metrics for each model-condition
        """
        print("\n" + "=" * 80)
        print("COMPUTING TPR @ FPR METRICS")
        print("=" * 80 + "\n")
        
        results = []
        
        # Group by model and condition
        for (model, condition), group in self.df.groupby(['model', 'evaluation_type']):
            print(f"Processing: {model} - {condition.upper()}")
            
            # Extract labels and scores
            y_true = group['is_harmful'].values
            y_scores = group['safety'].values  # Higher = safer/more likely refused
            
            # Skip if insufficient data
            if len(y_true) < 10 or len(np.unique(y_true)) < 2:
                print(f"  ⚠️ Insufficient data (n={len(y_true)}, classes={len(np.unique(y_true))})")
                continue
            
            # Compute metrics at each FPR threshold
            for target_fpr in target_fprs:
                metrics = compute_tpr_at_fpr(y_true, y_scores, target_fpr)
                
                results.append({
                    'model': model,
                    'condition': condition,
                    'target_fpr': target_fpr,
                    'actual_fpr': metrics['fpr'],
                    'tpr': metrics['tpr'],
                    'threshold': metrics['threshold'],
                    'auc': metrics['auc'],
                    'n_samples': len(y_true),
                    'n_harmful': int(y_true.sum()),
                    'n_benign': int((1 - y_true).sum())
                })
                
                if target_fpr == 0.01:  # Print 1% FPR result
                    print(f"  TPR @ 1% FPR: {metrics['tpr']:.1%} (AUC: {metrics['auc']:.4f})")
            
            print()
        
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = self.output_dir / "tables" / "tpr_fpr_metrics.csv"
        results_df.to_csv(output_path, index=False)
        print(f"✓ Saved TPR@FPR metrics to: {output_path}\n")
        
        return results_df
    
    def create_summary_table(self, tpr_fpr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create publication-ready summary table of key metrics.
        
        Args:
            tpr_fpr_df: DataFrame from compute_tpr_fpr_metrics
        
        Returns:
            Formatted summary DataFrame
        """
        print("=" * 80)
        print("CREATING SUMMARY TABLE")
        print("=" * 80 + "\n")
        
        # Filter to 1% FPR results
        summary = tpr_fpr_df[tpr_fpr_df['target_fpr'] == 0.01].copy()
        
        # Compute additional metrics per model-condition
        additional_metrics = []
        
        for (model, condition), group in self.df.groupby(['model', 'evaluation_type']):
            harmful = group[group['is_harmful'] == 1]
            benign = group[group['is_harmful'] == 0]
            
            # Refusal rates
            harmful_refusal_rate = harmful['simple_refusal'].mean() if len(harmful) > 0 else 0
            benign_refusal_rate = benign['simple_refusal'].mean() if len(benign) > 0 else 0
            
            # Safety scores (avg across all prompts; higher = safer)
            avg_safety = group['safety'].mean() if 'safety' in group.columns else 0
            harmful_avg_safety = harmful['safety'].mean() if len(harmful) > 0 else 0
            
            # Utility preservation
            avg_utility = group['utility'].mean() if 'utility' in group.columns else 0
            
            additional_metrics.append({
                'model': model,
                'condition': condition,
                'harmful_refusal_rate': harmful_refusal_rate,
                'benign_refusal_rate': benign_refusal_rate,
                'false_refusal_rate': benign_refusal_rate,  # Same as benign refusal
                'avg_safety': avg_safety,
                'harmful_avg_safety': harmful_avg_safety,
                'avg_utility': avg_utility
            })
        
        additional_df = pd.DataFrame(additional_metrics)
        
        # Merge with TPR@FPR results
        summary = summary.merge(additional_df, on=['model', 'condition'], how='left')
        
        # Compute Pareto score: Avg(safety, 1-false_refusal)
        summary['pareto_score'] = (summary['tpr'] + (1 - summary['false_refusal_rate'])) / 2
        
        # Select and reorder columns for publication
        columns_order = [
            'model', 'condition', 'tpr', 'actual_fpr', 'auc',
            'harmful_refusal_rate', 'false_refusal_rate',
            'avg_safety', 'avg_utility', 'pareto_score',
            'n_samples', 'n_harmful', 'n_benign'
        ]
        
        summary = summary[columns_order]
        
        # Save to CSV
        output_path = self.output_dir / "tables" / "summary_metrics.csv"
        summary.to_csv(output_path, index=False)
        print(f"✓ Saved summary table to: {output_path}")
        
        # Print formatted table
        self._print_summary_table(summary)
        
        return summary
    
    def _print_summary_table(self, summary_df: pd.DataFrame):
        """Print formatted summary table to console."""
        print("\n" + "=" * 120)
        print("SUMMARY: TPR @ 1% FPR and Key Safety Metrics")
        print("=" * 120)
        
        # Format for display
        display_df = summary_df.copy()
        display_df['tpr'] = display_df['tpr'].apply(lambda x: f"{x:.1%}")
        display_df['auc'] = display_df['auc'].apply(lambda x: f"{x:.3f}")
        display_df['harmful_refusal_rate'] = display_df['harmful_refusal_rate'].apply(lambda x: f"{x:.1%}")
        display_df['false_refusal_rate'] = display_df['false_refusal_rate'].apply(lambda x: f"{x:.1%}")
        display_df['avg_safety'] = display_df['avg_safety'].apply(lambda x: f"{x:.4f}")
        display_df['pareto_score'] = display_df['pareto_score'].apply(lambda x: f"{x:.3f}")
        
        # Print selected columns
        cols_to_print = ['model', 'condition', 'tpr', 'auc', 'harmful_refusal_rate', 
                        'false_refusal_rate', 'pareto_score']
        
        print(display_df[cols_to_print].to_string(index=False))
        print("=" * 120 + "\n")
        
        print("Column Definitions:")
        print("  - TPR: True Positive Rate @ 1% FPR (% harmful prompts refused at 1% benign refusal)")
        print("  - AUC: Area Under ROC Curve (overall discrimination ability)")
        print("  - Harmful Refusal Rate: % of harmful prompts refused")
        print("  - False Refusal Rate: % of benign prompts incorrectly refused (over-refusal)")
        print("  - Avg Safety Score: Mean safety score across all prompts (higher = safer responses)")
        print("  - Pareto Score: Avg(TPR, 1-FalseRefusal) - balances safety and utility\n")
    
    def plot_roc_curves(self):
        """Generate ROC curve plots for all models and conditions."""
        print("=" * 80)
        print("GENERATING ROC CURVES")
        print("=" * 80 + "\n")
        
        # Plot for each model
        for model in self.df['model'].unique():
            model_df = self.df[self.df['model'] == model]
            
            # Prepare data for comparison plot
            conditions_data = {}
            
            for condition in model_df['evaluation_type'].unique():
                cond_df = model_df[model_df['evaluation_type'] == condition]
                y_true = cond_df['is_harmful'].values
                y_scores = cond_df['safety'].values
                
                if len(np.unique(y_true)) >= 2:
                    conditions_data[condition] = (y_true, y_scores)
            
            # Create comparison plot
            if len(conditions_data) > 1:
                save_path = self.output_dir / "figures" / f"roc_comparison_{model.replace(' ', '_').replace('.', '_')}.png"
                fig = compare_conditions_roc(
                    conditions_data, 
                    model_name=model,
                    target_fpr=0.01,
                    save_path=str(save_path)
                )
                plt.close(fig)
        
        print("\n✓ ROC curves generated\n")
    
    def compute_confidence_intervals(self, confidence: float = 0.95) -> pd.DataFrame:
        """
        Compute Wilson confidence intervals for all metrics.
        
        Args:
            confidence: Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            DataFrame with confidence intervals for all model-condition pairs
        """
        print("=" * 80)
        print("COMPUTING CONFIDENCE INTERVALS")
        print("=" * 80 + "\n")
        
        # Compute CIs for all model-condition combinations
        ci_df = compute_all_cis(self.df, confidence=confidence)
        
        # Save to CSV
        output_path = self.output_dir / "tables" / "confidence_intervals.csv"
        ci_df.to_csv(output_path, index=False)
        print(f"✓ Saved confidence intervals to: {output_path}")
        
        # Print formatted summary
        print_ci_summary(ci_df)
        
        return ci_df
    
    def compute_safety_utility_frontier(self) -> Dict:
        """
        Compute safety-utility Pareto frontier analysis.
        
        Analyzes the tradeoff between safety (harmful refusal rate) and
        utility (capability preserved = 1 - false refusal rate) for each
        model-condition combination.
        
        Returns:
            Dictionary containing:
                - metrics_df: Safety-utility metrics for all model-conditions
                - pareto_df: Pareto optimal configurations
                - figures: Generated matplotlib figures
        """
        print("=" * 80)
        print("COMPUTING SAFETY-UTILITY PARETO FRONTIER")
        print("=" * 80 + "\n")
        
        # Generate full safety-utility report using the dedicated module
        results = generate_safety_utility_report(self.df, str(self.output_dir))
        
        print(f"\n✓ Safety-utility analysis complete")
        print(f"  - Metrics saved to: {self.output_dir / 'tables' / 'safety_utility_metrics.csv'}")
        print(f"  - Pareto frontier plot: {self.output_dir / 'figures' / 'safety_utility_pareto_frontier.png'}")
        print(f"  - Comparison plots: {self.output_dir / 'figures' / 'safety_utility_by_model.png'}")
        
        return results
    
    def compute_multiturn_degradation(self) -> Optional[Dict]:
        """
        Compute multi-turn degradation analysis if multi-turn data is available.
        
        Analyzes how safety degrades across conversation turns following
        Research-Roadmap.md Section 7.3 specification.
        
        Returns:
            Dictionary containing:
                - overall_results: Overall degradation statistics
                - comparison_df: Degradation by model and condition
                - degradation_metrics: Comprehensive degradation metrics
            Or None if no multi-turn data is available
        """
        if self.multiturn_df is None:
            print("\n⚠ Multi-turn data not available. Skipping multi-turn degradation analysis.")
            print("  To include multi-turn analysis, provide multiturn_csv parameter.\n")
            return None
        
        print("=" * 80)
        print("COMPUTING MULTI-TURN DEGRADATION ANALYSIS")
        print("=" * 80 + "\n")
        
        # Generate comprehensive multi-turn report
        results = generate_multiturn_report(
            self.multiturn_df,
            output_dir=str(self.output_dir)
        )
        
        return results
    
    def compute_false_refusal_analysis(self, confidence: float = 0.95) -> Dict:
        """
        Compute False Refusal Rate (FRR) analysis with Wilson confidence intervals.
        
        Measures over-refusal: how often models incorrectly refuse benign prompts.
        Critical for assessing safety-utility tradeoff.
        
        Args:
            confidence: Confidence level for Wilson CIs (default: 0.95)
        
        Returns:
            Dictionary containing:
                - frr_df: Per-model-condition FRR metrics
                - frr_condition_df: Per-condition aggregated FRR
                - metrics: Summary statistics
        """
        print("=" * 80)
        print("COMPUTING FALSE REFUSAL RATE ANALYSIS")
        print("=" * 80 + "\n")
        
        # Generate comprehensive FRR report
        results = generate_frr_report(
            self.df,
            output_dir=str(self.output_dir),
            confidence=confidence
        )
        
        return results
    
    def generate_report(self) -> Dict:
        """
        Run complete analysis pipeline and generate comprehensive report.
        
        Returns:
            Dictionary containing all computed metrics and analyses
        """
        # Apply publication-quality rcParams globally before any figures are created
        apply_publication_style()

        print("\n" + "=" * 80)
        print("🚀 STARTING SAFETY METRICS ANALYSIS PIPELINE")
        print("=" * 80)
        
        report = {}
        
        # 1. Compute TPR @ FPR metrics
        tpr_fpr_df = self.compute_tpr_fpr_metrics()
        report['tpr_fpr_metrics'] = tpr_fpr_df
        
        # 2. Create summary table
        summary_df = self.create_summary_table(tpr_fpr_df)
        report['summary'] = summary_df
        
        # 3. Compute confidence intervals
        ci_df = self.compute_confidence_intervals(confidence=0.95)
        report['confidence_intervals'] = ci_df
        
        # 4. Generate ROC curves
        self.plot_roc_curves()
        
        # 5. Compute safety-utility Pareto frontier
        safety_utility_results = self.compute_safety_utility_frontier()
        report['safety_utility'] = safety_utility_results
        
        # 6. Compute multi-turn degradation (if data available)
        multiturn_results = self.compute_multiturn_degradation()
        if multiturn_results is not None:
            report['multiturn_degradation'] = multiturn_results
        
        # 7. Compute False Refusal Rate analysis
        frr_results = self.compute_false_refusal_analysis(confidence=0.95)
        report['false_refusal_rate'] = frr_results
        
        # 8. Key findings
        print("=" * 80)
        print("KEY FINDINGS")
        print("=" * 80 + "\n")
        
        # Find best performing condition per model at 1% FPR
        best_by_model = summary_df.loc[summary_df.groupby('model')['tpr'].idxmax()]
        
        print("Best Performing Condition per Model (TPR @ 1% FPR):")
        for _, row in best_by_model.iterrows():
            print(f"  {row['model']:25s} → {row['condition']:15s} "
                  f"(TPR: {row['tpr']:.1%}, Pareto: {row['pareto_score']:.3f})")
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nOutputs saved to: {self.output_dir}")
        print(f"  - Tables: {self.output_dir / 'tables'}")
        print(f"  - Figures: {self.output_dir / 'figures'}\n")
        
        return report


def main():
    """
    Main execution function for standalone use.
    
    Usage:
        python safety_metrics_pipeline.py
    """
    # Default paths - modify as needed
    results_csv = "results/evaluation_harmful_benign_analysis.csv"
    multiturn_csv = "results/evaluation_multi_turn_analysis.csv"
    output_dir = "metrics_output"
    
    # Check if results file exists
    if not os.path.exists(results_csv):
        print(f"\n❌ Error: Results file not found: {results_csv}")
        print("\nUsage:")
        print("  1. Ensure evaluation results CSV exists")
        print("  2. Run: python safety_metrics_pipeline.py")
        print("  3. Or import and use SafetyMetricsPipeline class\n")
        return
    
    # Check for multi-turn data (optional)
    if not os.path.exists(multiturn_csv):
        print(f"\n⚠ Note: Multi-turn analysis CSV not found: {multiturn_csv}")
        print("  Multi-turn degradation analysis will be skipped.")
        multiturn_csv = None
    
    # Run pipeline
    pipeline = SafetyMetricsPipeline(results_csv, output_dir, multiturn_csv=multiturn_csv)
    report = pipeline.generate_report()
    
    return report


if __name__ == "__main__":
    main()

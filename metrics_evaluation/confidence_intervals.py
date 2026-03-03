"""
Wilson Confidence Intervals for Safety Metrics

Implements Wilson confidence intervals for binary proportions and derived metrics.
More accurate than normal approximation for small sample sizes.

Supported Metrics:
- Refusal rates (harmful, benign, overall)
- TPR (True Positive Rate)
- FPR (False Positive Rate)
- False refusal rate (over-refusal on benign prompts)
- Any binary proportion metric

Reference: Research-Roadmap.md Section 7.2.2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from statsmodels.stats.proportion import proportion_confint


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Compute Wilson confidence interval for a binary proportion.
    
    More accurate than normal approximation, especially for:
    - Small sample sizes (n < 30)
    - Proportions near 0 or 1
    - Unbalanced datasets
    
    Args:
        successes: Number of successes (e.g., refusals, true positives)
        trials: Total number of trials (e.g., total prompts)
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Dictionary with:
        - proportion: Point estimate (successes / trials)
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - margin: Half-width of CI (margin of error)
        - n_successes: Input successes count
        - n_trials: Input trials count
    
    Example:
        >>> # Model refused 180 out of 200 harmful prompts
        >>> ci = wilson_ci(180, 200, confidence=0.95)
        >>> print(f"Refusal Rate: {ci['proportion']:.1%}")
        >>> print(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")
        Refusal Rate: 90.0%
        95% CI: [85.2%, 93.7%]
    """
    if trials == 0:
        return {
            'proportion': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'margin': 0.0,
            'n_successes': successes,
            'n_trials': trials
        }
    
    # Compute Wilson confidence interval
    lower, upper = proportion_confint(
        count=successes,
        nobs=trials,
        alpha=1 - confidence,
        method='wilson'
    )
    
    proportion = successes / trials
    
    return {
        'proportion': proportion,
        'ci_lower': lower,
        'ci_upper': upper,
        'margin': (upper - lower) / 2,
        'n_successes': successes,
        'n_trials': trials
    }


def compute_metric_ci(data: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    """
    Compute confidence interval for a binary metric from pandas Series.
    
    Args:
        data: pandas Series of binary values (0/1, True/False)
        confidence: Confidence level (default: 0.95)
    
    Returns:
        Dictionary with CI statistics
    
    Example:
        >>> refusals = df[df['is_harmful'] == 1]['simple_refusal']
        >>> ci = compute_metric_ci(refusals)
        >>> print(f"Harmful Refusal Rate: {ci['proportion']:.1%} ± {ci['margin']:.1%}")
    """
    # Convert to numeric if needed
    if data.dtype == 'object' or data.dtype == 'bool':
        data = data.map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0})
        data = pd.to_numeric(data, errors='coerce')
    
    # Remove NaN values
    data = data.dropna()
    
    successes = int(data.sum())
    trials = len(data)
    
    return wilson_ci(successes, trials, confidence)


def compute_tpr_fpr_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                       confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Compute confidence intervals for TPR and FPR separately.
    
    Args:
        y_true: True binary labels (1=harmful, 0=benign)
        y_pred: Predicted binary labels (1=refused, 0=complied)
        confidence: Confidence level
    
    Returns:
        Dictionary with 'tpr' and 'fpr' CIs
    
    Example:
        >>> y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1])
        >>> y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 1])
        >>> cis = compute_tpr_fpr_ci(y_true, y_pred)
        >>> print(f"TPR: {cis['tpr']['proportion']:.1%} [{cis['tpr']['ci_lower']:.1%}, {cis['tpr']['ci_upper']:.1%}]")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # TPR: Of all harmful (y_true=1), what fraction were refused (y_pred=1)?
    harmful_mask = (y_true == 1)
    harmful_total = harmful_mask.sum()
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    
    # FPR: Of all benign (y_true=0), what fraction were refused (y_pred=1)?
    benign_mask = (y_true == 0)
    benign_total = benign_mask.sum()
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()
    
    return {
        'tpr': wilson_ci(int(true_positives), int(harmful_total), confidence),
        'fpr': wilson_ci(int(false_positives), int(benign_total), confidence)
    }


def compute_model_condition_cis(df: pd.DataFrame, 
                                model: str, 
                                condition: str,
                                confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute comprehensive confidence intervals for a model-condition pair.
    
    Args:
        df: Evaluation results DataFrame
        model: Model name
        condition: Evaluation condition (baseline, safedecoding, llamaguard3)
        confidence: Confidence level
    
    Returns:
        DataFrame with metric CIs
    
    Metrics computed:
    - harmful_refusal_rate: % harmful prompts refused
    - benign_refusal_rate: % benign prompts refused (false refusals)
    - overall_refusal_rate: % all prompts refused
    - tpr: True Positive Rate (from binary predictions)
    - fpr: False Positive Rate (from binary predictions)
    """
    # Filter data
    subset = df[(df['model'] == model) & (df['evaluation_type'] == condition)].copy()
    
    if len(subset) == 0:
        return pd.DataFrame()
    
    results = []
    
    # 1. Harmful refusal rate
    harmful = subset[subset['is_harmful'] == 1]
    if len(harmful) > 0:
        harmful_ci = compute_metric_ci(harmful['simple_refusal'], confidence)
        results.append({
            'model': model,
            'condition': condition,
            'metric': 'harmful_refusal_rate',
            **harmful_ci
        })
    
    # 2. Benign refusal rate (false refusals / over-refusal)
    benign = subset[subset['is_harmful'] == 0]
    if len(benign) > 0:
        benign_ci = compute_metric_ci(benign['simple_refusal'], confidence)
        results.append({
            'model': model,
            'condition': condition,
            'metric': 'false_refusal_rate',
            **benign_ci
        })
    
    # 3. Overall refusal rate
    overall_ci = compute_metric_ci(subset['simple_refusal'], confidence)
    results.append({
        'model': model,
        'condition': condition,
        'metric': 'overall_refusal_rate',
        **overall_ci
    })
    
    # 4. TPR and FPR (using binary predictions)
    y_true = subset['is_harmful'].values
    y_pred = subset['simple_refusal'].values
    
    tpr_fpr_cis = compute_tpr_fpr_ci(y_true, y_pred, confidence)
    
    results.append({
        'model': model,
        'condition': condition,
        'metric': 'tpr',
        **tpr_fpr_cis['tpr']
    })
    
    results.append({
        'model': model,
        'condition': condition,
        'metric': 'fpr',
        **tpr_fpr_cis['fpr']
    })
    
    return pd.DataFrame(results)


def compute_all_cis(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute confidence intervals for all model-condition combinations.
    
    Args:
        df: Full evaluation results DataFrame
        confidence: Confidence level
    
    Returns:
        DataFrame with all CIs
    """
    all_results = []
    
    for (model, condition), group in df.groupby(['model', 'evaluation_type']):
        cis = compute_model_condition_cis(df, model, condition, confidence)
        if not cis.empty:
            all_results.append(cis)
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.concat(all_results, ignore_index=True)


def format_ci_string(ci_dict: Dict[str, float], as_percentage: bool = True) -> str:
    """
    Format CI dictionary as readable string.
    
    Args:
        ci_dict: Dict from wilson_ci() or compute_metric_ci()
        as_percentage: Format as percentage (True) or decimal (False)
    
    Returns:
        Formatted string like "90.0% [85.2%, 93.7%]"
    
    Example:
        >>> ci = wilson_ci(180, 200)
        >>> print(format_ci_string(ci))
        90.0% [85.2%, 93.7%]
    """
    if as_percentage:
        return (f"{ci_dict['proportion']:.1%} "
                f"[{ci_dict['ci_lower']:.1%}, {ci_dict['ci_upper']:.1%}]")
    else:
        return (f"{ci_dict['proportion']:.4f} "
                f"[{ci_dict['ci_lower']:.4f}, {ci_dict['ci_upper']:.4f}]")


def print_ci_summary(ci_df: pd.DataFrame, model: Optional[str] = None, 
                    condition: Optional[str] = None):
    """
    Print formatted summary table of confidence intervals.
    
    Args:
        ci_df: DataFrame from compute_all_cis() or compute_model_condition_cis()
        model: Optional filter for specific model
        condition: Optional filter for specific condition
    """
    # Apply filters
    df = ci_df.copy()
    if model:
        df = df[df['model'] == model]
    if condition:
        df = df[df['condition'] == condition]
    
    if df.empty:
        print("No data to display")
        return
    
    print("\n" + "=" * 100)
    print("CONFIDENCE INTERVALS (95% Wilson Method)")
    print("=" * 100 + "\n")
    
    # Group by model and condition
    for (m, c), group in df.groupby(['model', 'condition']):
        print(f"\n{m} - {c.upper()}")
        print("-" * 100)
        print(f"{'Metric':<25} {'Value':<12} {'95% CI':<25} {'n':<10}")
        print("-" * 100)
        
        for _, row in group.iterrows():
            metric_name = row['metric'].replace('_', ' ').title()
            value = f"{row['proportion']:.1%}"
            ci = f"[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]"
            n = f"{row['n_successes']}/{row['n_trials']}"
            
            print(f"{metric_name:<25} {value:<12} {ci:<25} {n:<10}")
    
    print("\n" + "=" * 100 + "\n")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic Wilson CI
    print("Example 1: Basic Wilson CI Calculation")
    print("-" * 50)
    
    # Model refused 180 out of 200 harmful prompts
    refusal_ci = wilson_ci(180, 200, confidence=0.95)
    print(f"Refusal Rate: {refusal_ci['proportion']:.1%}")
    print(f"95% CI: [{refusal_ci['ci_lower']:.1%}, {refusal_ci['ci_upper']:.1%}]")
    print(f"Margin of Error: ±{refusal_ci['margin']:.1%}\n")
    
    # Example 2: TPR/FPR with CI
    print("Example 2: TPR/FPR with Confidence Intervals")
    print("-" * 50)
    
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])
    
    cis = compute_tpr_fpr_ci(y_true, y_pred)
    
    print(f"TPR: {format_ci_string(cis['tpr'])}")
    print(f"FPR: {format_ci_string(cis['fpr'])}")
    
    print("\n✓ Confidence interval module ready for use")

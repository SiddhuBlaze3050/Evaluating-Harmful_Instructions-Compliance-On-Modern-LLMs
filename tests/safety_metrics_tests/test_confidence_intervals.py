"""
Test Confidence Interval Calculations

Validates Wilson CI implementation using sample evaluation data.

Usage:
    python tests/safety_metrics_tests/test_confidence_intervals.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metrics_evaluation.confidence_intervals import (
    wilson_ci,
    compute_metric_ci,
    compute_tpr_fpr_ci,
    compute_model_condition_cis,
    compute_all_cis,
    format_ci_string,
    print_ci_summary
)


def test_basic_wilson_ci():
    """Test basic Wilson CI calculation."""
    
    print("\n" + "=" * 80)
    print("TEST 1: Basic Wilson CI Calculation")
    print("=" * 80 + "\n")
    
    # Example from Research Roadmap: 180/200 refusals
    ci = wilson_ci(180, 200, confidence=0.95)
    
    print(f"Input: 180 refusals out of 200 prompts")
    print(f"Refusal Rate: {ci['proportion']:.1%}")
    print(f"95% CI: [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")
    print(f"Margin of Error: ±{ci['margin']:.1%}")
    print(f"Formatted: {format_ci_string(ci)}")
    
    # Validation
    assert 0.85 <= ci['proportion'] <= 0.95, "Proportion should be 90%"
    assert ci['ci_lower'] < ci['proportion'] < ci['ci_upper'], "CI should contain proportion"
    assert ci['margin'] > 0, "Margin should be positive"
    
    print("\n✓ PASS: Basic Wilson CI calculation correct")
    return True


def test_tpr_fpr_ci():
    """Test TPR/FPR confidence intervals."""
    
    print("\n" + "=" * 80)
    print("TEST 2: TPR/FPR Confidence Intervals")
    print("=" * 80 + "\n")
    
    # Create test data
    # 10 samples: 5 harmful (4 refused), 5 benign (1 incorrectly refused)
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    
    cis = compute_tpr_fpr_ci(y_true, y_pred)
    
    print("Test Data:")
    print(f"  Harmful samples: 5")
    print(f"  True Positives (harmful & refused): 4")
    print(f"  Benign samples: 5")
    print(f"  False Positives (benign & refused): 1")
    
    print(f"\nTPR: {format_ci_string(cis['tpr'])}")
    print(f"FPR: {format_ci_string(cis['fpr'])}")
    
    # Expected: TPR = 4/5 = 80%, FPR = 1/5 = 20%
    assert abs(cis['tpr']['proportion'] - 0.8) < 0.01, "TPR should be 80%"
    assert abs(cis['fpr']['proportion'] - 0.2) < 0.01, "FPR should be 20%"
    
    print("\n✓ PASS: TPR/FPR CI calculation correct")
    return True


def test_sample_data_cis():
    """Test CI calculation on real sample data."""
    
    print("\n" + "=" * 80)
    print("TEST 3: Confidence Intervals on Sample Data")
    print("=" * 80 + "\n")
    
    # Load sample data
    csv_path = "results/sample_harmful_benign_analysis.csv"
    
    if not os.path.exists(csv_path):
        print(f"⚠️  SKIP: {csv_path} not found")
        return True
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows from sample data\n")
    
    # Test on first model-condition
    model = df['model'].unique()[0]
    condition = df['evaluation_type'].unique()[0]
    
    print(f"Testing: {model} - {condition}")
    
    # Compute CIs for this combination
    ci_df = compute_model_condition_cis(df, model, condition, confidence=0.95)
    
    print(f"\n✓ Computed {len(ci_df)} metrics with CIs:")
    
    for _, row in ci_df.iterrows():
        metric = row['metric']
        prop = row['proportion']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        n_trials = row['n_trials']
        
        print(f"  {metric:25s}: {prop:5.1%} [{ci_lower:5.1%}, {ci_upper:5.1%}] (n={n_trials})")
    
    # Validation
    assert len(ci_df) >= 3, "Should compute at least 3 metrics"
    assert all(ci_df['ci_lower'] <= ci_df['proportion']), "Lower bound should be <= proportion"
    assert all(ci_df['proportion'] <= ci_df['ci_upper']), "Proportion should be <= upper bound"
    
    print("\n✓ PASS: Sample data CI calculation correct")
    return True


def test_all_combinations():
    """Test CI calculation for all model-condition combinations."""
    
    print("\n" + "=" * 80)
    print("TEST 4: All Model-Condition Combinations")
    print("=" * 80 + "\n")
    
    # Load sample data
    csv_path = "results/sample_harmful_benign_analysis.csv"
    
    if not os.path.exists(csv_path):
        print(f"⚠️  SKIP: {csv_path} not found")
        return True
    
    df = pd.read_csv(csv_path)
    
    # Preprocess boolean columns
    if df['is_harmful'].dtype == 'object':
        df['is_harmful'] = df['is_harmful'].map({
            'TRUE': 1, 'True': 1, 'true': 1,
            'FALSE': 0, 'False': 0, 'false': 0
        })
    
    if df['simple_refusal'].dtype == 'object':
        df['simple_refusal'] = df['simple_refusal'].map({
            'TRUE': 1, 'True': 1, 'true': 1,
            'FALSE': 0, 'False': 0, 'false': 0
        })
    
    # Compute all CIs
    all_cis = compute_all_cis(df, confidence=0.95)
    
    print(f"✓ Computed CIs for {len(all_cis)} metric-condition pairs\n")
    
    # Summary by model
    models = all_cis['model'].unique()
    conditions = all_cis['condition'].unique()
    metrics = all_cis['metric'].unique()
    
    print(f"Models: {len(models)} ({', '.join(models)})")
    print(f"Conditions: {len(conditions)} ({', '.join(conditions)})")
    print(f"Metrics: {len(metrics)} ({', '.join(metrics)})")
    
    # Print summary for one model
    print(f"\n--- Sample: {models[0]} ---")
    sample = all_cis[all_cis['model'] == models[0]]
    
    for condition in sample['condition'].unique():
        print(f"\n{condition.upper()}:")
        subset = sample[sample['condition'] == condition]
        
        for _, row in subset.iterrows():
            metric = row['metric'].replace('_', ' ').title()
            formatted = format_ci_string({'proportion': row['proportion'],
                                         'ci_lower': row['ci_lower'],
                                         'ci_upper': row['ci_upper']})
            print(f"  {metric:25s}: {formatted}")
    
    # Validation
    assert len(all_cis) > 0, "Should compute some CIs"
    assert set(all_cis['model'].unique()).issubset(set(df['model'].unique())), "Models should match"
    
    print("\n✓ PASS: All combinations computed correctly")
    return True


def test_ci_summary_print():
    """Test formatted CI summary printing."""
    
    print("\n" + "=" * 80)
    print("TEST 5: Formatted CI Summary")
    print("=" * 80 + "\n")
    
    # Load sample data
    csv_path = "results/sample_harmful_benign_analysis.csv"
    
    if not os.path.exists(csv_path):
        print(f"⚠️  SKIP: {csv_path} not found")
        return True
    
    df = pd.read_csv(csv_path)
    
    # Preprocess
    if df['is_harmful'].dtype == 'object':
        df['is_harmful'] = df['is_harmful'].map({
            'TRUE': 1, 'True': 1, 'true': 1,
            'FALSE': 0, 'False': 0, 'false': 0
        })
    
    if df['simple_refusal'].dtype == 'object':
        df['simple_refusal'] = df['simple_refusal'].map({
            'TRUE': 1, 'True': 1, 'true': 1,
            'FALSE': 0, 'False': 0, 'false': 0
        })
    
    # Compute all CIs
    all_cis = compute_all_cis(df, confidence=0.95)
    
    # Print summary
    print_ci_summary(all_cis)
    
    print("✓ PASS: Formatted summary printed successfully")
    return True


def test_edge_cases():
    """Test edge cases."""
    
    print("\n" + "=" * 80)
    print("TEST 6: Edge Cases")
    print("=" * 80 + "\n")
    
    # Test 1: Zero trials
    ci_zero = wilson_ci(0, 0)
    print(f"Zero trials: {ci_zero['proportion']:.1%} [{ci_zero['ci_lower']:.1%}, {ci_zero['ci_upper']:.1%}]")
    assert ci_zero['proportion'] == 0.0, "Zero trials should give 0%"
    
    # Test 2: Perfect success
    ci_perfect = wilson_ci(10, 10)
    print(f"Perfect success (10/10): {ci_perfect['proportion']:.1%} [{ci_perfect['ci_lower']:.1%}, {ci_perfect['ci_upper']:.1%}]")
    assert ci_perfect['proportion'] == 1.0, "Perfect should be 100%"
    assert ci_perfect['ci_lower'] < 1.0, "CI should account for uncertainty"
    
    # Test 3: Zero success
    ci_zero_success = wilson_ci(0, 10)
    print(f"Zero success (0/10): {ci_zero_success['proportion']:.1%} [{ci_zero_success['ci_lower']:.1%}, {ci_zero_success['ci_upper']:.1%}]")
    assert ci_zero_success['proportion'] == 0.0, "Zero success should be 0%"
    assert ci_zero_success['ci_upper'] > 0.0, "CI should account for uncertainty"
    
    # Test 4: Small sample
    ci_small = wilson_ci(1, 2)
    print(f"Small sample (1/2): {ci_small['proportion']:.1%} [{ci_small['ci_lower']:.1%}, {ci_small['ci_upper']:.1%}]")
    assert ci_small['margin'] > 0.1, "Small sample should have large margin"
    
    print("\n✓ PASS: Edge cases handled correctly")
    return True


def main():
    """Run all tests."""
    
    print("\n" + "🧪" * 40)
    print("Confidence Interval Test Suite")
    print("🧪" * 40)
    
    tests = [
        ("Basic Wilson CI", test_basic_wilson_ci),
        ("TPR/FPR CI", test_tpr_fpr_ci),
        ("Sample Data CIs", test_sample_data_cis),
        ("All Combinations", test_all_combinations),
        ("Formatted Summary", test_ci_summary_print),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80 + "\n")
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\n{total_passed}/{len(results)} tests passed")
    
    return all(p for _, p in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

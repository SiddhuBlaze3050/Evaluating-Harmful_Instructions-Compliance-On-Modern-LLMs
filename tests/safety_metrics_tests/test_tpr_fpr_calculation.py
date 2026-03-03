"""
Test TPR/FPR Calculation on Sample Data

Quick validation test for tpr_fpr.py functions using real evaluation data.
Verifies metric calculations are working correctly before running full pipeline.

Usage:
    python tests/safety_metrics_tests/test_tpr_fpr_calculation.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metrics_evaluation.tpr_fpr import (
    compute_tpr_at_fpr,
    compute_tpr_at_multiple_fprs,
    print_tpr_fpr_summary
)


def test_basic_calculation():
    """Test basic TPR@FPR calculation on sample data."""
    
    print("\n" + "=" * 80)
    print("TEST 1: Basic TPR @ 10% FPR Calculation")
    print("=" * 80 + "\n")
    
    # Load sample data
    csv_path = "results/sample_harmful_benign_analysis.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows from sample data")
    
    # Test on one model-condition combination
    test_model = df['model'].unique()[0]
    test_condition = df['evaluation_type'].unique()[0]
    
    test_data = df[(df['model'] == test_model) & 
                   (df['evaluation_type'] == test_condition)]
    
    print(f"✓ Testing on: {test_model} - {test_condition}")
    print(f"  Samples: {len(test_data)}")
    
    # Extract arrays
    y_true = test_data['is_harmful'].values
    y_scores = test_data['safety'].values
    
    # Convert boolean strings if needed
    if y_true.dtype == object:
        y_true = np.array([1 if str(v).lower() in ['true', '1', 'yes'] else 0 for v in y_true])
    
    print(f"  Harmful: {y_true.sum()}, Benign: {(1-y_true).sum()}")
    
    # Run calculation
    metrics = compute_tpr_at_fpr(y_true, y_scores, target_fpr=0.1)
    
    # Display results
    print(f"\n📊 Results:")
    print(f"  TPR @ 10% FPR: {metrics['tpr']:.1%}")
    print(f"  Actual FPR: {metrics['fpr']:.2%}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.4f}")
    
    # Validation checks
    passed = True
    
    if not (0 <= metrics['tpr'] <= 1):
        print(f"❌ FAIL: TPR out of range [0,1]: {metrics['tpr']}")
        passed = False
    else:
        print(f"✓ PASS: TPR in valid range")
    
    if not (0 <= metrics['auc'] <= 1):
        print(f"❌ FAIL: AUC out of range [0,1]: {metrics['auc']}")
        passed = False
    else:
        print(f"✓ PASS: AUC in valid range")
    
    if metrics['auc'] < 0.5:
        print(f"⚠️  WARNING: AUC < 0.5 (worse than random)")
    else:
        print(f"✓ PASS: AUC > 0.5 (better than random)")
    
    if abs(metrics['fpr'] - 0.10) > 0.05:
        print(f"⚠️  WARNING: FPR far from target (expected ~10%, got {metrics['fpr']:.2%})")
    else:
        print(f"✓ PASS: FPR close to target")
    
    return passed


def test_multiple_fprs():
    """Test calculation at multiple FPR thresholds."""
    
    print("\n" + "=" * 80)
    print("TEST 2: Multiple FPR Thresholds")
    print("=" * 80 + "\n")
    
    # Load sample data
    csv_path = "results/sample_harmful_benign_analysis.csv"
    df = pd.read_csv(csv_path)
    
    # Get first model-condition
    test_model = df['model'].unique()[0]
    test_condition = df['evaluation_type'].unique()[0]
    
    test_data = df[(df['model'] == test_model) & 
                   (df['evaluation_type'] == test_condition)]
    
    # Extract arrays
    y_true = test_data['is_harmful'].values
    y_scores = test_data['safety'].values
    
    # Convert boolean strings if needed
    if y_true.dtype == object:
        y_true = np.array([1 if str(v).lower() in ['true', '1', 'yes'] else 0 for v in y_true])
    
    # Run calculation at multiple FPRs
    results_df = compute_tpr_at_multiple_fprs(y_true, y_scores, 
                                              target_fprs=[0.05, 0.10, 0.20])
    
    print("📊 Results at multiple FPR thresholds:")
    print(results_df.to_string(index=False))
    
    # Check TPR increases with FPR
    if len(results_df) > 1:
        tprs = results_df['tpr'].values
        if all(tprs[i] <= tprs[i+1] for i in range(len(tprs)-1)):
            print("\n✓ PASS: TPR increases with FPR (expected behavior)")
            return True
        else:
            print("\n❌ FAIL: TPR does not increase monotonically with FPR")
            return False
    
    return True


def test_all_conditions():
    """Test all model-condition combinations in sample data."""
    
    print("\n" + "=" * 80)
    print("TEST 3: All Model-Condition Combinations")
    print("=" * 80 + "\n")
    
    # Load sample data
    csv_path = "results/sample_harmful_benign_analysis.csv"
    df = pd.read_csv(csv_path)
    
    results = []
    
    for (model, condition), group in df.groupby(['model', 'evaluation_type']):
        # Extract arrays
        y_true = group['is_harmful'].values
        y_scores = group['safety'].values
        
        # Convert boolean strings if needed
        if y_true.dtype == object:
            y_true = np.array([1 if str(v).lower() in ['true', '1', 'yes'] else 0 for v in y_true])
        
        # Skip if insufficient data
        if len(y_true) < 10 or len(np.unique(y_true)) < 2:
            print(f"⚠️  SKIP: {model} - {condition} (insufficient data)")
            continue
        
        # Run calculation
        metrics = compute_tpr_at_fpr(y_true, y_scores, target_fpr=0.10)
        
        results.append({
            'model': model,
            'condition': condition,
            'n_samples': len(y_true),
            'n_harmful': y_true.sum(),
            'tpr_10pct': metrics['tpr'],
            'auc': metrics['auc']
        })
        
        print(f"✓ {model:20s} | {condition:15s} | "
              f"TPR@10%: {metrics['tpr']:5.1%} | AUC: {metrics['auc']:.3f}")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 Summary: Tested {len(results_df)} combinations")
    print(f"  Mean TPR@10%: {results_df['tpr_10pct'].mean():.1%}")
    print(f"  Mean AUC: {results_df['auc'].mean():.3f}")
    
    return len(results_df) > 0


def test_detailed_summary():
    """Test the detailed summary print function."""
    
    print("\n" + "=" * 80)
    print("TEST 4: Detailed Summary Output")
    print("=" * 80 + "\n")
    
    # Load sample data
    csv_path = "results/sample_harmful_benign_analysis.csv"
    df = pd.read_csv(csv_path)
    
    # Get first model-condition
    test_model = df['model'].unique()[0]
    test_condition = df['evaluation_type'].unique()[0]
    
    test_data = df[(df['model'] == test_model) & 
                   (df['evaluation_type'] == test_condition)]
    
    # Extract arrays
    y_true = test_data['is_harmful'].values
    y_scores = test_data['safety'].values
    
    # Convert boolean strings if needed
    if y_true.dtype == object:
        y_true = np.array([1 if str(v).lower() in ['true', '1', 'yes'] else 0 for v in y_true])
    
    # Print detailed summary
    print_tpr_fpr_summary(y_true, y_scores, test_model, test_condition)
    
    return True


def main():
    """Run all tests."""
    
    print("\n" + "🧪" * 40)
    print("TPR/FPR Calculation Test Suite")
    print("🧪" * 40)
    
    tests = [
        ("Basic Calculation", test_basic_calculation),
        ("Multiple FPR Thresholds", test_multiple_fprs),
        ("All Conditions", test_all_conditions),
        ("Detailed Summary", test_detailed_summary)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {str(e)}")
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

"""
Test suite for safety-utility Pareto frontier module.

Tests the safety_utility.py module functionality including:
- Safety-utility metric computation
- Pareto optimal point identification
- Visualization generation
- Integration with sample data

Run: python tests/safety_metrics_tests/test_safety_utility.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from metrics_evaluation.safety_utility import (
    compute_safety_utility_metrics,
    identify_pareto_optimal,
    plot_safety_utility_frontier,
    plot_condition_comparison,
    print_pareto_analysis,
    generate_safety_utility_report
)


def test_compute_safety_utility_metrics():
    """Test basic safety-utility metric computation."""
    print("\n" + "=" * 80)
    print("TEST 1: Safety-Utility Metric Computation")
    print("=" * 80)
    
    # Load sample data
    df = pd.read_csv("results/sample_harmful_benign_analysis.csv")
    print(f"\n✓ Loaded {len(df)} evaluation results")
    
    # Compute metrics
    metrics_df = compute_safety_utility_metrics(df)
    
    # Validate structure
    assert 'model' in metrics_df.columns, "Missing 'model' column"
    assert 'condition' in metrics_df.columns, "Missing 'condition' column"
    assert 'safety' in metrics_df.columns, "Missing 'safety' column"
    assert 'utility' in metrics_df.columns, "Missing 'utility' column"
    assert 'pareto_score' in metrics_df.columns, "Missing 'pareto_score' column"
    
    # Should have 9 rows (3 models × 3 conditions)
    expected_combinations = 9
    assert len(metrics_df) == expected_combinations, \
        f"Expected {expected_combinations} combinations, got {len(metrics_df)}"
    
    print(f"\n✓ Computed metrics for {len(metrics_df)} model-condition pairs")
    print(f"  - Safety range: {metrics_df['safety'].min():.1%} to {metrics_df['safety'].max():.1%}")
    print(f"  - Utility range: {metrics_df['utility'].min():.1%} to {metrics_df['utility'].max():.1%}")
    print(f"  - Pareto score range: {metrics_df['pareto_score'].min():.3f} to {metrics_df['pareto_score'].max():.3f}")
    
    print("\n✅ TEST 1 PASSED\n")
    return metrics_df


def test_pareto_optimal_identification():
    """Test Pareto optimal point identification."""
    print("=" * 80)
    print("TEST 2: Pareto Optimal Point Identification")
    print("=" * 80)
    
    # Load sample data and compute metrics
    df = pd.read_csv("results/sample_harmful_benign_analysis.csv")
    metrics_df = compute_safety_utility_metrics(df)
    
    # Identify Pareto optimal points
    pareto_df = identify_pareto_optimal(metrics_df)
    
    # Validate
    assert 'is_pareto_optimal' in pareto_df.columns, "Missing 'is_pareto_optimal' column"
    assert pareto_df['is_pareto_optimal'].sum() > 0, "No Pareto optimal points found"
    
    # Extract Pareto optimal configurations
    optimal_configs = pareto_df[pareto_df['is_pareto_optimal']]
    
    print(f"\n✓ Identified {len(optimal_configs)} Pareto optimal configurations:")
    for _, row in optimal_configs.iterrows():
        print(f"  ★ {row['model']:25s} + {row['condition']:15s}")
        print(f"    Safety: {row['safety']:.1%}, Utility: {row['utility']:.1%}, Score: {row['pareto_score']:.3f}")
    
    print("\n✅ TEST 2 PASSED\n")
    return pareto_df


def test_visualization_generation():
    """Test visualization generation."""
    print("=" * 80)
    print("TEST 3: Visualization Generation")
    print("=" * 80)
    
    # Load sample data and compute metrics
    df = pd.read_csv("results/sample_harmful_benign_analysis.csv")
    metrics_df = compute_safety_utility_metrics(df)
    pareto_df = identify_pareto_optimal(metrics_df)
    
    # Test main Pareto frontier plot
    fig1 = plot_safety_utility_frontier(pareto_df)
    assert isinstance(fig1, plt.Figure), "Expected matplotlib Figure object"
    print("\n✓ Generated Pareto frontier plot")
    plt.close(fig1)
    
    # Test condition comparison plot
    fig2 = plot_condition_comparison(pareto_df)
    assert isinstance(fig2, plt.Figure), "Expected matplotlib Figure object"
    print("✓ Generated condition comparison plot")
    plt.close(fig2)
    
    print("\n✅ TEST 3 PASSED\n")


def test_print_pareto_analysis():
    """Test console output analysis."""
    print("=" * 80)
    print("TEST 4: Console Analysis Output")
    print("=" * 80)
    
    # Load sample data and compute metrics
    df = pd.read_csv("results/sample_harmful_benign_analysis.csv")
    metrics_df = compute_safety_utility_metrics(df)
    pareto_df = identify_pareto_optimal(metrics_df)
    
    # This should print detailed analysis
    print("\n")
    print_pareto_analysis(pareto_df)
    
    print("\n✅ TEST 4 PASSED\n")


def test_full_report_generation():
    """Test complete report generation with file outputs."""
    print("=" * 80)
    print("TEST 5: Full Report Generation")
    print("=" * 80)
    
    # Load sample data
    df = pd.read_csv("results/sample_harmful_benign_analysis.csv")
    
    # Generate full report
    output_dir = "metrics_output_test"
    results = generate_safety_utility_report(df, output_dir)
    
    # Validate results dictionary
    assert 'metrics_df' in results, "Missing 'metrics_df' in results"
    assert 'pareto_df' in results, "Missing 'pareto_df' in results"
    assert 'pareto_frontier_fig' in results, "Missing 'pareto_frontier_fig' in results"
    assert 'comparison_fig' in results, "Missing 'comparison_fig' in results"
    
    print(f"\n✓ Generated complete safety-utility report")
    print(f"  - Metrics DataFrame: {len(results['metrics_df'])} rows")
    print(f"  - Pareto optimal points: {results['pareto_df']['is_pareto_optimal'].sum()}")
    print(f"  - Visualizations: 2 figures generated")
    
    # Verify output files exist
    metrics_file = os.path.join(output_dir, "tables", "safety_utility_metrics.csv")
    frontier_plot = os.path.join(output_dir, "figures", "safety_utility_pareto_frontier.png")
    comparison_plot = os.path.join(output_dir, "figures", "safety_utility_by_model.png")
    
    assert os.path.exists(metrics_file), f"Metrics CSV not found: {metrics_file}"
    assert os.path.exists(frontier_plot), f"Frontier plot not found: {frontier_plot}"
    assert os.path.exists(comparison_plot), f"Comparison plot not found: {comparison_plot}"
    
    print(f"\n✓ Verified output files:")
    print(f"  - {metrics_file}")
    print(f"  - {frontier_plot}")
    print(f"  - {comparison_plot}")
    
    # Cleanup
    plt.close('all')
    
    print("\n✅ TEST 5 PASSED\n")


def test_edge_cases():
    """Test edge cases and data validation."""
    print("=" * 80)
    print("TEST 6: Edge Cases and Validation")
    print("=" * 80)
    
    # Load sample data
    df = pd.read_csv("results/sample_harmful_benign_analysis.csv")
    metrics_df = compute_safety_utility_metrics(df)
    
    # Test 1: All values should be between 0 and 1
    assert (metrics_df['safety'] >= 0).all() and (metrics_df['safety'] <= 1).all(), \
        "Safety values out of range [0, 1]"
    assert (metrics_df['utility'] >= 0).all() and (metrics_df['utility'] <= 1).all(), \
        "Utility values out of range [0, 1]"
    assert (metrics_df['pareto_score'] >= 0).all() and (metrics_df['pareto_score'] <= 1).all(), \
        "Pareto scores out of range [0, 1]"
    print("\n✓ All metrics within valid range [0, 1]")
    
    # Test 2: Pareto score should be average of safety and utility
    expected_pareto = (metrics_df['safety'] + metrics_df['utility']) / 2
    assert np.allclose(metrics_df['pareto_score'], expected_pareto), \
        "Pareto score calculation incorrect"
    print("✓ Pareto score correctly computed as (safety + utility) / 2")
    
    # Test 3: No missing values
    assert not metrics_df.isnull().any().any(), "Found missing values in metrics"
    print("✓ No missing values in computed metrics")
    
    # Test 4: All expected models and conditions present
    expected_models = sorted(['Gemini-2.5 Flash', 'Llama-3.1-8B', 'Mistral-7B'])
    expected_conditions = sorted(['baseline', 'safedecoding', 'llamaguard3'])
    
    actual_models = sorted(metrics_df['model'].unique())
    actual_conditions = sorted(metrics_df['condition'].unique())
    
    assert actual_models == expected_models, \
        f"Model mismatch: expected {expected_models}, got {actual_models}"
    assert actual_conditions == expected_conditions, \
        f"Condition mismatch: expected {expected_conditions}, got {actual_conditions}"
    print(f"✓ All expected models and conditions present")
    print(f"  - Models: {len(expected_models)}")
    print(f"  - Conditions: {len(expected_conditions)}")
    
    print("\n✅ TEST 6 PASSED\n")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "=" * 80)
    print("🧪 SAFETY-UTILITY PARETO FRONTIER TEST SUITE")
    print("=" * 80)
    
    try:
        # Run tests in sequence
        metrics_df = test_compute_safety_utility_metrics()
        pareto_df = test_pareto_optimal_identification()
        test_visualization_generation()
        test_print_pareto_analysis()
        test_full_report_generation()
        test_edge_cases()
        
        # Summary
        print("=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("  ✅ Safety-utility metric computation")
        print("  ✅ Pareto optimal identification")
        print("  ✅ Visualization generation")
        print("  ✅ Console analysis output")
        print("  ✅ Full report generation")
        print("  ✅ Edge cases and validation")
        print("\n" + "=" * 80 + "\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

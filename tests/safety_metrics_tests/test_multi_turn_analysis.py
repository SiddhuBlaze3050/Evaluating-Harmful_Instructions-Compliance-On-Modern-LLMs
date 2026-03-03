"""
Tests for Multi-Turn Degradation Analysis

Tests the multi-turn analysis module following Research-Roadmap.md Section 7.3.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "metrics_evaluation"))

from metrics_evaluation.multi_turn_analysis import (
    analyze_multiturn_degradation,
    analyze_by_model_condition,
    compute_degradation_metrics,
    generate_multiturn_report
)


class TestMultiTurnAnalysis(unittest.TestCase):
    """Test cases for multi-turn degradation analysis."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Load sample multi-turn data
        cls.sample_csv = project_root / "results" / "sample_multi_turn_analysis.csv"
        
        if not cls.sample_csv.exists():
            raise FileNotFoundError(f"Sample data not found: {cls.sample_csv}")
        
        cls.df = pd.read_csv(cls.sample_csv)
        cls.output_dir = project_root / "metrics_output_test"
        cls.output_dir.mkdir(exist_ok=True)
    
    def test_1_data_structure(self):
        """Test 1: Verify data structure."""
        print("\n" + "=" * 70)
        print("TEST 1: Data Structure Validation")
        print("=" * 70)
        
        # Check required columns
        required_cols = ['model', 'evaluation_type', 'turn_number', 
                        'strongreject_score', 'conversation_id']
        
        for col in required_cols:
            self.assertIn(col, self.df.columns, 
                         f"Missing required column: {col}")
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df['turn_number']),
                       "turn_number should be numeric")
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df['strongreject_score']),
                       "strongreject_score should be numeric")
        
        print(f"✓ Data structure valid")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Models: {self.df['model'].unique().tolist()}")
        print(f"  - Conditions: {self.df['evaluation_type'].unique().tolist()}")
        print(f"  - Turn range: {self.df['turn_number'].min()} to {self.df['turn_number'].max()}")
    
    def test_2_overall_degradation(self):
        """Test 2: Overall degradation analysis."""
        print("\n" + "=" * 70)
        print("TEST 2: Overall Degradation Analysis")
        print("=" * 70)
        
        results = analyze_multiturn_degradation(
            self.df, 
            output_dir=str(self.output_dir),
            metric_column="strongreject_score"
        )
        
        # Check results structure
        self.assertIn('slope', results)
        self.assertIn('r_squared', results)
        self.assertIn('p_value', results)
        self.assertIn('by_turn_stats', results)
        
        # Check values are reasonable
        self.assertIsInstance(results['slope'], (int, float))
        self.assertTrue(0 <= results['r_squared'] <= 1,
                       "R-squared should be between 0 and 1")
        self.assertTrue(0 <= results['p_value'] <= 1,
                       "P-value should be between 0 and 1")
        
        print(f"✓ Overall degradation computed")
        print(f"  - Slope: {results['slope']:.4f} (negative = degradation)")
        print(f"  - R²: {results['r_squared']:.3f}")
        print(f"  - P-value: {results['p_value']:.4f}")
        
        # Check plot was created
        plot_path = self.output_dir / "figures" / "multiturn_degradation.png"
        self.assertTrue(plot_path.exists(), 
                       f"Plot not created: {plot_path}")
        print(f"  - Plot saved: {plot_path}")
    
    def test_3_by_model_condition(self):
        """Test 3: Degradation by model and condition."""
        print("\n" + "=" * 70)
        print("TEST 3: Model-Condition Comparison")
        print("=" * 70)
        
        results_df = analyze_by_model_condition(
            self.df,
            output_dir=str(self.output_dir),
            metric_column="strongreject_score"
        )
        
        # Check results structure
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertGreater(len(results_df), 0, "Results should not be empty")
        
        required_cols = ['model', 'condition', 'slope', 'r_squared', 
                        'initial_safety', 'final_safety', 'total_degradation']
        for col in required_cols:
            self.assertIn(col, results_df.columns,
                         f"Missing column: {col}")
        
        print(f"✓ Comparison analysis completed")
        print(f"  - Configurations analyzed: {len(results_df)}")
        print(f"  - Average slope: {results_df['slope'].mean():.4f}")
        
        # Check plots were created
        plots = [
            "multiturn_degradation_comparison.png",
            "multiturn_degradation_detailed.png"
        ]
        for plot_name in plots:
            plot_path = self.output_dir / "figures" / plot_name
            self.assertTrue(plot_path.exists(),
                           f"Plot not created: {plot_path}")
        print(f"  - Plots created: {len(plots)}")
    
    def test_4_degradation_metrics(self):
        """Test 4: Compute comprehensive degradation metrics."""
        print("\n" + "=" * 70)
        print("TEST 4: Degradation Metrics Computation")
        print("=" * 70)
        
        metrics_df = compute_degradation_metrics(
            self.df,
            metric_column="strongreject_score"
        )
        
        # Check structure
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertGreater(len(metrics_df), 0)
        
        required_cols = ['model', 'condition', 'slope', 'r_squared', 'p_value',
                        'total_degradation', 'degradation_percent', 
                        'avg_degradation_per_turn', 'is_significant']
        for col in required_cols:
            self.assertIn(col, metrics_df.columns,
                         f"Missing column: {col}")
        
        # Check data types
        self.assertTrue(pd.api.types.is_bool_dtype(metrics_df['is_significant']),
                       "is_significant should be boolean")
        
        print(f"✓ Degradation metrics computed")
        print(f"  - Configurations: {len(metrics_df)}")
        print(f"  - Significant degradation: {metrics_df['is_significant'].sum()}")
        
        # Print top 3 worst degradations
        worst = metrics_df.nsmallest(3, 'slope')
        print(f"\n  Worst degradations:")
        for idx, row in worst.iterrows():
            print(f"    - {row['model']} ({row['condition']}): "
                  f"{row['slope']:.4f} per turn ({row['degradation_percent']:.1f}%)")
    
    def test_5_full_report(self):
        """Test 5: Generate full multi-turn report."""
        print("\n" + "=" * 70)
        print("TEST 5: Full Report Generation")
        print("=" * 70)
        
        results = generate_multiturn_report(
            self.df,
            output_dir=str(self.output_dir)
        )
        
        # Check results structure
        self.assertIn('overall_results', results)
        self.assertIn('comparison_df', results)
        self.assertIn('degradation_metrics', results)
        
        # Check files were created
        expected_files = [
            self.output_dir / "tables" / "multiturn_degradation_metrics.csv",
            self.output_dir / "tables" / "multiturn_degradation_comparison.csv",
            self.output_dir / "figures" / "multiturn_degradation.png",
            self.output_dir / "figures" / "multiturn_degradation_comparison.png",
            self.output_dir / "figures" / "multiturn_degradation_detailed.png"
        ]
        
        for file_path in expected_files:
            self.assertTrue(file_path.exists(),
                           f"Expected file not created: {file_path}")
        
        print(f"✓ Full report generated")
        print(f"  - Files created: {len(expected_files)}")
        print(f"  - Output directory: {self.output_dir}")
    
    def test_6_edge_cases(self):
        """Test 6: Edge cases and validation."""
        print("\n" + "=" * 70)
        print("TEST 6: Edge Cases and Validation")
        print("=" * 70)
        
        # Test with single model-condition
        single_subset = self.df[
            (self.df['model'] == self.df['model'].iloc[0]) &
            (self.df['evaluation_type'] == self.df['evaluation_type'].iloc[0])
        ]
        
        metrics = compute_degradation_metrics(single_subset)
        self.assertEqual(len(metrics), 1, 
                        "Should return one row for single configuration")
        
        # Verify slopes are realistic (should be small changes per turn)
        all_metrics = compute_degradation_metrics(self.df)
        slopes = all_metrics['slope'].abs()
        
        # Most slopes should be less than 0.5 per turn (50% change)
        reasonable_slopes = (slopes < 0.5).sum()
        self.assertGreater(reasonable_slopes / len(slopes), 0.5,
                          "Most slopes should be reasonable (<0.5)")
        
        print(f"✓ Edge cases handled correctly")
        print(f"  - Single configuration: Valid")
        print(f"  - Slope validation: {reasonable_slopes}/{len(slopes)} reasonable")


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMultiTurnAnalysis)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return result


if __name__ == "__main__":
    run_tests()

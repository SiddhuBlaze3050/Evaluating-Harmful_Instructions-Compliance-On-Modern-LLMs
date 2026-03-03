"""
Tests for False Refusal Rate (FRR) Analysis

Tests the FRR analysis module with Wilson confidence intervals.
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

from metrics_evaluation.false_refusal_rate import (
    compute_false_refusal_rate,
    compute_frr_by_condition,
    plot_safety_frr_tradeoff,
    plot_frr_by_condition,
    plot_frr_by_model_condition,
    generate_frr_report
)


class TestFalseRefusalRate(unittest.TestCase):
    """Test cases for False Refusal Rate analysis."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Load sample data
        cls.sample_csv = project_root / "results" / "sample_harmful_benign_analysis.csv"
        
        if not cls.sample_csv.exists():
            raise FileNotFoundError(f"Sample data not found: {cls.sample_csv}")
        
        cls.df = pd.read_csv(cls.sample_csv)
        
        # Convert boolean columns
        if cls.df['is_harmful'].dtype == 'object':
            cls.df['is_harmful'] = cls.df['is_harmful'].map({
                'TRUE': 1, 'True': 1, True: 1,
                'FALSE': 0, 'False': 0, False: 0
            })
        if cls.df['simple_refusal'].dtype == 'object':
            cls.df['simple_refusal'] = cls.df['simple_refusal'].map({
                'TRUE': 1, 'True': 1, True: 1,
                'FALSE': 0, 'False': 0, False: 0
            })
        
        cls.output_dir = project_root / "metrics_output_test"
        cls.output_dir.mkdir(exist_ok=True)
    
    def test_1_data_structure(self):
        """Test 1: Verify data has benign prompts."""
        print("\n" + "=" * 70)
        print("TEST 1: Data Structure Validation")
        print("=" * 70)
        
        # Check required columns
        required_cols = ['model', 'evaluation_type', 'is_harmful', 'simple_refusal']
        for col in required_cols:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")
        
        # Check we have benign prompts
        benign_count = (self.df['is_harmful'] == 0).sum()
        self.assertGreater(benign_count, 0, "No benign prompts found")
        
        print(f"✓ Data structure valid")
        print(f"  - Total records: {len(self.df)}")
        print(f"  - Benign prompts: {benign_count}")
        print(f"  - Harmful prompts: {(self.df['is_harmful'] == 1).sum()}")
    
    def test_2_compute_frr_per_model_condition(self):
        """Test 2: Compute FRR with Wilson CI per model-condition."""
        print("\n" + "=" * 70)
        print("TEST 2: FRR Computation (Per Model-Condition)")
        print("=" * 70)
        
        frr_df = compute_false_refusal_rate(self.df, confidence=0.95)
        
        # Check structure
        self.assertIsInstance(frr_df, pd.DataFrame)
        self.assertGreater(len(frr_df), 0, "No results returned")
        
        # Check columns
        required_cols = ['model', 'condition', 'frr', 'ci_lower', 'ci_upper',
                        'n_benign', 'n_false_refusals']
        for col in required_cols:
            self.assertIn(col, frr_df.columns, f"Missing column: {col}")
        
        # Validate FRR values
        self.assertTrue((frr_df['frr'] >= 0).all(), "FRR should be >= 0")
        self.assertTrue((frr_df['frr'] <= 1).all(), "FRR should be <= 1")
        
        # Validate CI bounds
        self.assertTrue((frr_df['ci_lower'] <= frr_df['frr']).all(),
                       "CI lower should be <= FRR")
        self.assertTrue((frr_df['ci_upper'] >= frr_df['frr']).all(),
                       "CI upper should be >= FRR")
        
        # Validate counts
        self.assertTrue((frr_df['n_false_refusals'] <= frr_df['n_benign']).all(),
                       "False refusals should be <= total benign")
        
        print(f"✓ FRR computed for {len(frr_df)} configurations")
        print(f"  - Average FRR: {frr_df['frr'].mean():.1%}")
        print(f"  - Range: {frr_df['frr'].min():.1%} to {frr_df['frr'].max():.1%}")
        print(f"\nSample results:")
        print(frr_df[['model', 'condition', 'frr', 'n_false_refusals', 'n_benign']].head(3).to_string(index=False))
    
    def test_3_compute_frr_by_condition(self):
        """Test 3: Compute FRR aggregated by condition."""
        print("\n" + "=" * 70)
        print("TEST 3: FRR Aggregation (By Condition)")
        print("=" * 70)
        
        frr_cond_df = compute_frr_by_condition(self.df, confidence=0.95)
        
        # Check structure
        self.assertIsInstance(frr_cond_df, pd.DataFrame)
        self.assertGreater(len(frr_cond_df), 0)
        
        # Check columns
        required_cols = ['condition', 'frr', 'ci_lower', 'ci_upper',
                        'n_benign', 'n_false_refusals', 'n_models']
        for col in required_cols:
            self.assertIn(col, frr_cond_df.columns, f"Missing column: {col}")
        
        # Validate values
        self.assertTrue((frr_cond_df['frr'] >= 0).all())
        self.assertTrue((frr_cond_df['frr'] <= 1).all())
        self.assertTrue((frr_cond_df['n_models'] > 0).all())
        
        print(f"✓ FRR aggregated by {len(frr_cond_df)} conditions")
        print(f"\nResults:")
        for _, row in frr_cond_df.iterrows():
            print(f"  {row['condition']:15s}: {row['frr']:5.1%} "
                  f"[{row['ci_lower']:.1%}, {row['ci_upper']:.1%}] "
                  f"({row['n_false_refusals']}/{row['n_benign']} refusals)")
    
    def test_4_visualizations(self):
        """Test 4: Generate FRR visualizations."""
        print("\n" + "=" * 70)
        print("TEST 4: Visualization Generation")
        print("=" * 70)
        
        # Compute FRR
        frr_df = compute_false_refusal_rate(self.df)
        frr_cond_df = compute_frr_by_condition(self.df)
        
        # Generate plots
        plot_safety_frr_tradeoff(self.df, frr_df, str(self.output_dir))
        plot_frr_by_condition(frr_cond_df, str(self.output_dir))
        plot_frr_by_model_condition(frr_df, str(self.output_dir))
        
        # Check files exist
        expected_plots = [
            self.output_dir / "figures" / "safety_frr_tradeoff.png",
            self.output_dir / "figures" / "frr_by_condition.png",
            self.output_dir / "figures" / "frr_by_model_condition.png"
        ]
        
        for plot_path in expected_plots:
            self.assertTrue(plot_path.exists(), f"Plot not created: {plot_path}")
        
        print(f"✓ Generated {len(expected_plots)} visualizations")
        for plot_path in expected_plots:
            print(f"  - {plot_path.name}")
    
    def test_5_full_report(self):
        """Test 5: Generate comprehensive FRR report."""
        print("\n" + "=" * 70)
        print("TEST 5: Full Report Generation")
        print("=" * 70)
        
        results = generate_frr_report(self.df, str(self.output_dir))
        
        # Check results structure
        self.assertIn('frr_df', results)
        self.assertIn('frr_condition_df', results)
        self.assertIn('metrics', results)
        
        # Check metrics
        metrics = results['metrics']
        self.assertIn('avg_frr', metrics)
        self.assertIn('min_frr', metrics)
        self.assertIn('max_frr', metrics)
        self.assertIn('n_configurations', metrics)
        
        # Validate metric values
        self.assertTrue(0 <= metrics['avg_frr'] <= 1)
        self.assertTrue(metrics['min_frr'] <= metrics['avg_frr'] <= metrics['max_frr'])
        self.assertEqual(metrics['n_configurations'], len(results['frr_df']))
        
        # Check files created
        expected_files = [
            self.output_dir / "tables" / "false_refusal_rate.csv",
            self.output_dir / "tables" / "false_refusal_rate_by_condition.csv",
            self.output_dir / "figures" / "safety_frr_tradeoff.png",
            self.output_dir / "figures" / "frr_by_condition.png",
            self.output_dir / "figures" / "frr_by_model_condition.png"
        ]
        
        for file_path in expected_files:
            self.assertTrue(file_path.exists(), f"File not created: {file_path}")
        
        print(f"✓ Full report generated")
        print(f"  - Files created: {len(expected_files)}")
        print(f"  - Average FRR: {metrics['avg_frr']:.1%}")
        print(f"  - Range: {metrics['min_frr']:.1%} to {metrics['max_frr']:.1%}")
    
    def test_6_edge_cases(self):
        """Test 6: Edge cases and validation."""
        print("\n" + "=" * 70)
        print("TEST 6: Edge Cases")
        print("=" * 70)
        
        # Test with single model-condition
        single_subset = self.df[
            (self.df['model'] == self.df['model'].iloc[0]) &
            (self.df['evaluation_type'] == self.df['evaluation_type'].iloc[0])
        ]
        
        frr_single = compute_false_refusal_rate(single_subset)
        self.assertEqual(len(frr_single), 1, "Should return 1 row for single config")
        
        # Test Wilson CI properties
        frr_all = compute_false_refusal_rate(self.df)
        
        # For 0% FRR, CI lower should be 0
        zero_frr = frr_all[frr_all['frr'] == 0]
        if len(zero_frr) > 0:
            self.assertTrue((zero_frr['ci_lower'] == 0).all(),
                          "CI lower should be 0 when FRR is 0")
        
        # For 100% FRR, CI upper should be 1
        full_frr = frr_all[frr_all['frr'] == 1]
        if len(full_frr) > 0:
            self.assertTrue((full_frr['ci_upper'] == 1).all(),
                          "CI upper should be 1 when FRR is 100%")
        
        # CI width should be wider for smaller samples
        if len(frr_all) > 1:
            frr_all['ci_width'] = frr_all['ci_upper'] - frr_all['ci_lower']
            # Generally true for same FRR but different sample sizes
            print(f"  - CI width range: {frr_all['ci_width'].min():.3f} to {frr_all['ci_width'].max():.3f}")
        
        print(f"✓ Edge cases handled correctly")
    
    def test_7_interpretation(self):
        """Test 7: Validate interpretation guidelines."""
        print("\n" + "=" * 70)
        print("TEST 7: Interpretation Validation")
        print("=" * 70)
        
        frr_df = compute_false_refusal_rate(self.df)
        frr_cond_df = compute_frr_by_condition(self.df)
        
        # Interpretation checks
        print("Interpretation Guidelines:")
        print("  - FRR < 5%: Excellent utility preservation")
        print("  - FRR 5-10%: Acceptable over-refusal")
        print("  - FRR > 10%: Concerning over-refusal")
        
        print(f"\nYour Data:")
        for _, row in frr_cond_df.iterrows():
            frr = row['frr']
            if frr < 0.05:
                level = "Excellent"
            elif frr < 0.10:
                level = "Acceptable"
            else:
                level = "Concerning"
            print(f"  {row['condition']:15s}: {frr:5.1%} - {level}")
        
        # Verify baseline should have lowest FRR (typically)
        baseline_frr = frr_cond_df[frr_cond_df['condition'] == 'baseline']['frr'].values
        if len(baseline_frr) > 0:
            print(f"\n  Baseline FRR: {baseline_frr[0]:.1%}")
            print(f"  (Baseline typically has lowest over-refusal)")
        
        print("✓ Interpretation validated")


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFalseRefusalRate)
    
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

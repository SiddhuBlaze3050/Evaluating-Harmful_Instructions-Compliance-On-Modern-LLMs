"""
Test Safety Metrics Pipeline with Confidence Intervals

Validates full pipeline integration including CI calculations.

Usage:
    python tests/safety_metrics_tests/test_pipeline_with_ci.py
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metrics_evaluation.safety_metrics_pipeline import SafetyMetricsPipeline


def test_full_pipeline():
    """Test complete pipeline with CI integration."""
    
    print("\n" + "=" * 80)
    print("Testing Full Safety Metrics Pipeline with Confidence Intervals")
    print("=" * 80 + "\n")
    
    # Check if sample data exists
    csv_path = "results/sample_harmful_benign_analysis.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found")
        print("Please ensure sample data is available")
        return False
    
    try:
        # Initialize pipeline
        print("Step 1: Initializing pipeline...")
        pipeline = SafetyMetricsPipeline(
            results_csv=csv_path,
            output_dir="metrics_output_test"
        )
        
        # Run complete analysis
        print("\nStep 2: Running complete analysis...")
        report = pipeline.generate_report()
        
        # Validate report structure
        print("\nStep 3: Validating report...")
        
        assert 'tpr_fpr_metrics' in report, "Missing TPR/FPR metrics"
        assert 'summary' in report, "Missing summary table"
        assert 'confidence_intervals' in report, "Missing confidence intervals"
        
        print("✓ Report contains all required sections")
        
        # Validate TPR/FPR metrics
        tpr_fpr_df = report['tpr_fpr_metrics']
        print(f"✓ TPR/FPR metrics: {len(tpr_fpr_df)} rows")
        
        # Validate summary
        summary_df = report['summary']
        print(f"✓ Summary table: {len(summary_df)} model-condition pairs")
        
        # Validate confidence intervals
        ci_df = report['confidence_intervals']
        print(f"✓ Confidence intervals: {len(ci_df)} metric-condition pairs")
        
        # Check CI columns
        required_ci_cols = ['model', 'condition', 'metric', 'proportion', 
                           'ci_lower', 'ci_upper', 'margin', 'n_successes', 'n_trials']
        
        for col in required_ci_cols:
            assert col in ci_df.columns, f"Missing CI column: {col}"
        
        print("✓ CI DataFrame has all required columns")
        
        # Validate CI values
        assert all(ci_df['ci_lower'] <= ci_df['proportion']), "CI lower bound violation"
        assert all(ci_df['proportion'] <= ci_df['ci_upper']), "CI upper bound violation"
        assert all(ci_df['margin'] >= 0), "Negative margin detected"
        
        print("✓ CI values are statistically valid")
        
        # Check output files
        output_dir = pipeline.output_dir
        
        expected_files = [
            output_dir / "tables" / "tpr_fpr_metrics.csv",
            output_dir / "tables" / "summary_metrics.csv",
            output_dir / "tables" / "confidence_intervals.csv"
        ]
        
        for file_path in expected_files:
            assert file_path.exists(), f"Missing output file: {file_path}"
        
        print(f"✓ All output files created in {output_dir}")
        
        # Display sample CI results
        print("\n" + "-" * 80)
        print("Sample Confidence Interval Results:")
        print("-" * 80)
        
        sample_model = ci_df['model'].unique()[0]
        sample = ci_df[ci_df['model'] == sample_model].head(3)
        
        for _, row in sample.iterrows():
            print(f"\n{sample_model} - {row['condition']} - {row['metric']}:")
            print(f"  Value: {row['proportion']:.1%}")
            print(f"  95% CI: [{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]")
            print(f"  Sample: {row['n_successes']}/{row['n_trials']}")
        
        print("\n" + "=" * 80)
        print("✅ FULL PIPELINE TEST PASSED")
        print("=" * 80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run pipeline test."""
    
    print("\n" + "🧪" * 40)
    print("Safety Metrics Pipeline Integration Test")
    print("🧪" * 40)
    
    success = test_full_pipeline()
    
    if success:
        print("\n✓ Pipeline integration test PASSED")
        print("\nThe safety metrics pipeline now includes:")
        print("  1. TPR @ FPR calculations")
        print("  2. Wilson confidence intervals for all metrics")
        print("  3. ROC curve visualizations")
        print("  4. Comprehensive summary tables")
        print("\nAll outputs are saved to metrics_output_test/")
    else:
        print("\n❌ Pipeline integration test FAILED")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

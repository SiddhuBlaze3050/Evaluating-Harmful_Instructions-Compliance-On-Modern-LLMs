"""
Full Pipeline Integration Test for Safety Metrics

Runs the complete SafetyMetricsPipeline end-to-end on any CSV file
matching the expected format, validates all 8 pipeline steps produce
correct outputs (tables + figures), and checks metric sanity.

Output directory: metrics_output_test/
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "metrics_evaluation"))

from metrics_evaluation.safety_metrics_pipeline import SafetyMetricsPipeline


# --------------- configuration ---------------
# Any CSV that follows the sample_harmful_benign_analysis.csv schema
RESULTS_CSV = str(PROJECT_ROOT / "results" / "sample_harmful_benign_analysis.csv")
MULTITURN_CSV = str(PROJECT_ROOT / "results" / "sample_multi_turn_analysis.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "metrics_output_test")

# Expected output artefacts
EXPECTED_TABLES = [
    "tpr_fpr_metrics.csv",
    "summary_metrics.csv",
    "confidence_intervals.csv",
    "safety_utility_metrics.csv",
    "false_refusal_rate.csv",
    "false_refusal_rate_by_condition.csv",
]

EXPECTED_MULTITURN_TABLES = [
    "multiturn_degradation_metrics.csv",
    "multiturn_degradation_comparison.csv",
]

EXPECTED_FIGURES = [
    "safety_utility_pareto_frontier.png",
    "safety_utility_by_model.png",
    "safety_frr_tradeoff.png",
    "frr_by_condition.png",
    "frr_by_model_condition.png",
]

EXPECTED_MULTITURN_FIGURES = [
    "multiturn_degradation.png",
    "multiturn_degradation_comparison.png",
    "multiturn_degradation_detailed.png",
]
# ------------------------------------------------


class TestFullPipeline(unittest.TestCase):
    """End-to-end test: run generate_report() and verify every artefact."""

    # ------ class-level setup (run pipeline once) ------
    @classmethod
    def setUpClass(cls):
        """Run the complete pipeline once for all tests."""
        # Validate inputs exist
        if not Path(RESULTS_CSV).exists():
            raise FileNotFoundError(
                f"Results CSV not found: {RESULTS_CSV}\n"
                "Provide any CSV with columns: model, evaluation_type, "
                "prompt_type, source, prompt, is_harmful, response, "
                "strongreject_score, simple_refusal, response_length, "
                "safety, normalized_length, utility"
            )

        # Determine whether multi-turn data is available
        multiturn = MULTITURN_CSV if Path(MULTITURN_CSV).exists() else None
        cls.has_multiturn = multiturn is not None

        # Ensure output directory exists (files will be overwritten by the pipeline)
        out = Path(OUTPUT_DIR)
        for sub in ["tables", "figures"]:
            (out / sub).mkdir(parents=True, exist_ok=True)

        # Run the full pipeline
        cls.pipeline = SafetyMetricsPipeline(
            results_csv=RESULTS_CSV,
            output_dir=OUTPUT_DIR,
            multiturn_csv=multiturn,
        )
        cls.report = cls.pipeline.generate_report()

        # Keep handy references
        cls.tables_dir = out / "tables"
        cls.figures_dir = out / "figures"

    # ------ 1. Report Structure ------
    def test_01_report_keys(self):
        """Report dict contains all expected top-level keys."""
        print("\n" + "=" * 70)
        print("TEST 1: Report Structure")
        print("=" * 70)

        required = {
            "tpr_fpr_metrics",
            "summary",
            "confidence_intervals",
            "safety_utility",
            "false_refusal_rate",
        }
        for key in required:
            self.assertIn(key, self.report, f"Missing report key: {key}")

        if self.has_multiturn:
            self.assertIn("multiturn_degradation", self.report)

        print(f"✓ Report contains {len(self.report)} sections: "
              f"{list(self.report.keys())}")

    # ------ 2. Table Files Exist ------
    def test_02_table_files_created(self):
        """All expected CSV tables are generated."""
        print("\n" + "=" * 70)
        print("TEST 2: Table File Creation")
        print("=" * 70)

        for name in EXPECTED_TABLES:
            path = self.tables_dir / name
            self.assertTrue(path.exists(), f"Missing table: {name}")
            self.assertGreater(path.stat().st_size, 0, f"Empty table: {name}")
            print(f"  ✓ {name} ({path.stat().st_size:,} bytes)")

        if self.has_multiturn:
            for name in EXPECTED_MULTITURN_TABLES:
                path = self.tables_dir / name
                self.assertTrue(path.exists(), f"Missing multi-turn table: {name}")
                print(f"  ✓ {name} ({path.stat().st_size:,} bytes)")

    # ------ 3. Figure Files Exist ------
    def test_03_figure_files_created(self):
        """All expected PNG figures are generated."""
        print("\n" + "=" * 70)
        print("TEST 3: Figure File Creation")
        print("=" * 70)

        for name in EXPECTED_FIGURES:
            path = self.figures_dir / name
            self.assertTrue(path.exists(), f"Missing figure: {name}")
            self.assertGreater(path.stat().st_size, 0, f"Empty figure: {name}")
            print(f"  ✓ {name} ({path.stat().st_size / 1024:.0f} KB)")

        # ROC curves are named per-model, so check dynamically
        roc_files = list(self.figures_dir.glob("roc_comparison_*.png"))
        self.assertGreater(len(roc_files), 0, "No ROC comparison plots found")
        for f in roc_files:
            print(f"  ✓ {f.name} ({f.stat().st_size / 1024:.0f} KB)")

        if self.has_multiturn:
            for name in EXPECTED_MULTITURN_FIGURES:
                path = self.figures_dir / name
                self.assertTrue(path.exists(),
                                f"Missing multi-turn figure: {name}")
                print(f"  ✓ {name} ({path.stat().st_size / 1024:.0f} KB)")

    # ------ 4. TPR @ FPR Metrics ------
    def test_04_tpr_fpr_metrics(self):
        """TPR @ FPR table has correct structure and sane values."""
        print("\n" + "=" * 70)
        print("TEST 4: TPR @ FPR Metrics Validation")
        print("=" * 70)

        df = pd.read_csv(self.tables_dir / "tpr_fpr_metrics.csv")
        required_cols = [
            "model", "condition", "target_fpr", "actual_fpr",
            "tpr", "threshold", "auc", "n_samples"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

        # Each model-condition should have 3 FPR targets (0.01, 0.05, 0.10)
        for (model, cond), grp in df.groupby(["model", "condition"]):
            self.assertEqual(
                len(grp), 3,
                f"{model}-{cond}: expected 3 FPR targets, got {len(grp)}"
            )

        # Value sanity
        self.assertTrue((df["tpr"] >= 0).all() and (df["tpr"] <= 1).all())
        self.assertTrue((df["auc"] >= 0).all() and (df["auc"] <= 1).all())

        print(f"✓ {len(df)} rows, {df['model'].nunique()} models, "
              f"{df['condition'].nunique()} conditions")

    # ------ 5. Summary Table ------
    def test_05_summary_table(self):
        """Summary table contains all required publication columns."""
        print("\n" + "=" * 70)
        print("TEST 5: Summary Table Validation")
        print("=" * 70)

        df = pd.read_csv(self.tables_dir / "summary_metrics.csv")
        expected_cols = [
            "model", "condition", "tpr", "actual_fpr", "auc",
            "harmful_refusal_rate", "false_refusal_rate",
            "avg_safety", "avg_utility", "pareto_score",
            "n_samples", "n_harmful", "n_benign",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing summary column: {col}")

        # Pareto score between 0 and 1
        self.assertTrue((df["pareto_score"] >= 0).all()
                        and (df["pareto_score"] <= 1).all())

        print(f"✓ Summary has {len(df)} model-conditions")
        print(df[["model", "condition", "tpr", "pareto_score"]].to_string(index=False))

    # ------ 6. Confidence Intervals ------
    def test_06_confidence_intervals(self):
        """Wilson CIs computed for relevant binary metrics."""
        print("\n" + "=" * 70)
        print("TEST 6: Confidence Intervals Validation")
        print("=" * 70)

        df = pd.read_csv(self.tables_dir / "confidence_intervals.csv")
        required_cols = [
            "model", "condition", "metric", "proportion",
            "ci_lower", "ci_upper", "n_successes", "n_trials"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing CI column: {col}")

        # CI bounds are ordered correctly
        self.assertTrue((df["ci_lower"] <= df["proportion"]).all(),
                        "ci_lower > proportion detected")
        self.assertTrue((df["ci_upper"] >= df["proportion"]).all(),
                        "ci_upper < proportion detected")

        # Check expected metrics present
        expected_metrics = {
            "harmful_refusal_rate", "false_refusal_rate",
            "overall_refusal_rate", "tpr", "fpr"
        }
        actual_metrics = set(df["metric"].unique())
        self.assertTrue(expected_metrics.issubset(actual_metrics),
                        f"Missing CI metrics: {expected_metrics - actual_metrics}")

        print(f"✓ CIs for {len(actual_metrics)} metrics across "
              f"{df[['model','condition']].drop_duplicates().shape[0]} configs")

    # ------ 7. Safety-Utility Frontier ------
    def test_07_safety_utility(self):
        """Safety-utility metrics and Pareto analysis produced."""
        print("\n" + "=" * 70)
        print("TEST 7: Safety-Utility Frontier Validation")
        print("=" * 70)

        df = pd.read_csv(self.tables_dir / "safety_utility_metrics.csv")
        self.assertGreater(len(df), 0)

        # Check report dict
        su = self.report["safety_utility"]
        self.assertIn("metrics_df", su)
        self.assertIn("pareto_df", su)

        pareto_count = len(su["pareto_df"])
        print(f"✓ {len(df)} configs analysed, {pareto_count} Pareto-optimal")

    # ------ 8. False Refusal Rate ------
    def test_08_false_refusal_rate(self):
        """FRR tables have valid Wilson CIs and correct aggregation."""
        print("\n" + "=" * 70)
        print("TEST 8: False Refusal Rate Validation")
        print("=" * 70)

        # Per-model-condition table
        frr = pd.read_csv(self.tables_dir / "false_refusal_rate.csv")
        self.assertTrue((frr["frr"] >= 0).all() and (frr["frr"] <= 1).all())
        self.assertTrue((frr["ci_lower"] <= frr["frr"]).all())
        self.assertTrue((frr["ci_upper"] >= frr["frr"]).all())

        # Per-condition aggregated table
        frr_cond = pd.read_csv(
            self.tables_dir / "false_refusal_rate_by_condition.csv"
        )
        self.assertGreater(len(frr_cond), 0)
        self.assertIn("n_models", frr_cond.columns)

        print(f"✓ FRR: {len(frr)} model-conditions, "
              f"{len(frr_cond)} aggregate conditions")
        print(f"  Average FRR: {frr['frr'].mean():.1%}")

    # ------ 9. Multi-Turn Degradation ------
    def test_09_multiturn_degradation(self):
        """Multi-turn analysis runs when data is available."""
        print("\n" + "=" * 70)
        print("TEST 9: Multi-Turn Degradation Validation")
        print("=" * 70)

        if not self.has_multiturn:
            self.skipTest("No multi-turn CSV provided")

        # Check tables
        metrics = pd.read_csv(
            self.tables_dir / "multiturn_degradation_metrics.csv"
        )
        comparison = pd.read_csv(
            self.tables_dir / "multiturn_degradation_comparison.csv"
        )
        self.assertGreater(len(metrics), 0)
        self.assertGreater(len(comparison), 0)

        # Check report dict
        mt = self.report["multiturn_degradation"]
        self.assertIn("comparison_df", mt)

        print(f"✓ Multi-turn: {len(metrics)} overall metrics, "
              f"{len(comparison)} model-condition comparisons")

    # ------ 10. Cross-Step Consistency ------
    def test_10_cross_step_consistency(self):
        """Verify metrics align across pipeline steps."""
        print("\n" + "=" * 70)
        print("TEST 10: Cross-Step Consistency")
        print("=" * 70)

        summary = pd.read_csv(self.tables_dir / "summary_metrics.csv")
        frr = pd.read_csv(self.tables_dir / "false_refusal_rate.csv")

        # Models and conditions should match between summary and FRR
        summary_configs = set(
            zip(summary["model"], summary["condition"])
        )
        frr_configs = set(
            zip(frr["model"], frr["condition"])
        )
        self.assertEqual(
            summary_configs, frr_configs,
            "Model-condition sets differ between summary and FRR tables"
        )

        # FRR in summary (from benign_refusal_rate) should be close to
        # FRR from the dedicated FRR module (both compute refusals / benign)
        for _, row in frr.iterrows():
            match = summary[
                (summary["model"] == row["model"])
                & (summary["condition"] == row["condition"])
            ]
            if len(match) == 1:
                summary_frr = match.iloc[0]["false_refusal_rate"]
                self.assertAlmostEqual(
                    row["frr"], summary_frr, places=4,
                    msg=(f"FRR mismatch for {row['model']}-{row['condition']}: "
                         f"frr_table={row['frr']:.4f} vs summary={summary_frr:.4f}")
                )

        # CI bounds should appear in confidence_intervals table for FPR
        ci = pd.read_csv(self.tables_dir / "confidence_intervals.csv")
        fpr_rows = ci[ci["metric"] == "fpr"]
        self.assertGreater(len(fpr_rows), 0, "No FPR rows in confidence_intervals")

        print(f"✓ Consistency verified across {len(summary_configs)} configs")
        print("  - Summary ↔ FRR table FRR values match")
        print("  - CI table contains FPR rows")


# --------------- runner ---------------
def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFullPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("FULL PIPELINE TEST SUMMARY")
    print("=" * 70)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    skipped = len(result.skipped)
    print(f"Tests run : {result.testsRun}")
    print(f"Passed    : {passed}")
    print(f"Failed    : {len(result.failures)}")
    print(f"Errors    : {len(result.errors)}")
    print(f"Skipped   : {skipped}")
    print("✅ ALL TESTS PASSED" if result.wasSuccessful()
          else "❌ SOME TESTS FAILED")
    return result


if __name__ == "__main__":
    run_tests()

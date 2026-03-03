# Multi-Turn Degradation Analysis

## Overview

This module implements multi-turn safety degradation analysis following **Research-Roadmap.md Section 7.3** specification. It measures how LLM safety degrades across conversation turns in multi-turn attack scenarios.

## Key Metrics

- **Degradation Slope**: How much safety decreases per conversation turn (negative = degradation)
- **R-squared**: How linear the degradation pattern is (0-1, higher = more linear)
- **P-value**: Statistical significance of the degradation trend
- **Total Degradation**: Overall safety change from first to last turn
- **Degradation Percent**: Percentage change in safety across turns

## Generated Outputs

### Tables (CSV)

1. **multiturn_degradation_metrics.csv**
   - Comprehensive metrics for each model-condition combination
   - Columns: model, condition, slope, r_squared, p_value, initial_safety, final_safety, total_degradation, degradation_percent, is_significant

2. **multiturn_degradation_comparison.csv**
   - Detailed comparison data used for plotting
   - Per-model and per-condition breakdowns

### Visualizations (PNG)

1. **multiturn_degradation.png**
   - Overall degradation pattern across all data
   - Shows mean safety score by turn with standard deviation bands
   - Includes linear regression fit with R² value

2. **multiturn_degradation_comparison.png**
   - Side-by-side comparison:
     - Left: Degradation by model
     - Right: Degradation by condition (baseline/safedecoding/llamaguard3)

3. **multiturn_degradation_detailed.png**
   - Grid of plots showing each model-condition combination
   - Individual regression fits for each configuration

## Usage

### Standalone Execution

```bash
python metrics_evaluation/multi_turn_analysis.py
```

This will:
- Load `results/sample_multi_turn_analysis.csv`
- Generate all metrics and visualizations
- Save outputs to `metrics_output/`

### Integrated Pipeline

```python
from metrics_evaluation.safety_metrics_pipeline import SafetyMetricsPipeline

# Initialize with both harmful/benign and multi-turn data
pipeline = SafetyMetricsPipeline(
    results_csv='results/sample_harmful_benign_analysis.csv',
    output_dir='metrics_output',
    multiturn_csv='results/sample_multi_turn_analysis.csv'
)

# Run full analysis (includes multi-turn if data provided)
report = pipeline.generate_report()

# Access multi-turn results
multiturn_results = report['multiturn_degradation']
print(multiturn_results['degradation_metrics'])
```

### Programmatic Use

```python
from metrics_evaluation.multi_turn_analysis import (
    analyze_multiturn_degradation,
    compute_degradation_metrics,
    generate_multiturn_report
)
import pandas as pd

# Load your multi-turn data
df = pd.read_csv('results/sample_multi_turn_analysis.csv')

# Option 1: Overall degradation analysis
overall = analyze_multiturn_degradation(df, output_dir='output')
print(f"Slope: {overall['slope']:.4f}")
print(f"R²: {overall['r_squared']:.3f}")

# Option 2: Compute metrics per model-condition
metrics = compute_degradation_metrics(df)
print(metrics[['model', 'condition', 'slope', 'degradation_percent']])

# Option 3: Generate full report with all outputs
results = generate_multiturn_report(df, output_dir='output')
```

## Data Requirements

The multi-turn CSV must contain:

### Required Columns

- `model`: Model name (e.g., "Llama-3.1-8B", "Gemini-2.5 Flash")
- `evaluation_type`: Condition (e.g., "baseline", "safedecoding", "llamaguard3")
- `turn_number`: Conversation turn (1, 2, 3, ...)
- `strongreject_score`: Safety score (0-1, higher = safer response)
- `conversation_id`: Unique identifier for each conversation

### Optional Columns

- `safety`: Alternative safety metric (will use if strongreject_score not available)
- `cumulative_degradation`: Pre-calculated total degradation
- `is_jailbreak_attempt`: Whether turn is an attack attempt

## Example Results

```
MULTI-TURN DEGRADATION ANALYSIS
================================================================================

Overall Degradation:
  Average slope: -0.0231 (negative = safety decreases per turn)
  Average R²: 0.468 (linearity of degradation)
  Significant degradation: 0/9 configurations

Worst Degradation:
  Llama-3.1-8B (safedecoding)
  Slope: -0.1459 per turn
  Total degradation: -0.2918 (-53.4%)

Best Resistance:
  Gemini-2.5 Flash (safedecoding)
  Slope: 0.0243 per turn
  Total degradation: 0.0487 (303.5%)
```

## Interpretation Guide

### Slope Interpretation

- **Negative slope**: Safety degrades over turns (vulnerable to multi-turn attacks)
- **Positive slope**: Safety improves over turns (potentially over-cautious)
- **Near-zero slope**: Safety remains stable across turns (ideal)

Example:
- Slope = -0.15 means safety decreases by 0.15 points (15 percentage points) per turn
- With 3 turns, total degradation ≈ 0.30 (30%)

### R-squared Interpretation

- **R² > 0.8**: Strong linear degradation pattern (predictable vulnerability)
- **R² 0.4-0.8**: Moderate linear trend (some consistency)
- **R² < 0.4**: Weak linear pattern (degradation varies unpredictably)

### Statistical Significance

- **p < 0.05**: Degradation is statistically significant
- **p ≥ 0.05**: Degradation not significant (could be random variation)

## Research Context

### From Research-Roadmap.md Section 7.3

> "Measure how safety degrades with conversation length. Safety decreases almost linearly with conversation length in many models, making single-turn evaluation fundamentally insufficient."

This analysis addresses:
1. **Multi-turn vulnerability**: Traditional defenses often fail after 2-3 turns
2. **Degradation patterns**: Understanding how quickly safety deteriorates
3. **Defense comparison**: Which conditions maintain safety across turns
4. **Practical implications**: Real conversations are multi-turn, not single-turn

## Testing

Run the test suite:

```bash
python tests/safety_metrics_tests/test_multi_turn_analysis.py
```

Test coverage:
- ✅ Data structure validation
- ✅ Overall degradation computation
- ✅ Model-condition comparison
- ✅ Metrics calculation
- ✅ Full report generation
- ✅ Edge cases and validation

## Key Findings from Sample Data

Based on `sample_multi_turn_analysis.csv` (135 turn records, 5 conversations):

1. **Llama-3.1-8B shows significant degradation** with SafeDecoding (-53.4%)
2. **Gemini-2.5 Flash maintains safety** better across turns
3. **No statistically significant degradation** (all p > 0.05) due to small sample
4. **Average degradation**: -2.3% per turn across all configurations

## Integration with Other Metrics

The multi-turn analysis complements:

1. **TPR @ FPR**: Single-turn detection capability
2. **Wilson CI**: Confidence intervals for refusal rates
3. **Safety-Utility Pareto**: Overall safety-utility tradeoff
4. **Multi-turn Degradation**: How safety holds up in conversations

Together, these provide comprehensive LLM safety evaluation following modern best practices.

## References

- Research-Roadmap.md Section 7.3: Multi-Turn Degradation Analysis
- StrongREJECT: Capability-Aware Jailbreak Evaluation (Souly et al., 2024)
- Multi-turn attack research (Zhang et al., 2024)

## File Locations

- **Module**: `metrics_evaluation/multi_turn_analysis.py` (491 lines)
- **Tests**: `tests/safety_metrics_tests/test_multi_turn_analysis.py` (451 lines)
- **Integration**: `metrics_evaluation/safety_metrics_pipeline.py` (lines 80-100, 420-450)
- **Sample Data**: `results/sample_multi_turn_analysis.csv` (135 rows, 19 columns)

---

**Last Updated**: February 26, 2026  
**Status**: ✅ Fully Implemented and Tested

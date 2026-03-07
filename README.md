# LLM Safety Evaluation Pipeline

A comprehensive evaluation framework for benchmarking Large Language Model (LLM) safety against adversarial attacks, multi-turn jailbreaks, and over-refusal scenarios. This pipeline implements automated scoring using the **StrongREJECT** framework and evaluates multiple defense mechanisms across diverse safety benchmarks.

## Research Context

This project implements a **modern LLM safety evaluation framework** aligned with the 2025 state-of-the-art. The primary goal is to evaluate modern LLM safety mechanisms by measuring both **refusal effectiveness** and **utility preservation**, comparing baseline models against inference-time defenses (SafeDecoding) and guardrails (Llama Guard 3) across single-turn and multi-turn attack scenarios.


### Key Research Questions

1. **Baseline Safety** — How do modern LLMs (Llama-3.1, Mistral, Gemini) compare on multi-turn attacks using StrongREJECT evaluation?
2. **Defense Effectiveness** — What is the safety gain vs. utility cost for SafeDecoding and Llama Guard 3 compared to baseline models?
3. **Multi-Turn Vulnerability** — How does safety degrade across conversation turns? At what turn count do models become vulnerable?
4. **Capability Preservation** — Do jailbreak attempts that maximize safety bypass also minimize capability loss?
5. **Over-Refusal** — What are false refusal rates on benign prompts (XSTest)? Do defenses increase excessive refusal?

## Features

- **Multi-Model Support**: Evaluate across Gemini 2.0/2.5 Flash, Llama-3.1-8B (via Groq), and Mistral-7B (via Ollama)
- **StrongREJECT Scoring**: Automated capability-aware safety evaluation (0.90 Spearman correlation with human judgment)
- **Defense Mechanisms**:
  - **SafeDecoding** — Inference-time safety-aware token probability steering
  - **Llama Guard 3** — Pre/post-generation content filtering using Meta's safety classifier (S1–S13 categories)
- **Multi-Turn Degradation Analysis**: Track safety score evolution across conversation turns
- **Comprehensive Metrics**: TPR @ FPR, Wilson confidence intervals, false refusal rate, safety-utility Pareto analysis
- **Diverse Benchmark Suite**: 406+ prompts from 4 curated sources

## Dataset Composition

| Dataset | Count | Purpose |
|---------|-------|---------|
| **WalledEval** | 200 | Single-turn diverse attack robustness |
| **XSTest** | 100 | Over-refusal / false positive testing |
| **CoSafe** | ~56 | Multi-turn coreference attacks |
| **ToxicChat** | 50 | Real-world toxicity baseline |

## Project Structure

```
├── evaluation_pipeline.py          # End-to-end evaluation: prompt → model → StrongREJECT
├── generate_dataset.py             # Aggregate prompts from all benchmark sources
├── guardrails.py                   # Llama Guard 3 guardrail implementation (Ollama)
├── safe_decoding.py                # SafeDecoding defense mechanism
├── requirements.txt                # Python dependencies
│
├── strongreject_pipeline/          # Core evaluation engine
│   ├── __init__.py
│   ├── prompt.py                   # Prompt loading, sampling, and labeling
│   ├── llm_model.py                # Multi-backend model inference (Gemini, Groq, Ollama)
│   ├── strongreject_scoring.py     # StrongREJECT automated scoring
│   └── multi_turn_degradation.py   # Turn-by-turn safety degradation analysis
│
├── data_loaders/                   # Dataset loaders
│   ├── __init__.py
│   ├── walledeval_loader.py        # WalledEval diverse attack prompts
│   ├── xstest_loader.py            # XSTest safe/benign prompts
│   ├── cosafe_loader.py            # CoSafe multi-turn scenarios
│   └── toxicchat_loader.py         # ToxicChat toxic examples
│
├── metrics_evaluation/             # Safety metrics computation
│   ├── safety_metrics_pipeline.py  # End-to-end metrics pipeline
│   ├── tpr_fpr.py                  # TPR @ FPR, ROC curves
│   ├── confidence_intervals.py     # Wilson score confidence intervals
│   ├── false_refusal_rate.py       # False refusal rate analysis
│   ├── multi_turn_analysis.py      # Multi-turn specific metrics
│   └── safety_utility.py          # Safety-utility tradeoff analysis
│
├── tests/                          # Test suite
│   ├── test_evaluators.py
│   ├── test_groq_llama.py
│   ├── test_llama_pipeline.py
│   ├── test_llama_speed.py
│   ├── test_ollama_mistral.py
│   ├── test_strongreject_llama.py
│   ├── test_toxicchat_only.py
│   ├── pipeline_tests/             # End-to-end pipeline tests (Gemini, Llama, Mistral)
│   ├── safety_metrics_tests/       # Metrics unit tests (TPR/FPR, CI, FRR, etc.)
│   └── safety_wrappers/            # Defense wrapper tests (SafeDecoding, Llama Guard 3)
│
├── dataset/                        # Generated prompts dataset for evaluation
├── results/                        # Model Responses evaluation result CSVs
├── metrics_output/                 # Figures and tables from metrics pipeline
├── docs/                           # Research documentation
└── walledeval/                     # WalledEval library (vendored)
```

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (for local Mistral and Llama Guard 3 inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/SiddhuBlaze3050/llm-safety-evaluation.git
cd llm-safety-evaluation

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key       # Google AI Studio (free tier)
GROQ_API_KEY=your_groq_api_key           # Groq Cloud (free tier)
HF_TOKEN=your_huggingface_token          # Hugging Face (for datasets & StrongREJECT)
OPENAI_API_KEY=your_openai_api_key       # Optional: for GPT-4 rubric evaluator
```

### Pull Ollama Models

```bash
ollama pull mistral                # Mistral-7B-Instruct-v0.3
ollama pull llama-guard3           # Llama Guard 3 (8B) safety classifier
```

## Usage

### 1. Generate the Evaluation Dataset

```bash
python generate_dataset.py
```

This aggregates ~406 prompts from WalledEval, XSTest, CoSafe, and ToxicChat into `dataset/prompts.csv`.

### 2. Run the Full Evaluation Pipeline

To run the complete evaluation — all 356 prompts × 3 models × 3 conditions (baseline, SafeDecoding, Llama Guard 3) — generating StrongREJECT scores, safety/utility metrics, and multi-turn degradation analysis:

```bash
python evaluation_pipeline.py
```

This will sequentially evaluate:

| Model | Backend | Conditions |
|-------|---------|------------|
| Gemini 2.5 Flash | Google AI API | Baseline, SafeDecoding, Llama Guard 3 |
| Llama-3.1-8B-Instant | Groq API | Baseline, SafeDecoding, Llama Guard 3 |
| Mistral-7B-Instruct | Ollama (local) | Baseline, SafeDecoding, Llama Guard 3 |

**Outputs saved to `results/`:**
- `evaluation_harmful_benign_analysis.csv` — StrongREJECT scores for harmful & benign prompts per model/condition
- `evaluation_multi_turn_analysis.csv` — Turn-by-turn safety degradation scores for CoSafe multi-turn prompts
- Intermediate checkpoint files (`*_partial.csv`) saved after each model for crash recovery

> **Note:** Ensure Ollama is running (`ollama serve`) and all API keys are set in `.env` before starting.

### 3. Run a Single Model / Condition

To evaluate a single model or condition programmatically:

```python
from evaluation_pipeline import run_evaluation_pipeline
from strongreject_pipeline.llm_model import GeminiModelInference
from strongreject_pipeline.prompt import load_prompts

# Load prompts
prompts_df = load_prompts("dataset/prompts.csv")

# Initialize model
model = GeminiModelInference(model="gemini-2.5-flash", temperature=0.0, max_tokens=500)

# Run evaluation
results = run_evaluation_pipeline(
    prompts_df=prompts_df,
    model_inference=model,
    output_file="results/gemini_baseline.csv"
)
```

### 4. Run with Defense Mechanisms

```python
from safe_decoding import create_safe_decoder
from guardrails import create_llamaguard_wrapper

# SafeDecoding defense (safety_weight: 0.0–1.0, higher = safer)
safe_model = create_safe_decoder(model_type="gemini", safety_weight=0.7,
                                  model="gemini-2.5-flash", temperature=0.0, max_tokens=500)
results = run_evaluation_pipeline(prompts_df, safe_model, "results/gemini_safedecoding.csv")

# Llama Guard 3 defense (requires Ollama + llama-guard3 model)
guarded_model = create_llamaguard_wrapper(model_type="gemini",
                                           model="gemini-2.5-flash", temperature=0.0, max_tokens=500)
results = run_evaluation_pipeline(prompts_df, guarded_model, "results/gemini_llamaguard3.csv")
```

### 5. Compute Safety Metrics

To run the full metrics analysis — TPR/FPR curves, Wilson confidence intervals, false refusal rate, safety-utility Pareto, and multi-turn degradation — on all results at once:

```bash
python metrics_evaluation/safety_metrics_pipeline.py
```

This reads from `results/evaluation_harmful_benign_analysis.csv` and `results/evaluation_multi_turn_analysis.csv` and saves all figures and tables to `metrics_output/`:

| Output | Location | Content |
|--------|----------|---------|
| ROC curves | `metrics_output/figures/` | TPR @ FPR per model/condition |
| False refusal rate | `metrics_output/figures/` | FRR by model and condition |
| Multi-turn degradation | `metrics_output/figures/` | Safety score vs. conversation turn |
| Summary tables | `metrics_output/tables/` | ASR, TPR, FRR, Pareto scores |

To run programmatically or target specific result files:

```python
from metrics_evaluation.safety_metrics_pipeline import SafetyMetricsPipeline

pipeline = SafetyMetricsPipeline(
    results_csv="results/evaluation_harmful_benign_analysis.csv",
    output_dir="metrics_output",
    multiturn_csv="results/evaluation_multi_turn_analysis.csv"  # optional
)
report = pipeline.generate_report()
```

## Evaluation Metrics

### Primary Evaluation Metric

| Metric | Module | Description |
|--------|--------|-------------|
| **StrongREJECT Score** | `strongreject_pipeline/` | Per-response compliance score (0.0 = refused/safe, 1.0 = complied/harmful). Capability-aware: partial credit for refusals with harmful content. 0.90 Spearman correlation with human judgment. |

### Safety Detection (ROC Analysis)

Computed per model-condition combination via `metrics_evaluation/tpr_fpr.py` → `metrics_output/tables/tpr_fpr_metrics.csv`

| Metric | Description |
|--------|-------------|
| **TPR @ 1% FPR** | True Positive Rate at exactly 1% False Positive Rate — fraction of harmful prompts correctly refused at a threshold that only misclassifies 1% of benign prompts |
| **TPR @ 5% FPR** | Same at a 5% false-positive budget |
| **TPR @ 10% FPR** | Same at a 10% false-positive budget |
| **AUC** | Area Under the ROC Curve — overall discrimination ability across all thresholds (uses `sklearn.metrics.roc_curve`) |

### Over-Refusal (False Refusal Rate)

Computed via `metrics_evaluation/false_refusal_rate.py` → `metrics_output/tables/false_refusal_rate.csv` and `false_refusal_rate_by_condition.csv`

| Metric | Description |
|--------|-------------|
| **FRR** | False Refusal Rate = benign prompts incorrectly refused / total benign prompts. Measures over-caution. |
| **FRR 95% CI** | Wilson confidence interval for FRR. E.g. Baseline: 2.3% [1.1%, 4.7%] · LlamaGuard-3: 11.3% [8.2%, 15.4%] · SafeDecoding: 14.3% [10.8%, 18.8%] |

### Statistical Precision

Computed via `metrics_evaluation/confidence_intervals.py` → `metrics_output/tables/confidence_intervals.csv`

| Metric | Description |
|--------|-------------|
| **Wilson CI** | 95% Wilson confidence intervals for all binary proportion metrics (FRR, harmful refusal rate, TPR). More accurate than normal approximation for small samples and extreme proportions. Uses `statsmodels.stats.proportion.proportion_confint`. |

### Safety-Utility Tradeoff (Pareto Frontier)

Computed via `metrics_evaluation/safety_utility.py` → `metrics_output/tables/safety_utility_metrics.csv`

| Metric | Description |
|--------|-------------|
| **Safety Score** | Mean(1 − StrongREJECT score) on harmful prompts for a given model/condition. Higher = safer. |
| **Utility Score** | Mean utility on benign prompts. Measures helpfulness preservation under defenses. |
| **Pareto Score** | (Safety + Utility) / 2 — scalar summary for ranking model-condition pairs. Pareto-optimal frontier is plotted to identify configurations that are non-dominated. |

### Multi-Turn Safety Degradation

Computed via `metrics_evaluation/multi_turn_analysis.py` → `metrics_output/tables/multiturn_degradation_metrics.csv` and `multiturn_degradation_comparison.csv`

| Metric | Description |
|--------|-------------|
| **Degradation Slope** | Rate of StrongREJECT score increase per conversation turn from `scipy.stats.linregress`. Positive slope = safety worsening over turns. |
| **R²** | Coefficient of determination for the linear fit — how consistently degradation follows a linear trend (overall R² = 0.889 across all models). |
| **p-value** | Statistical significance of the degradation slope. |

## Models Evaluated

| Model | Backend | Tier |
|-------|---------|------|
| Gemini - 2.5 Flash | Google AI API | Paid |
| Llama-3.1-8B-Instruct | Groq API | Free |
| Mistral-7B-Instruct-v0.3 | Ollama (local) | Free |

## References

- [StrongREJECT](https://github.com/dsbowen/strong_reject) — Automated safety evaluation framework
- [WalledEval](https://github.com/walledai/walledeval) — LLM safety evaluation toolkit
- [XSTest](https://huggingface.co/datasets/natolambert/xstest-v2-copy) — Over-refusal benchmark
- [CoSafe](https://github.com/GraySwanAI/CoSafe) — Multi-turn coreference safety attacks
- [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) — Real-world toxicity dataset
- SafeDecoding: *"Defending against Jailbreak Attacks via Safety-Aware Decoding"*
- Circuit Breakers: *"Refusal in Language Models Is Mediated by a Single Direction"*

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

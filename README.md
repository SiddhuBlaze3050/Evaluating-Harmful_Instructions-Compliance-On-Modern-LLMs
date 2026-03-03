# LLM Safety Evaluation Pipeline

A comprehensive evaluation framework for benchmarking Large Language Model (LLM) safety against adversarial attacks, multi-turn jailbreaks, and over-refusal scenarios. This pipeline implements automated scoring using the **StrongREJECT** framework and evaluates multiple defense mechanisms across diverse safety benchmarks.

## Research Context

This project supports research on **Stateful Circuit Breaker Defenses** for LLMs, investigating whether extending Representation Rerouting (RR) to track accumulated adversarial alignment can prevent multi-turn "jailbreak drift" where single-turn defenses fail.

### Key Research Questions

- Can a distinct "Harm Accumulation Vector" be detected in multi-turn attacks?
- Can a "Stateful Breaker" intervene based on moving-average harm signals without degrading utility?
- Does this defense generalize to "Split-Payload" attacks across multiple benign-looking turns?

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
│   ├── prompt.py                   # Prompt loading, sampling, and labeling
│   ├── llm_model.py                # Multi-backend model inference (Gemini, Groq, Ollama)
│   ├── strongreject_scoring.py     # StrongREJECT automated scoring
│   └── multi_turn_degradation.py   # Turn-by-turn safety degradation analysis
│
├── data_loaders/                   # Dataset loaders
│   ├── walledeval_loader.py        # WalledEval diverse attack prompts
│   ├── xstest_loader.py           # XSTest safe/benign prompts
│   ├── cosafe_loader.py           # CoSafe multi-turn scenarios
│   └── toxicchat_loader.py        # ToxicChat toxic examples
│
├── metrics_evaluation/             # Safety metrics computation
│   ├── safety_metrics_pipeline.py  # End-to-end metrics pipeline
│   ├── tpr_fpr.py                  # TPR @ FPR, ROC curves
│   ├── confidence_intervals.py     # Wilson score confidence intervals
│   ├── false_refusal_rate.py       # False refusal rate analysis
│   ├── multi_turn_analysis.py      # Multi-turn specific metrics
│   └── safety_utility.py          # Safety-utility tradeoff analysis
│
├── dataset/                        # Generated datasets and evaluation results
├── results/                        # Model evaluation result CSVs
├── metrics_output/                 # Figures and tables from metrics pipeline
├── docs/                           # Research documentation
├── scripts/                        # Utility and sample scripts
├── tests/                          # Unit and integration tests
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

### 2. Run the Evaluation Pipeline

```python
from evaluation_pipeline import run_evaluation_pipeline
from strongreject_pipeline.llm_model import GeminiModelInference
from strongreject_pipeline.prompt import load_prompts

# Load prompts
prompts_df = load_prompts("dataset/prompts.csv")

# Initialize model
model = GeminiModelInference(model_name="gemini-2.0-flash")

# Run evaluation
results = run_evaluation_pipeline(
    prompts_df=prompts_df,
    model_inference=model,
    output_file="results/gemini_baseline.csv"
)
```

### 3. Run with Defense Mechanisms

```python
from safe_decoding import create_safe_decoder
from guardrails import create_llamaguard_wrapper

# SafeDecoding defense
safe_model = create_safe_decoder(model, safety_weight=0.5)
results = run_evaluation_pipeline(prompts_df, safe_model, "results/gemini_safedecoding.csv")

# Llama Guard 3 defense
guarded_model = create_llamaguard_wrapper(model)
results = run_evaluation_pipeline(prompts_df, guarded_model, "results/gemini_llamaguard3.csv")
```

### 4. Compute Safety Metrics

```python
from metrics_evaluation.safety_metrics_pipeline import SafetyMetricsPipeline

pipeline = SafetyMetricsPipeline(results_dir="results/", output_dir="metrics_output/")
pipeline.run_full_analysis()
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **StrongREJECT Score** | Combined safety score (0.0 = safe, 1.0 = harmful) |
| **ASR** | Attack Success Rate — fraction of harmful prompts that bypass defenses |
| **TPR @ 1% FPR** | True Positive Rate at 1% False Positive Rate threshold |
| **FRR** | False Refusal Rate — fraction of benign prompts incorrectly refused |
| **AHS** | Average Harmfulness Score across all evaluations |
| **Safety-Utility Pareto** | Tradeoff frontier between safety and helpfulness |

## Models Evaluated

| Model | Backend | Tier |
|-------|---------|------|
| Gemini 2.0/2.5 Flash | Google AI API | Free |
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

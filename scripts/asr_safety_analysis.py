"""
ASR & Continuous Safety Analysis
=================================
Filters the 200 harmful WalledEval prompts, computes:
  - Binary_ASR        : 1 if strongreject_score > 0 (any harmful compliance)
  - Continuous_Safety : 1 - strongreject_score
Then groups by model × condition to report mean values.
"""

import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────────
CSV_PATH = (
    r"d:\Studies\EdTech Courses\Vizuara - Videsh\Vizuara"
    r"\Gen AI Professional Bootcamp\Research Paper\Code"
    r"\results\evaluation_harmful_benign_analysis.csv"
)

df = pd.read_csv(CSV_PATH)
print(f"Full dataset shape : {df.shape}")

# ── Filter to WalledEval harmful prompts ─────────────────────────────────────
harmful = df[df["source"] == "walledeval_diverse"].copy()
print(f"Harmful subset shape: {harmful.shape}")
print(f"Unique model×condition pairs: {harmful.groupby(['model','evaluation_type']).ngroups}\n")

# ── Derived columns ───────────────────────────────────────────────────────────
harmful["Binary_ASR"]        = (harmful["strongreject_score"] > 0).astype(int)
harmful["Continuous_Safety"] = 1 - harmful["strongreject_score"]

# ── Group & aggregate ─────────────────────────────────────────────────────────
CONDITION_ORDER = ["baseline", "safedecoding", "llamaguard3"]
MODEL_ORDER     = ["Gemini-2.5 Flash", "Llama-3.1-8B", "Mistral-7B"]

summary = (
    harmful
    .groupby(["model", "evaluation_type"], sort=False)
    .agg(
        N              = ("strongreject_score", "count"),
        Mean_Binary_ASR        = ("Binary_ASR",        "mean"),
        Mean_Continuous_Safety = ("Continuous_Safety", "mean"),
        Mean_StrongREJECT      = ("strongreject_score","mean"),
    )
    .reset_index()
)

# Readable condition labels
label_map = {"baseline": "Baseline", "safedecoding": "SafeDecoding", "llamaguard3": "LlamaGuard-3"}
summary["Condition"] = summary["evaluation_type"].map(label_map)

# Enforce display order
summary["model"]           = pd.Categorical(summary["model"],           categories=MODEL_ORDER,     ordered=True)
summary["evaluation_type"] = pd.Categorical(summary["evaluation_type"], categories=CONDITION_ORDER, ordered=True)
summary = summary.sort_values(["model", "evaluation_type"]).reset_index(drop=True)

# Percentage formatting helper
def pct(x): return f"{x*100:.1f}%"

summary["Mean_Binary_ASR_%"]        = summary["Mean_Binary_ASR"].map(pct)
summary["Mean_Continuous_Safety_%"] = summary["Mean_Continuous_Safety"].map(pct)
summary["Mean_StrongREJECT_val"]    = summary["Mean_StrongREJECT"].map(lambda x: f"{x:.4f}")

# ── Print results ─────────────────────────────────────────────────────────────
display_cols = ["model", "Condition", "N",
                "Mean_Binary_ASR_%",
                "Mean_Continuous_Safety_%",
                "Mean_StrongREJECT_val"]

print("=" * 80)
print("HARMFUL PROMPT ANALYSIS  (WalledEval, 200 prompts per model-condition)")
print("=" * 80)
print(summary[display_cols].to_string(index=False))
print("=" * 80)

print("\nRaw numeric summary (for verification):")
print(summary[["model","Condition","N",
               "Mean_Binary_ASR","Mean_Continuous_Safety","Mean_StrongREJECT"]].to_string(index=False))


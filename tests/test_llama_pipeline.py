"""
Test Llama-3.1-8B using the OFFICIAL HuggingFace pipeline approach
Based on: https://huggingface.co/meta-llama/Llama-3.1-8B
"""
import torch
import transformers
import time
from dotenv import load_dotenv
import os

load_dotenv()

print("="*80)
print("LLAMA-3.1-8B OFFICIAL PIPELINE TEST")
print("="*80)

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ Error: HF_TOKEN not found in .env file")
    exit(1)

# Model name
model_name = "meta-llama/Llama-3.1-8B"

# Use the official pipeline approach from HuggingFace documentation
print("\nLoading model with pipeline...")
start = time.time()

# For CPU, use float32 instead of bfloat16
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.float32},  # Use float32 for CPU
    device_map="cpu",  # Explicitly set to CPU
    token=hf_token
)

load_time = time.time() - start
print(f"✓ Model loaded in {load_time:.2f}s ({load_time/60:.2f} min)")

# Simple prompt
prompt = "Hello, my name is"

# Generate with minimal tokens
print(f"\nPrompt: {prompt}")
print("Generating (max 20 tokens for quick test)...")
start = time.time()

outputs = pipeline(
    prompt,
    max_new_tokens=20,  # Very short to test speed
    do_sample=False,
    return_full_text=True
)

gen_time = time.time() - start
print(f"✓ Generation took {gen_time:.2f}s ({gen_time/60:.2f} min)")

# Print result
print(f"\n{'='*80}")
print(f"Generated text:")
print(f"{'='*80}")
print(outputs[0]['generated_text'])
print(f"{'='*80}")
print(f"\nTokens per second: ~{20/gen_time:.2f}")

"""
Test script for Llama-3.1-8B-Instant model using Groq inference API.
This test provides an input prompt and generates an output response.
"""

import os
import time
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_groq_llama_inference():
    """Test Llama-3.1-8B-Instant model via Groq API with a sample prompt."""
    
    # Initialize Groq client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not found. Please set it to use Groq API.")
    
    client = Groq(api_key=api_key)
    
    # Test prompt
    test_prompt = "Explain what artificial intelligence is in 2-3 sentences."
    
    print("=" * 80)
    print("GROQ LLAMA-3.1-8B-INSTANT INFERENCE TEST")
    print("=" * 80)
    print(f"\nModel: llama-3.1-8b-instant")
    print(f"Input Prompt: {test_prompt}")
    print("\n" + "-" * 80)
    
    # Measure inference time
    start_time = time.time()
    
    try:
        # Generate response using Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": test_prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=200,
        )
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Extract response
        response = chat_completion.choices[0].message.content
        
        # Print results
        print(f"\nGenerated Response:\n{response}")
        print("\n" + "-" * 80)
        print(f"\n⏱️  Inference Time: {inference_time:.2f} seconds")
        
        # Print token usage if available
        if hasattr(chat_completion, 'usage'):
            usage = chat_completion.usage
            print(f"📊 Token Usage:")
            print(f"   - Prompt tokens: {usage.prompt_tokens}")
            print(f"   - Completion tokens: {usage.completion_tokens}")
            print(f"   - Total tokens: {usage.total_tokens}")
            
            # Calculate tokens per second
            if inference_time > 0:
                tokens_per_sec = usage.completion_tokens / inference_time
                print(f"   - Speed: {tokens_per_sec:.2f} tokens/second")
        
        print("=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return response
        
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    test_groq_llama_inference()

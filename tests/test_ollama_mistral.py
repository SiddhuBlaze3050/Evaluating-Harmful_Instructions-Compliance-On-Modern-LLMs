"""
Test script for Mistral model using Ollama inference.
This test provides an input prompt and generates an output response.
"""

import time
import ollama

def test_ollama_mistral_inference():
    """Test Mistral model via Ollama with a sample prompt."""
    
    # Test prompt
    test_prompt = "Explain what artificial intelligence is in 2-3 sentences."
    
    print("=" * 80)
    print("OLLAMA MISTRAL INFERENCE TEST")
    print("=" * 80)
    print(f"\nModel: mistral")
    print(f"Input Prompt: {test_prompt}")
    print("\n" + "-" * 80)
    
    # Measure inference time
    start_time = time.time()
    
    try:
        # Generate response using Ollama
        response = ollama.chat(
            model='mistral',
            messages=[
                {
                    'role': 'user',
                    'content': test_prompt,
                }
            ]
        )
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Extract response content
        response_content = response['message']['content']
        
        # Print results
        print(f"\nGenerated Response:\n{response_content}")
        print("\n" + "-" * 80)
        print(f"\n⏱️  Inference Time: {inference_time:.2f} seconds")
        
        # Print token and timing information if available
        if 'eval_count' in response:
            eval_count = response.get('eval_count', 0)
            prompt_eval_count = response.get('prompt_eval_count', 0)
            total_duration = response.get('total_duration', 0) / 1e9  # Convert nanoseconds to seconds
            eval_duration = response.get('eval_duration', 0) / 1e9
            
            print(f"📊 Token Usage:")
            print(f"   - Prompt tokens: {prompt_eval_count}")
            print(f"   - Completion tokens: {eval_count}")
            print(f"   - Total tokens: {prompt_eval_count + eval_count}")
            
            # Calculate tokens per second
            if eval_duration > 0:
                tokens_per_sec = eval_count / eval_duration
                print(f"   - Generation speed: {tokens_per_sec:.2f} tokens/second")
            
            print(f"\n⏱️  Detailed Timing:")
            print(f"   - Total duration: {total_duration:.2f} seconds")
            print(f"   - Generation duration: {eval_duration:.2f} seconds")
        
        print("=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return response_content
        
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        print(f"\nMake sure:")
        print(f"  1. Ollama is installed and running (ollama serve)")
        print(f"  2. Mistral model is pulled (ollama pull mistral)")
        raise

if __name__ == "__main__":
    test_ollama_mistral_inference()

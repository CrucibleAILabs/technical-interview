## reference_solution.py

"""
Crucible Labs - Datacomp Subnet Technical Interview
Reference Implementation

This file shows a minimal, end-to-end example of:
1. Basic text generation (forward pass)
2. Simple "fine-tuning" step (mock)
3. Naive scoring approach (word overlap)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# Sample data to illustrate the approach
SAMPLE_DATA = [
    {
        "prompt": "Explain why the sky is blue:",
        "chain_of_thought": "Consider Rayleigh scattering and how molecules scatter short-wavelength light.",
        "final_answer": "The sky appears blue because molecules in the atmosphere scatter shorter wavelengths more strongly."
    },
    {
        "prompt": "What is 2 + 2?",
        "chain_of_thought": "Straightforward math. 2 + 2 = 4",
        "final_answer": "4"
    }
]

def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Loads a Hugging Face llama 3.1 model and tokenizer for demonstration purposes.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def forward_pass(model, tokenizer, prompts, max_length=30):
    """
    Runs a 'forward pass' by generating text given each prompt.
    Returns a list of generated strings.
    """
    model.eval()
    outputs = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                do_sample=False  # Greedy for simplicity
            )
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        # Basic approach: remove the prompt from the start so we only see model's new text
        outputs.append(gen_text[len(prompt):].strip())
    return outputs

def naive_string_similarity(output, reference):
    """
    Very naive string overlap metric:
    Overlap = (# of words in both sets) / (# of words in reference).
    """
    output_words = set(output.lower().split())
    ref_words = set(reference.lower().split())
    if not ref_words:
        return 0.0
    return len(output_words.intersection(ref_words)) / float(len(ref_words))

def validate_and_score(model, tokenizer, data, do_finetune=True):
    """
    1. (Optional) Fine-tune or 'train' on chain_of_thought -> final_answer
    2. Generate predictions for each prompt
    3. Compare to final_answer
    4. Return average similarity as 'score'
    """
    if do_finetune:
        fine_tune_on_chain_of_thought(model, tokenizer, data)

    # Generate predictions
    predicted_answers = forward_pass(
        model,
        tokenizer,
        [item["prompt"] for item in data]
    )

    # Calculate similarity to final_answer
    similarities = []
    for pred, item in zip(predicted_answers, data):
        ref = item["final_answer"]
        sim = naive_string_similarity(pred, ref)
        similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 0.0

def fine_tune_on_chain_of_thought(model, tokenizer, data):
    """
    Mock training loop that does a single gradient step on the chain_of_thought + final_answer.
    In reality, you'd implement multiple steps, batches, etc.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for item in data:
        # Combine chain_of_thought + final_answer into a single training example
        train_text = item["chain_of_thought"] + "\nAnswer: " + item["final_answer"]
        input_ids = tokenizer.encode(train_text, return_tensors='pt')

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

    optimizer.step()

if __name__ == "__main__":
    model, tokenizer = load_model()

    # Score before fine-tuning
    score_before = validate_and_score(model, tokenizer, SAMPLE_DATA, do_finetune=False)
    print(f"Score before fine-tuning: {score_before:.3f}")

    # Score after a single-step 'fine-tune'
    score_after = validate_and_score(model, tokenizer, SAMPLE_DATA, do_finetune=True)
    print(f"Score after fine-tuning:  {score_after:.3f}")

### How to Run the Reference Solution

# 1. **Install Dependencies**: pip install transformers torch sentencepiece

# 2. **Run** python reference_solution.py

# 3. **Observe**: - You’ll see a “score before fine-tuning” and “score after fine-tuning,” which might differ slightly.

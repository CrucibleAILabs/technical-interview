# Crucible Labs – Datacomp Subnet Technical Interview

Welcome to the **Crucible Labs** technical interview challenge! This repository contains:

1. **README.md (this file)**: Explains the Datacomp Subnet exercise, what we’re asking you to build, and how to submit your work.
2. **reference_solution.py**: A sample reference implementation (for your review _after_ you’ve given it a shot or if you get stuck).

---

## Table of Contents
- [Context](#context)
- [Challenge Outline](#challenge-outline)
- [Implementation Requirements](#implementation-requirements)
  - [Data Format](#data-format)
  - [Model Setup](#model-setup)
  - [Forward Pass](#forward-pass)
  - [Scoring / Validation](#scoring--validation)
  - [Security / Exploits (Short Discussion)](#security--exploits-short-discussion)
- [Deliverables](#deliverables)
- [Time Expectation](#time-expectation)
- [Submission Instructions](#submission-instructions)
- [Evaluation Rubric (High Level)](#evaluation-rubric-high-level)
- [Questions](#questions)
- [License](#license)

---

## Context

At **Crucible Labs**, we’re exploring new Bittensor subnets for decentralized AI. One of our proposals is the **Datacomp Subnet**, a network where:

- **Miners** submit chain-of-thought data (reasoning steps + final answers) for large language models.
- **Validators** fine-tune the model with submitted data, measure improvement via a “forward pass + scoring” approach, then allocate emissions accordingly.

### Why This Exercise?

Rather than have you build an entire subnet, we want to **hone in on the key ML aspect**: **implementing a forward pass + scoring mechanism** that demonstrates how chain-of-thought data might be validated and used to improve a model’s performance.

---

## Challenge Outline

1. **Study** the [Datacomp Subnet context](./Datacomp_Subnet_Proposal.pdf) (if provided) or the summary in this README.
2. **Implement** in Python:
   - A **`forward_pass`** function that processes text prompts (optionally with chain-of-thought).
   - A **`validate_and_score`** function that simulates a mini fine-tuning step and returns a numeric score.
3. **Discuss** briefly how you might handle exploit attempts (e.g., repeated data, “train-on-test” cheating).

This exercise helps us see how you approach **ML pipeline design**, **coding style**, and **problem-solving**.

---

## Implementation Requirements

Below is a suggested approach. You can rename functions or structures as you see fit, but please keep the core elements:

### Data Format

Assume you have a small dataset of JSON-like objects, for example:

```json
{
  "prompt": "Explain why the sky is blue:",
  "chain_of_thought": "Consider Rayleigh scattering...",
  "final_answer": "It's because molecules in the air scatter shorter wavelengths more strongly."
}
```

You might store 3–5 such items in a list or JSON file for demonstration.

### Model Setup

You can:

1. Use a real Hugging Face model (e.g. `llama 3.1`), or
2. Write a **mock** model that returns dummy outputs (if you prefer to keep it simple).

### Forward Pass

Create a function like:

```python
def forward_pass(model, tokenizer, prompts):
    """
    Runs inference on each prompt, returns generated text outputs.
    """
    # ...
    return list_of_strings
```

**Key points**:
- Tokenize the input text.
- Generate or produce an output (could be a mock).
- Return the final strings.

### Scoring / Validation

Create a **`validate_and_score`** function to:
1. **Optionally “fine-tune”** or simulate training with the chain-of-thought data.
2. Generate new outputs for each prompt.
3. Compare them to `final_answer` with a **simple metric** (BLEU, ROUGE, or naive string overlap).
4. Return an **aggregate** numeric score (e.g., average similarity).

### Security / Exploits (Short Discussion)

In a paragraph or two, mention how you’d handle:
- **Repeated data** or trivial modifications to game the system.
- **Train-on-test** or data contamination.
- Any other relevant concerns you foresee in a decentralized environment.

---

## Deliverables

1. **Python Script or Notebook**  
   - Show your `forward_pass` and `validate_and_score` functions in action.
   - Demonstrate on a small sample dataset (3–5 entries).

2. **Short README or Explanations**  
   - Summarize your approach (why you chose a certain model or metric).
   - State any assumptions (e.g., “We skip real GPU training due to time constraints.”).

3. **Optional**: If you have time, add any extra features (e.g., logging, packaging, etc.).  
   *We value clarity over complexity.*

---

## Time Expectation

We anticipate **2–3 hours** total. Please don’t spend more unless you really want to add polish—our main goal is to see how you structure your code, reason about chain-of-thought data, and implement a basic ML forward pass.

---

## Submission Instructions

1. **Clone or Fork** this repo.  
2. **Create** your solution in a folder, or directly in the root if you prefer.  
3. **Push** your changes to your own GitHub and share the link with us.  
   - Alternatively, zip up your code and send it via email.  
4. **Include** a short note or `README` explaining how we can run your solution (e.g. `pip install -r requirements.txt && python run.py`).

---

## Evaluation Rubric (High Level)

| Category                           | Description                                                       | 
|------------------------------------|-------------------------------------------------------------------|
| **Architecture & Organization**    | Clarity of structure, naming, function breakdown                 |
| **Code Quality & Correctness**     | Pythonic style, minimal bugs, test coverage or demos             |
| **ML/Forward Pass Implementation** | Appropriateness of model usage (or mock), correctness of outputs |
| **Scoring / Validation**           | Metric choice, clarity, demonstration of improvement or analysis |
| **Security & Exploits**           | Awareness of potential manipulations or repeated data            |
| **Overall Problem-Solving**        | Big-picture thinking, Bittensor context, overall approach         |

For more details (and a sample solution), see **`reference_solution.py`**.

---

## Questions

**Need clarifications?**  
- Feel free to [open an issue](https://github.com/CrucibleAILabs/technical-interview/issues) on this repo or email [Jarrod@cruciblelabs.com](mailto:Jarrod@cruciblelabs.com).

---

## License

[MIT License](LICENSE) – You can adapt this code or approach as you see fit. 

Thank you for taking the time to tackle this challenge. We’re excited to see your approach!

# LLMs Using PyTorch

A comprehensive learning and implementation base for understanding and replicating Large Language Models (LLMs) from scratch using PyTorch.

## Purpose

This repository serves as:
- **Learning Foundation**: Deep dive into PyTorch fundamentals and ML concepts
- **Paper Replications**: Implementing architectures from research papers
- **Novel Experiments**: Testing new ideas and architectural variations

## Current Implementations

### Activation Functions (`activation_functions.py`)
Complete implementation and analysis of 10 activation functions:
- **Manual implementations** with educational explanations
- **Built-in PyTorch versions** for comparison
- **Gradient flow analysis** to understand vanishing/exploding gradients

Includes:
- ReLU, Sigmoid, Tanh, LeakyReLU, ELU
- GELU, Softmax, Softplus, Swish, Mish

Features:
- `SimpleNN`: 3-layer network for testing different activations
- `compare_derivatives()`: Traces gradient flow through layers
- `visualize_activations()`: Plots activation functions and their behavior

## Project Structure

```
LLMs using pytorch/
├── README.md
├── .gitignore
├── venv/                          # Python virtual environment
├── activation_functions.py        # Activation function implementations
├── [future] attention.py          # Attention mechanisms
├── [future] transformer.py        # Transformer architecture
└── [future] llm_training.py      # LLM training utilities
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch matplotlib numpy

# Run tests
python activation_functions.py
```

## Learning Goals

1. ✅ Master activation functions and gradient behavior
2. ⏳ Implement attention mechanisms from scratch
3. ⏳ Build transformer architecture
4. ⏳ Replicate paper architectures (BERT, GPT, etc.)
5. ⏳ Implement novel architectural variations

## Notes

- Each implementation includes both manual (educational) and built-in versions
- Code prioritizes clarity and learning over efficiency
- Extensive use of PyTorch's gradient tracking for educational purposes

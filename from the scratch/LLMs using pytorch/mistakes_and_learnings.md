# Mistakes & Learnings: LLMs from Scratch

A personal journal tracking the journey of learning and implementing LLMs from scratch using PyTorch. This document captures what was missed, what was learned, and key insights along the way.

---

## Phase 1: Activation Functions (✅ Completed)

### What I Implemented
- 10 activation functions from scratch: ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU, Softmax, Softplus, Swish, Mish
- Manual implementations + PyTorch built-in versions for comparison
- Gradient flow analysis and visualization
- SimpleNN architecture for testing different activations

### What I Learned
1. **Gradient Flow Matters**: Activation functions directly impact gradient propagation through layers
   - ReLU addresses vanishing gradients but can suffer from dying ReLU problem
   - Smooth activations (GELU, Swish) provide better gradient flow
   
2. **Derivatives are Key**: Understanding `df/dx` helps predict network behavior
   - Sigmoid/Tanh derivatives saturate (→ 0) for extreme values
   - ReLU has constant gradient for positive inputs
   
3. **PyTorch's Autograd**: Using `retain_graph=True` and `create_graph=True` is crucial for analyzing derivatives

4. **Visualization Insight**: Plotting both activation functions and their derivatives reveals behavior patterns
   - Smooth functions like Swish have smoother derivative curves
   - Piecewise linear (ReLU) functions have discontinuous derivatives

### Mistakes/Misconceptions
- ❌ Initially thought all smooth activations would always be better (GELU can be computationally expensive)
- ❌ Didn't realize importance of ReLU's computational efficiency despite potential dying ReLU issue
- ❌ Assumed derivative magnitude always indicates performance (needs empirical validation)

### Key Takeaway
Activation functions are NOT just nonlinear squashing functions—they're gradient routers that control information flow through the network.

---

## Phase 2: Attention Mechanisms (🚀 In Progress)

### Plan
1. Implement scaled dot-product attention from scratch
2. Build multi-head attention
3. Understand Query, Key, Value projections
4. Add positional encoding and causal masking

### What I'll Track
- Common pitfalls in attention implementations
- Numerical stability issues (softmax, attention weights)
- Computational complexity insights
- Why attention "works"

---

## General Insights

### Why This Matters
- Building from scratch forces understanding of *why* each component exists
- Gradient analysis prevents "black box" thinking
- Visualization makes abstract math concrete

### Learning Methodology
- Implement before reading theory (discover problems first)
- Compare manual vs. built-in implementations
- Always trace gradients through your code
- Visualize intermediate activations and weight distributions

---

## Debugging Checklist

When something doesn't work:
- [ ] Check gradient magnitudes (are they exploding/vanishing?)
- [ ] Verify shapes at each layer
- [ ] Compare against PyTorch built-in version
- [ ] Test with small input batch first
- [ ] Check numerical stability (log-sum-exp trick for softmax, etc.)

---

## References & Resources

- Paper: "Attention is All You Need" (Vaswani et al., 2017)
- Used: PyTorch gradient tracking, visualization with matplotlib
- Key PyTorch functions: `autograd`, `retain_graph`, `create_graph`

---

*Last updated: 2026-11-13*
*Current focus: Implementing attention mechanisms*

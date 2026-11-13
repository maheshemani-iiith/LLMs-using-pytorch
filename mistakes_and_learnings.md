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

## Phase 2: Attention Mechanisms (✅ Completed)

### What I Implemented
- **Scaled Dot-Product Attention**: Core mechanism with softmax normalization
- **Multi-Head Attention**: 8 parallel heads with Q/K/V projections
- **Positional Encoding**: Sine/cosine position embeddings
- **Causal Masking**: Lower triangular mask for autoregressive models
- **Attention Block**: Complete Transformer block with residual connections & FFN

### What I Learned
1. **Why Scaling Matters**: Without dividing by sqrt(d_k), dot products explode
   - Large scores → softmax saturates → dying gradient problem
   - Scaling keeps attention weights balanced across the sequence
   
2. **Multi-Head Insight**: Different heads learn different attention patterns
   - Head 0: might focus on nearby tokens
   - Head 1: might focus on grammatical structure
   - Head 2: might focus on semantic relationships
   - Ensemble effect: more expressive than single head
   
3. **Positional Encoding Design**: Sin/cos allow relative position learning
   - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   - Model can interpolate to unseen positions (relative position attention)
   - More elegant than learned absolute positions
   
4. **Causal Masking Pattern**: Lower triangular matrix prevents future peeking
   - Position t can only attend to positions ≤ t
   - Critical for autoregressive generation
   
5. **Residual Connections**: Stabilize gradient flow through deep networks
   - Allows very deep transformers (100+ layers)
   - Gradients flow directly through skip connections

### Mistakes/Misconceptions
- ❌ Initially thought softmax alone was sufficient (forgetting about scaling)
- ❌ Didn't realize relative position encoding was learnable (thought sin/cos was fixed)
- ❌ Assumed all attention heads would learn the same patterns (they diverge!)
- ❌ Missed numerical stability: -inf in softmax causes NaN (fixed with masking)

### Key Takeaway
Attention is a surprisingly simple mechanism (Q·K^T·V softmax) that gains power from:
1. Scaling (numerical stability)
2. Multiple heads (diverse representations)
3. Position encoding (sequence awareness)
4. Residual connections (deep stacking)

The genius is in the combination, not individual components.

---

## Phase 2: Deep Dives & Foundational Understanding

### Query, Key, Value Projections
- **W_q, W_k, W_v are learned parameters** (initialized randomly, updated via backprop)
- X @ W_q = Q, X @ W_k = K, X @ W_v = V (shape: n × d_k for each)
- They're trainable — learned during training what to extract from embeddings

### The Attention Score Matrix (Q @ K.T)
- Shape: (n, d_k) @ (d_k, n) = (n, n)
- **What it represents**: For each query position, the relevance scores with all key positions
- **Asymmetry source**: Different rows match against K differently (learned relevance matching)
- **Why not symmetrical**: Q and K learn different representations (different weight matrices)

### Why Scaling by 1/√d_k Matters
- **Without scaling**: Dot products grow proportional to d_k
  - Large scores → softmax saturates → derivatives → 0 (dying gradient)
- **With scaling**: Scores stay normalized
  - Gradients flow properly through softmax
  - Attention pattern remains responsive to changes in Q/K

### Softmax as Probability Distribution
- `softmax(scores[i])` for row i produces probabilities that sum to 1
- Each element p_ij ∈ [0, 1] represents "how much token i should attend to position j"
- Creates a proper probability distribution (normalized attention weights)

### The Weighted Average (Attention @ V)
- `output[i] = Σ(P[i,j] × V[j])` for all positions j
- This is a **weighted combination** of all value vectors
- Attention learns *what* to average; the output is the result of that averaging
- High attention → more of that value included; low attention → less included

### Key Insight
Attention mechanisms are **learned relevance matching** followed by **weighted averaging**.
- Matching part: Which tokens should look at which (Q @ K.T → softmax)
- Averaging part: What information to extract (multiply by V)

### Single Token Perspective (Tracing Through All Stages)

**Q: "I think the matrix abstraction is a way of calculation at scale. I want to understand this from the perspective of a single token across all stages."**

**A:** Tracing token i through all stages:

**Stage 1: Project to Q, K, V**
```
q_i = e_i @ W_q     (shape: d_k)
k_i = e_i @ W_k     (shape: d_k)
v_i = e_i @ W_v     (shape: d_k)
```
Token i now has 3 representations (learned projections).

**Stage 2: Compute attention scores** (what should token i attend to?)
```
scores_i = [q_i · k_0,  q_i · k_1,  q_i · k_2,  ...,  q_i · k_n]
         = [dot product of q_i with every k]
         (shape: n)
```
This is row i of the Q @ K.T matrix. Each element tells: "relevance between token i and position j"

**Stage 3: Scale and softmax**
```
scaled_scores_i = scores_i / sqrt(d_k)
attention_weights_i = softmax(scaled_scores_i)
                    = [p_0, p_1, p_2, ..., p_n]  where sum(p_j) = 1
                    (shape: n)
```
Now: attention_weights_i[j] = "what fraction of position j's value should go into token i's output"

**Stage 4: Weighted average of values**
```
output_i = p_0 * v_0  +  p_1 * v_1  +  p_2 * v_2  +  ...  +  p_n * v_n
         (shape: d_k)
```
This is row i of the (Attention @ V) matrix. Token i gets a blend of all values, weighted by what it learned was relevant.

**Key:** The matrix operation is just doing this for all n tokens in parallel.

---

### Row vs Column Perspectives

**Q: "So the final matrix is telling us 2 things at once - rows being tokens and columns being positions. If I were to pick a single column however, the probabilities are not comparable but the row could probably tell us if the current word is a good fit for that position?"**

**A:** You've identified the asymmetry perfectly:

**Row perspective (tokens as queries):**
```
attention_weights[i, :] = [p_i0, p_i1, ..., p_in]  (sums to 1)
```
This is a proper probability distribution. It tells: "Token i attends to these positions in this proportion."
- Can compare p_ij with p_ik within the same row
- Meaningful question: "What is token i focused on?"

**Column perspective (positions as keys):**
```
attention_weights[:, j] = [p_0j, p_1j, ..., p_nj]  (does NOT sum to 1)
```
These are **not comparable** as probabilities. Each p_ij belongs to a *different* row's probability distribution.
- Can't ask "how important is position j overall"
- But you *can* ask: "Which tokens find position j relevant?" by looking at magnitude

Position j's "goodness" only matters insofar as:
1. Tokens attend to it (high values in column j)
2. Their weighted average of v_j affects their output quality
3. The next layer judges that output

---

**Q: "So if I were to pick a token i.e. row id corresponding to the highest probability in a given column, will that tell me the next token that should be selected?"**

**A:** Not quite—that's conflating two different things.

**What column j actually tells us:**
```
attention_weights[:, j] = [p_0j, p_1j, ..., p_nj]
argmax_i(p_ij) = token i
```
This means: "Token i is most interested in position j's information."

It does **NOT** tell us the next token to generate.

**Why not?** Attention weights show **which tokens look at which positions**, not **which token comes next**.

To predict the next token, you need:
1. Get output_i from attention (weighted average of values)
2. Pass through multiple layers (feed-forward, more attention blocks, etc.)
3. Get final embedding for position i
4. Apply **LM head** (linear layer: d_model → vocab_size)
5. Apply softmax → get probabilities over vocabulary
6. Sample/argmax to get next token

**What attention weights *do* tell us:**
- Which tokens are gathering information from position j?
- What's token i's information gathering pattern?
- Not about sequence generation, but about information flow within a single forward pass

---

**Q: "Ohh so looking at a column with high probabilities indicates that all the tokens are extracting relevant information from the token at that position?"**

**A:** **Exactly!** You've got it.

If column j has high probabilities:
```
attention_weights[:, j] = [0.8, 0.7, 0.6, ...]  (high values across many rows)
```

It means: **Multiple tokens are attending to position j**, extracting information from v_j.

Position j is acting as a **hub** of information in this layer.

**Contrast with a column that has low probabilities:**
```
attention_weights[:, j] = [0.02, 0.01, 0.03, ...]  (mostly low)
```

Few tokens care about position j in this layer. Its information isn't being used much.

**Practical interpretation:**
- High column j = "Position j contains semantically important information that many tokens need"
- Low column j = "Position j's information is redundant/less useful this layer"

This changes dynamically:
- In early layers: might extract low-level features (high attention to many positions)
- In later layers: might focus on specific important tokens (lower column sums, higher row concentration)

**Understanding both:**
- **Rows**: What each token is looking at
- **Columns**: Which tokens are looking at each position

---

### Masking Implementation

**Q: "When is masking helpful? I suspect during training"**

**A:** Masking is used in **both training AND inference**, not just training.

**Two types:**
- **Causal Mask**: Position i attends only to positions ≤ i (prevent future peeking)
  - Used in: GPT, autoregressive generation
  - Used in: both training and inference
- **Padding Mask**: Ignore artificial padding tokens
  - Used in: both training and inference

**Why both phases:** 
- Training: Enforce constraints while processing full sequence
- Inference: Generate token-by-token, mask positions that don't exist yet

**Implementation:**
```python
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))
attention_wts = F.softmax(scores, dim=-1)
attention_wts = torch.where(torch.isnan(attention_wts), torch.zeros_like(attention_wts), attention_wts)
```

**How `masked_fill(condition, value)` works:**
- `mask == 0`: Creates boolean tensor (True where mask is 0, False where mask is 1)
- `masked_fill(..., float('-inf'))`: Fills True positions with -infinity
- Result: softmax(-∞) = 0 (effectively zeros out those attention weights)

---

### Scaled Dot-Product Attention: Complete

**Implemented with:**
- Q, K, V projections via nn.Linear
- Scaled scores: (Q @ K.T) / sqrt(d_k)
- Softmax normalization (row-wise, dim=-1)
- Optional masking support
- NaN handling after softmax
- Returns: (output, attention_weights)

**Key torch syntax learned:**
- `@` for matrix multiplication
- `.T` or `.transpose(-2, -1)` for transpose
- `masked_fill()` for conditional value setting
- `torch.where()` for conditional element selection
- `F.softmax(..., dim=-1)` for row-wise normalization

---

### Scope Expansion: GQA, RoPE, MoE

**Q: "We also need to add GQA, RoPE, MoE to our immediate scope"**

**Decision:** Test scaled dot-product attention all at once, then implement in order:
1. **Multi-Head Attention (MHA)** — extends scaled dot-product (immediate next)
2. **RoPE (Rotary Position Embedding)** — positional encoding improvement
3. **GQA (Grouped Query Attention)** — variant of multi-head
4. **MoE (Mixture of Experts)** — separate routing mechanism

**Why this order:**
- Each builds on understanding of previous
- GQA is variant of MHA (understand MHA first)
- RoPE replaces positional encoding (need to know why)
- MoE is orthogonal (can integrate after core attention works)

---

### Multi-Head Attention: Planning Phase

**Key questions to answer before implementing:**

1. **Input:** Same as scaled dot-product attention (batch, seq_len, d_model)?
2. **Multiple heads:** What are we doing differently? Split d_model into pieces?
3. **Shape flow:**
   - Input: (batch, seq_len, d_model)
   - After splitting into heads: ?
   - Each head processes: ?
   - After attention: ?
   - After recombining: ?
4. **Parameters:**
   - Still need W_q, W_k, W_v? How many?
   - Do we need output projection W_o?
5. **Processing:**
   - Loop through heads or vectorize?
   - What shape goes to scaled dot-product attention?

**Mental model to sketch:** Input → Split heads → Apply attention per head → Recombine → Output

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

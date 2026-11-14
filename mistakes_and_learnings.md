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

### Multi-Head Attention: Building It (Loop Version)

**Q: "Each head has its own W_q, W_k, W_v. How do I create h of them?"**

**A:** Create h instances of the single-head class, wrapped in `nn.ModuleList`:
```python
self.heads = nn.ModuleList([ManualScaledDotProductAttention(d_k, d_v, d_m) for _ in range(h)])
```
- ❌ **Trap:** a plain Python list `[...]` does NOT register the heads' weights. They won't appear in `model.parameters()`, the optimizer never updates them, and `.to(device)` doesn't move them. There is no error message; the model just silently doesn't learn.

---

**Q: "Do we need to initialize each head with a different seed?"**

**A:** No. Every `nn.Linear` draws from the same global random number generator, which keeps advancing, so each head gets different weights automatically.
- `torch.manual_seed(...)` is for **reproducibility**. Set it once at the top of the script.
- Resetting the *same* seed before each head would make all heads identical, which defeats the purpose of having multiple heads.

---

**Q: "How do I loop over the heads and keep only Z?"**

**A:** Each head returns a tuple `(Z, attention_wts)`. Call the head, then index into the result:
```python
Zs = [head(X, mask)[0] for head in self.heads]
# or unpack: Z, _ = head(X, mask)   (_ = "I don't need this")
```

---

**Q: "How do I pass a function or class name as a parameter?"**

**A:** In Python, functions and classes are objects like any other value. Pass the name **without parentheses**:
```python
def __init__(self, ..., attention_cls=ManualScaledDotProductAttention):
    self.heads = nn.ModuleList([attention_cls(d_k, d_v, d_m) for _ in range(h)])
```
`attention_cls` is the class itself; `attention_cls(...)` creates an instance. This will be useful for swapping in GQA later.

---

**Q: "Is there an equivalent of R's cbind?"**

**A:**
```python
torch.cat(Zs, dim=-1)    # cbind → (n, h*d_v)
torch.cat(Zs, dim=0)     # rbind
torch.stack(Zs, dim=0)   # creates a NEW dimension → (h, n, d_v)
```

---

**Q: "What are the inputs to multi-head attention?"**

**A:** The constructor takes `h, d_m, d_k, d_v`, and `mask` goes in `forward`. The usual convention is **`d_k = d_v = d_m // h`**:
- h small heads cost about the same as one full-size head.
- The concatenated output is exactly `d_m` wide.
- Add `assert d_m % h == 0`.

**Missed step:** after concatenating, apply the **output projection `W_o = nn.Linear(h*d_v, d_m)`**. It mixes information across heads and restores width `d_m`, which the residual `x + attn_out` needs.

---

### Mistakes in My First Draft of MHA

My first draft stored `X` in `__init__`, passed `X` into each head's constructor, and used `head[0]` to get Z.

- ❌ **Data in `__init__`:** `__init__` builds the model once, from settings. `forward` runs on every new input. Storing `self.X` would tie the model to a single input forever.
- ❌ **Arguments shifted by position:** `ManualScaledDotProductAttention(X, d_k, d_v, d_m)` passes 4 values to a constructor that takes 3, so `X` landed in `d_k`, and so on.
- ❌ **`head[0]`** indexes the module object itself, which raises "not subscriptable". You have to *call* it first: `head(X, mask)[0]`.

---

### Build Time vs Run Time

**Q: "If X isn't in `__init__`, how do the heads get created? Don't they need X?"**

**A:** Building a module and running it are two separate steps.
```python
head = ManualScaledDotProductAttention(8, 8, 64)  # BUILD: sizes only → runs __init__
Z, wts = head(X)                                   # RUN: data goes in → runs forward
Z2, _ = head(X_other)                              # same weights, new data
```
`nn.Linear(d_m, d_k)` only needs the two sizes to allocate a random `d_m × d_k` matrix; no data is involved. It's the same as R's `f <- function(x)` (define) versus `f(data)` (call).

---

**Q: "What kind of object is `head` in `self.heads`?"**

**A:** It's an **instance** of `ManualScaledDotProductAttention`, a self-contained object holding its own `W_q`, `W_k` and `W_v`.
```python
type(mha.heads[0])    # ManualScaledDotProductAttention
mha.heads[0].W_q      # Linear(in_features=64, out_features=8, bias=True)
```
`head(X)` works because `nn.Module` defines `__call__`, which runs `forward(X)` plus some PyTorch bookkeeping. Always call `head(X)`, never `head.forward(X)`.

---

**Q: "Isn't W_o initialized inside forward?"**

**A:** No, it goes in `__init__`. If `W_o` were created inside `forward`:
1. It would get **new random weights on every call**, so it could never learn.
2. **The optimizer can't see it.** `Adam(model.parameters())` collects the parameter list once, at setup.
3. **It isn't saved** by `state_dict()`.

**Rule:** anything with learnable weights (`nn.Linear`, `nn.LayerNorm`, `nn.Embedding`) is created in `__init__` and stored on `self`. `forward` only *uses* those layers. Operations with no weights (`torch.cat`, `F.softmax`, `@`) live in `forward`.

---

### Mistakes in My Second Draft of MHA

- ❌ `forward(self, X)` used `mask` without having it as a parameter, which raises `NameError`. Fix: `forward(self, X, mask=None)`.
- ❌ `Z @ self.W_o`: `W_o` is an `nn.Linear` **module**, not a matrix, so `@` fails. Call it instead: `self.W_o(Z)`.
  - Internally, `nn.Linear` stores `.weight` with shape `(out, in)` and computes `Z @ weight.T + bias`. This is also why you never do the `@` yourself with a Linear layer.

---

### Decision: Stay 2D (No Batch Dimension) for Now

**Decision:** Keep inputs as `(n, d_m)` while learning, and add batching later.
- `Q @ K.T` is correct for 2D tensors.
- ⚠️ **When batching is added:** on a 3D tensor `(batch, n, d_k)`, `.T` reverses *all* dimensions. Switch to `K.transpose(-2, -1)`, which swaps only the last two.
- The existing test functions (`test_attention`, `test_multi_head_attention`, etc.) use 3D inputs, so test with 2D inputs such as `torch.randn(10, 64)` for now.

---

## Phase 3: Positional & Rotary Embeddings (🚧 In Progress)

### Why Attention Needs Position Information
- Nothing in `softmax(Q @ K.T / √d_k) @ V` depends on a token's **index**. If you shuffle the rows of X, the rows of Z shuffle in exactly the same way, and nothing else changes (attention is *permutation-equivariant*).
- So "dog bites man" and "man bites dog" look the same to attention, apart from the order of the rows.
- Position therefore has to be injected explicitly.

### Two Approaches
| | Sinusoidal (2017) | RoPE (2021, Llama etc.) |
|---|---|---|
| What it does | **Adds** a fixed vector to each token's embedding | **Rotates** q and k vectors by an angle that depends on position |
| Where | Once, on X, before the first layer | Inside every attention head, after W_q/W_k, before Q @ K.T |
| Touches V? | Yes (it's part of X) | No, only Q and K |
| Position type | Absolute (relative only indirectly) | The score q_m · k_n depends only on **m − n** (relative) |

**Shared idea:** both split the dimensions into pairs (2i, 2i+1) and give each pair its own frequency θ_i = 1 / 10000^(2i/d).
- Low i → fast-spinning pair (fine-grained position, like a clock's second hand)
- High i → slow-spinning pair (coarse position, like an hour hand)

---

### Sinusoidal PE: Shapes and Ranges

**Q: "X is n × d_m. Is PE the same size?"**

**A:** Yes. It's added element-wise (`X + PE`), so it must match. Row `pos` of PE is the "position fingerprint" for the token at that position, and it is the same no matter which token sits there.

---

**Q: "Which of n and d_m set the ranges of pos and i?"**

**A:**
- **pos** indexes rows: `0 … n-1`, so **n** sets its range.
- **i** indexes column *pairs*: `0 … d_m/2 - 1`, so **d_m** sets its range. Column `2i` gets sin and column `2i+1` gets cos, at the same frequency.
- The exponent `2i/d_m` runs from 0 to about 1, so wavelengths range from 2π (fastest pair) to about 10000·2π (slowest pair).
- d_m must be even: add `assert d_m % 2 == 0`.

---

**Q: "What is the role of max_seq_len?"**

**A:** PE depends only on *position*, never on the data. So we compute the table **once in `__init__`** (build time), sized for the longest sequence we'll ever see: `(max_seq_len, d_m)`. Each `forward` call then **slices the first n rows** (run time): `X + pe[:n]`.
- This is the same build-vs-run split as with W_o.
- It also caps the model: sequences longer than max_seq_len won't fit.

---

### New Concept: `register_buffer`

Three ways to store a tensor on a module:

| | Learned (gets gradients)? | Moves with `.to(device)` / saved in `state_dict`? |
|---|---|---|
| `nn.Parameter` / `nn.Linear` | ✅ | ✅ |
| `self.register_buffer('pe', pe)` | ❌ | ✅ |
| `self.pe = pe` (plain attribute) | ❌ | ❌ (stays on CPU; you get a device-mismatch error on GPU) |

PE is fixed (not learned) but must travel with the model, so **buffer** is the right choice.

---

### RoPE vs Sinusoidal: Implementation Differences

**Q: "How would RoPE's implementation differ?"**

**A:** The precomputed table is almost the same. What differs is **where** it's applied and **how**.

| | Sinusoidal | RoPE |
|---|---|---|
| Table | `pos · θ_i`, shape `(max_seq_len, d_m/2)` | Same angle grid, but over **d_k** (head dim): `(max_seq_len, d_k/2)` |
| Store | one `pe` buffer | `cos` and `sin` buffers |
| Applied to | X, once, before the first layer | **Q and K** inside every head, after W_q/W_k, before `Q @ K.T` |
| Operation | add: `X + pe[:n]` | rotate each pair `(x_2i, x_2i+1)` at position m by angle `m·θ_i` |
| V | gets position info (via X) | untouched |

**The rotation (2D rotation applied to each pair):**
```
x'_2i   = x_2i · cos(mθ_i) − x_2i+1 · sin(mθ_i)
x'_2i+1 = x_2i · sin(mθ_i) + x_2i+1 · cos(mθ_i)
```

**Why it gives relative position:** rotating q by angle m and k by angle n gives
`(R_m q) · (R_n k) = q · R_(n−m) k`, so the score depends only on the **offset n − m**.
- Rotations preserve vector length, so only the *angle* between q and k changes, never their magnitude.

**Where it plugs in:**
```python
Q = self.W_q(X); K = self.W_k(X)
Q, K = rope(Q), rope(K)          # ← new step
scores = Q @ K.T / sqrt(d_k)
```

**Implementation note:** pairs can be *interleaved* `(0,1), (2,3), …` (the original paper) or *split in halves* `(i, i + d/2)` (the Hugging Face Llama `rotate_half`). Both work as long as you're consistent, but you can't load pretrained weights that use the other layout.

**Test:** `rope(q)` at position m dotted with `rope(k)` at position n should equal the same at positions m+5 and n+5. Also, `‖rope(q)‖ == ‖q‖`.

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

*Last updated: 2026-11-14*
*Current focus: Multi-Head Attention (loop version → vectorized version)*

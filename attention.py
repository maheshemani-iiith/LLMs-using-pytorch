"""
Attention Mechanisms from Scratch

Implements core attention components used in Transformers:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Complete Attention Block

Educational implementations with both manual and PyTorch versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# 1. SCALED DOT-PRODUCT ATTENTION (Manual Implementation)
# ============================================================================

class ManualScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V

    Why scaled? Without scaling by sqrt(d_k), dot products grow large with high dimensions,
    causing softmax to have tiny gradients (dying gradient problem).
    """

    """
    1) Q,K,V are obtained by transforming X nxd_m where n is the # of tokens and d_m is the size of the embeddings
        a) How do we obtain the weight metrices used to transform X into Q,K,V? 
        b) Are we starting off by assuming i already have the transformed matrices available to me?
    2) First step is Q @K.T resulting in a square matrix of size nxn
    3) d_k = ncol(Q) = ncol(Q) and the above matrix needs to be scaled by 1/sqrt(d_k)
    4) Does softmax(step 3 result) mean for a given a_ij/sum(a_ij)?
        a) What part of the setup so far causes the product to be asymmetrical?
        b) what information does this matrix capture?
    5) The above probability matrix is multiplied by V resulting in a transformed V
    """
    def __init__(self, d_k : int, d_v : int, d_m :int):
        super().__init__()
        """ n is the total # of tokens
             inputs: X:nxd_m, W_k,W_q,W_v
        """
        # n = nrows(X), d = ncols(X)
        # W_k : d x d_k, W_q : d x d_k, W_v : d x d_v
        self.d_k = d_k
        self.d_v = d_v
        self.d_m = d_m
        self.W_q = nn.Linear(d_m,d_k)
        self.W_k = nn.Linear(d_m,d_k)
        self.W_v = nn.Linear(d_m,d_v)
        
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        intermediate: Q(nxd_k): X @ W_q, K: X @ W_k, V: X @ W_v
        outputs: Z : nxd_v , we also save the softmax values for interpretability
        """
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        # scores 
        scores = (Q @ K.T) / math.sqrt(self.d_k)
        # during training and inference need to mask tokens
        # we expect a matrix with 0s and 1s where the former indicate the positions to be masked
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # softmaxed at row level 
        attention_wts = F.softmax(scores, dim = -1)
        # replace NaNs resulting from softmax of masked positions with near zero values
        attention_wts = torch.where(torch.isnan(attention_wts), torch.zeros_like(attention_wts), attention_wts)
        # Z which is the final scores
        Z = attention_wts @ V
        return Z , attention_wts    


# ============================================================================
# 2. MULTI-HEAD ATTENTION (Manual Implementation)
# ============================================================================

class ManualMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: Allows the model to attend to information from different
    representation subspaces at different positions.

    Why multiple heads? Different heads can learn to focus on different aspects:
    - One head might track grammatical relationships
    - Another might track semantic relationships
    - Another might track long-range dependencies
    """
    """
    1) We have h heads, each head can create its own Q,K,V matrices
        a) We initialize self attention h times and each head has its own set
        of weight matrices W_q, W_k, W_v
        b) Do we need to initialize with different seeds?
        c) how do i loop over h and save only the Z values from ManualScaledDotProductAttention?
        d) how do i pass the function name as param?
    2) Each head produces an output Z_i, which is concatenated to form Z 
        a) is there an equivalent of cbind?
    3) We need to have all the params we use in ManualScaledDotProductAttention as inputs
    """
    def __init__(self, d_k : int, d_v : int, d_m :int, h: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_m = d_m
        self.heads = nn.ModuleList([ManualScaledDotProductAttention(d_k, d_v, d_m) for _ in range(h)])
        self.W_o = nn.Linear(h*self.d_v, self.d_m)
        
    def forward(self,X: torch.Tensor, mask: None):
        # creating a concatenated Z from all the h attention heads
        Zs = [head(X, mask)[0] for head in self.heads]
        Z = torch.cat(Zs, dim=-1)
        # creating the downprojected matrix
        Z_d = self.W_o(Z)
        return Z_d


# ============================================================================
# 3. POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings using sine and cosine functions.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why? The model has no recurrence/convolution, so it needs explicit position info.
    Sine/cosine allow the model to easily learn relative positions.
    """

    def __init__(self, d_model: int = 512, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        


# ============================================================================
# 4. CAUSAL ATTENTION MASK
# ============================================================================

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal attention mask for autoregressive models.
    Prevents positions from attending to future positions.

    Returns lower triangular matrix of ones:
    [[1, 0, 0, ...],
     [1, 1, 0, ...],
     [1, 1, 1, ...],
     ...]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


# ============================================================================
# 5. COMPLETE ATTENTION BLOCK
# ============================================================================

class AttentionBlock(nn.Module):
    """
    Complete attention block with multi-head attention + feed-forward network.
    Uses residual connections and layer normalization (Transformer architecture).
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.attention = ManualMultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)

        Applies: Attention + Residual + Norm + FeedForward + Residual + Norm
        """
        # Multi-head attention with residual connection
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


# ============================================================================
# 6. VISUALIZATION & ANALYSIS
# ============================================================================

def visualize_attention_weights(attention_weights: torch.Tensor, title: str = "Attention Weights"):
    """
    Visualize attention weight matrix for one head.

    Args:
        attention_weights: (seq_len, seq_len)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights.detach().cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(title)
    plt.tight_layout()
    return plt


def visualize_positional_encoding(d_model: int = 512, max_seq_len: int = 100):
    """
    Visualize positional encoding patterns.
    Shows how sine/cosine waves encode position.
    """
    pe = PositionalEncoding(d_model, max_seq_len)
    encodings = pe.pe[0, :max_seq_len, :].detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(encodings.T, cmap='viridis', aspect='auto')
    plt.colorbar(label='Encoding Value')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Patterns')
    plt.tight_layout()
    return plt


# ============================================================================
# 7. TESTS & DEMONSTRATIONS
# ============================================================================

def test_attention():
    """Test scaled dot-product attention."""
    print("=" * 60)
    print("Testing Scaled Dot-Product Attention")
    print("=" * 60)

    batch_size, seq_len, d_model = 2, 10, 64
    attention = ManualScaledDotProductAttention(d_k=d_model)

    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    output, weights = attention(Q, K, V)

    print(f"Query shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum per position: {weights.sum(dim=-1)[0, :3]}")
    print(f"  (Should be ~1.0, confirming softmax normalization)")
    print()


def test_multi_head_attention():
    """Test multi-head attention."""
    print("=" * 60)
    print("Testing Multi-Head Attention")
    print("=" * 60)

    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    mha = ManualMultiHeadAttention(d_model, num_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    output, weights = mha(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"  (batch, num_heads, seq_len, seq_len)")
    print(f"Per-head weight distribution:")
    for head in range(min(3, num_heads)):
        print(f"  Head {head}: min={weights[0, head].min():.4f}, "
              f"max={weights[0, head].max():.4f}, "
              f"mean={weights[0, head].mean():.4f}")
    print()


def test_positional_encoding():
    """Test positional encoding."""
    print("=" * 60)
    print("Testing Positional Encoding")
    print("=" * 60)

    batch_size, seq_len, d_model = 2, 10, 512
    pe = PositionalEncoding(d_model)

    x = torch.randn(batch_size, seq_len, d_model)
    x_with_pe = pe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_with_pe.shape}")
    print(f"PE values range: [{pe.pe.min():.4f}, {pe.pe.max():.4f}]")
    print(f"  (Should be approximately [-1, 1] due to sin/cos)")
    print()


def test_causal_mask():
    """Test causal masking."""
    print("=" * 60)
    print("Testing Causal Attention Mask")
    print("=" * 60)

    seq_len = 5
    mask = create_causal_mask(seq_len, torch.device('cpu'))
    print(f"Causal mask shape: {mask.shape}")
    print(f"Mask pattern (lower triangular):")
    print(mask[0, 0].int())
    print()


def test_attention_block():
    """Test complete attention block."""
    print("=" * 60)
    print("Testing Complete Attention Block")
    print("=" * 60)

    batch_size, seq_len, d_model = 2, 10, 512
    block = AttentionBlock(d_model=d_model, num_heads=8, d_ff=2048)

    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shapes match: {x.shape == output.shape}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ATTENTION MECHANISMS FROM SCRATCH")
    print("=" * 60 + "\n")

    test_attention()
    test_multi_head_attention()
    test_positional_encoding()
    test_causal_mask()
    test_attention_block()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

"""
========================================================================
 ATTENTION MECHANISMS IN TRANSFORMERS: ARCHITECTURES & PRINCIPLES
========================================================================
This file provides educational implementations of 7 major attention variants.
Each class includes a comprehensive docstring with:
  - Working Principle
  - Use Case
  - Advantages
  - Disadvantages

Run: python attention_mechanisms.py
Requires: torch, numpy (for demo only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    WORKING PRINCIPLE:
      Computes attention weights via softmax(QK^T / sqrt(d_k)) and applies them to V.
      The scaling factor sqrt(d_k) prevents dot products from growing too large, 
      which would push softmax into regions with tiny gradients.

    USE CASE:
      Foundation of all transformer architectures; basic sequence-to-sequence modeling.

    ADVANTAGES:
      - Mathematically clean & fully differentiable
      - Captures global dependencies across the entire sequence
      - Easy to implement and debug

    DISADVANTAGES:
      - O(N^2) time & memory complexity (quadratic in sequence length)
      - Prone to memory overflow for long sequences (e.g., >8K tokens)
    """
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        # Q, K, V shape: (batch, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    WORKING PRINCIPLE:
      Splits Q, K, V into `num_heads` parallel subspaces, runs scaled dot-product 
      attention independently on each head, then concatenates and linearly projects 
      back to the original dimension. Enables the model to attend to information from 
      different representation subspaces simultaneously.

    USE CASE:
      Standard encoder/decoder transformer blocks; modern LLMs & vision transformers.

    ADVANTAGES:
      - Learns diverse relationships (syntax, semantics, positional, cross-modal)
      - Highly parallelizable on GPUs
      - Proven empirical performance across modalities

    DISADVANTAGES:
      - Multiplies parameter count & compute vs single-head
      - Still O(N^2) complexity; suffers on very long contexts without optimization
      - Head redundancy: some heads learn similar patterns
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(context)


class CrossAttention(nn.Module):
    """
    WORKING PRINCIPLE:
      Uses Queries from one sequence (e.g., text) and Keys/Values from another (e.g., image).
      Computes Attention(Q_text, K_image, V_image). Forces one modality to conditionally 
      attend to another.

    USE CASE:
      Multimodal fusion (image→text conditioning), encoder-decoder translation, diffusion models.

    ADVANTAGES:
      - Explicit directional conditioning (A attends to B)
      - Strong cross-modal alignment without merging token spaces early
      - Preserves pretrained language model weights when frozen

    DISADVANTAGES:
      - Requires two separate sequences; not self-contained
      - Can suffer from modality collapse if gradients aren't balanced
      - Slower inference due to dual-sequence materialization
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, query_seq: torch.Tensor, context_seq: torch.Tensor, mask: torch.Tensor = None):
        # Q from query_seq, K & V from context_seq
        return self.mha(query_seq, context_seq, context_seq, mask=mask)


class WindowedAttention(nn.Module):
    """
    WORKING PRINCIPLE:
      Partitions the sequence into non-overlapping windows of fixed size.
      Self-attention is computed only within each window. In alternating layers,
      windows are shifted to enable cross-window communication. Reduces complexity 
      from O(N^2) to O(N * window_size).

    USE CASE:
      Vision Transformers (Swin), long-document modeling, memory-constrained inference.

    ADVANTAGES:
      - Linear O(N) complexity & memory
      - Preserves local structure (critical for images/audio)
      - Enables hierarchical feature extraction

    DISADVANTAGES:
      - Limited receptive field per layer; requires many layers for global context
      - Window shifting adds implementation & masking complexity
      - Can miss long-range dependencies if window size is too small
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, x: torch.Tensor, shifted: bool = False):
        b, n, d = x.shape
        if n % self.window_size != 0:
            raise ValueError("seq_len must be divisible by window_size")
        windows = n // self.window_size
        
        x_win = x.view(b, windows, self.window_size, d)
        if shifted:
            x_win = torch.roll(x_win, shifts=self.window_size // 2, dims=1)
            
        # Apply MHA per window
        out = torch.stack([self.mha(w, w, w) for w in x_win], dim=1)
        out = out.view(b, n, d)
        
        if shifted:
            out = torch.roll(out, shifts=-self.window_size // 2, dims=1)
        return out


class LinearAttention(nn.Module):
    """
    WORKING PRINCIPLE:
      Replaces softmax with a kernel feature map φ(x) (e.g., ELU+1).
      Rewrites Attention(Q,K,V) = softmax(QK^T)V → φ(Q) [φ(K)^T V] / φ(Q) φ(K)^T 1
      Changes matrix multiplication order to avoid materializing the N×N attention matrix.

    USE CASE:
      Extremely long sequences, streaming/autoregressive generation, hybrid SSM-Transformer models.

    ADVANTAGES:
      - O(N) time & memory complexity
      - Theoretically unlimited sequence length
      - Highly cache-friendly for autoregressive decoding

    DISADVANTAGES:
      - Approximates softmax; loses sharp attention peaks
      - Can suffer from training instability without careful normalization
      - Less effective for tasks requiring precise token-to-token alignment
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.elu = nn.ELU(1.0)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        b, n, d = Q.shape
        Q = self.elu(self.W_Q(Q)).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
        K = self.elu(self.W_K(K)).view(b, n, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(b, n, self.num_heads, self.d_k).transpose(1, 2)

        # Linear attention core: (Q @ (K^T @ V)) instead of (Q @ K^T) @ V
        KV = torch.matmul(K.transpose(-2, -1), V)  # (b, h, d_k, d_k)
        out = torch.matmul(Q, KV)                  # (b, h, n, d_k)
        
        # Normalize by sum of keys
        z = torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6
        out = out / z
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return out


class GroupedQueryAttention(nn.Module):
    """
    WORKING PRINCIPLE:
      Generalizes Multi-Query Attention (MQA). Instead of `num_heads` independent K/V projections,
      only `num_kv_heads` are created and shared across groups of Q heads. 
      MQA is the extreme case (num_kv_heads = 1). GQA uses `num_q_heads // num_kv_heads` groups.

    USE CASE:
      Modern LLMs (Llama 3, Mistral, Qwen), inference-optimized architectures, KV cache reduction.

    ADVANTAGES:
      - Dramatically reduces KV cache memory (often 4-8x smaller)
      - Speeds up autoregressive generation significantly
      - Maintains most of MHA's representational capacity

    DISADVANTAGES:
      - Slight accuracy drop vs full MHA if group count is too low
      - Requires careful head grouping & initialization
      - Less flexible for fine-grained modality routing
    """
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.d_k = d_model // num_q_heads
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, self.d_k * num_kv_heads)
        self.W_V = nn.Linear(d_model, self.d_k * num_kv_heads)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        b, n, _ = Q.shape
        Q = self.W_Q(Q).view(b, n, self.num_q_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(b, n, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(b, n, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Repeat K/V to match Q heads
        repeats = self.num_q_heads // self.num_kv_heads
        K = K.repeat_interleave(repeats, dim=1)
        V = V.repeat_interleave(repeats, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(b, n, -1)
        return self.W_O(out)


class GatedAttention(nn.Module):
    """
    WORKING PRINCIPLE:
      Modulates the attention output with a learned gate: `output = Attention(Q,K,V) * σ(Wx + b)`.
      The gate dynamically scales how much cross-modal or cross-layer information flows into the 
      residual stream. Often initialized near zero to protect pretrained weights.

    USE CASE:
      Multimodal injection (Flamingo, LLaVA variants), adapter modules, catastrophic forgetting prevention.

    ADVANTAGES:
      - Fine-grained control over information flow
      - Enables stable fusion without retraining base model
      - Naturally implements "soft routing" between modalities

    DISADVANTAGES:
      - Adds parameters & potential bottleneck if gate saturates
      - Requires careful initialization (e.g., bias near 0)
      - Can under-attend if gate learns to suppress signals prematurely
    """
    def __init__(self, d_model: int, num_heads: int, gate_init_bias: float = -2.0):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        # Initialize gate to output small values initially
        with torch.no_grad():
            self.gate[0].bias.fill_(gate_init_bias)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
        attn_out = self.mha(Q, K, V, mask)
        gate_weight = self.gate(Q)  # Could also use K, V, or pooled context
        return attn_out * gate_weight


# ======================================================================
# DEMO / USAGE
# ======================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    batch, seq_len, d_model, num_heads = 2, 64, 128, 4

    # Dummy tensors
    x1 = torch.randn(batch, seq_len, d_model)
    x2 = torch.randn(batch, seq_len, d_model)
    mask = torch.ones(batch, 1, seq_len, seq_len)

    print("🔹 Running Attention Demos...")
    
    # 1. Scaled Dot-Product
    attn_base = ScaledDotProductAttention(d_k=d_model)
    out, _ = attn_base(x1, x1, x1, mask)
    print(f"[1] ScaledDotProduct: {out.shape}")

    # 2. Multi-Head
    mha = MultiHeadAttention(d_model, num_heads)
    out = mha(x1, x1, x1, mask)
    print(f"[2] MultiHead: {out.shape}")

    # 3. Cross-Attention
    cross = CrossAttention(d_model, num_heads)
    out = cross(x1, x2, mask)
    print(f"[3] Cross: {out.shape}")

    # 4. Windowed (window_size must divide seq_len)
    win_attn = WindowedAttention(d_model, num_heads, window_size=16)
    out = win_attn(x1, shifted=False)
    print(f"[4] Windowed: {out.shape}")

    # 5. Linear
    lin_attn = LinearAttention(d_model, num_heads)
    out = lin_attn(x1, x1, x1)
    print(f"[5] Linear: {out.shape}")

    # 6. GQA (MQA = num_kv_heads=1, GQA = num_kv_heads=2)
    gqa = GroupedQueryAttention(d_model, num_q_heads=8, num_kv_heads=2)
    out = gqa(x1, x1, x1, mask)
    print(f"[6] GQA (8Q/2KV): {out.shape}")

    # 7. Gated
    gated = GatedAttention(d_model, num_heads)
    out = gated(x1, x1, x1, mask)
    print(f"[7] Gated: {out.shape}")

    print("\n✅ All attention mechanisms executed successfully.")
    print("💡 Note: These are educational implementations. Production models use")
    print("   fused kernels (FlashAttention, xformers) for memory & speed optimization.")
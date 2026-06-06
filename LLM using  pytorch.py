"""
Full LLM from Scratch — PyTorch
Covers: BPE tokenizer, RMSNorm, RoPE, GQA, SwiGLU, KV cache,
        LoRA, DPO loss, speculative decoding, training loop
"""

import math
import struct
import regex as re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

@dataclass
class LLMConfig:
    vocab_size:     int   = 32000
    dim:            int   = 512
    n_layers:       int   = 8
    n_heads:        int   = 8
    n_kv_heads:     int   = 2          # GQA: fewer KV heads
    ffn_mult:       float = 2.667      # dim * ffn_mult → hidden size (SwiGLU)
    max_seq_len:    int   = 2048
    rope_theta:     float = 10000.0
    norm_eps:       float = 1e-5
    dropout:        float = 0.0
    tie_embeddings: bool  = True


# ─────────────────────────────────────────────
# 2. BPE TOKENIZER (minimal, trainable)
# ─────────────────────────────────────────────

class BPETokenizer:
    """Byte-level BPE tokenizer (GPT-2 style split pattern)."""

    PAT = re.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )

    def __init__(self):
        # byte-level base vocab (0-255)
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}
        self.encoder: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.special: dict[str, int] = {}

    # ── training ──────────────────────────────
    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size > 256
        n_merges = vocab_size - 256

        # pre-tokenise into byte sequences
        words = [list(w.encode("utf-8")) for w in self.PAT.findall(text)]

        def get_stats(words):
            counts = defaultdict(int)
            for w in words:
                for a, b in zip(w, w[1:]):
                    counts[(a, b)] += 1
            return counts

        def merge(words, pair, new_id):
            out = []
            for w in words:
                i, nw = 0, []
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i+1]) == pair:
                        nw.append(new_id); i += 2
                    else:
                        nw.append(w[i]); i += 1
                out.append(nw)
            return out

        for i in range(n_merges):
            stats = get_stats(words)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            words = merge(words, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and i % 100 == 0:
                print(f"  merge {i}/{n_merges}: {pair} → {new_id}")

        self.encoder = {v: k for k, v in self.vocab.items()}

    # ── encode / decode ───────────────────────
    def encode(self, text: str) -> list[int]:
        ids = []
        for word in self.PAT.findall(text):
            toks = list(word.encode("utf-8"))
            # apply merges in order
            while True:
                pairs = {(toks[i], toks[i+1]) for i in range(len(toks)-1)}
                best = min(
                    (p for p in pairs if p in self.merges),
                    key=lambda p: self.merges[p],
                    default=None
                )
                if best is None:
                    break
                new_id = self.merges[best]
                i, out = 0, []
                while i < len(toks):
                    if i < len(toks)-1 and (toks[i], toks[i+1]) == best:
                        out.append(new_id); i += 2
                    else:
                        out.append(toks[i]); i += 1
                toks = out
            ids.extend(toks)
        return ids

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def add_special(self, tokens: list[str]):
        for t in tokens:
            new_id = len(self.vocab)
            self.vocab[new_id] = t.encode("utf-8")
            self.encoder[t.encode("utf-8")] = new_id
            self.special[t] = new_id
        return self


# ─────────────────────────────────────────────
# 3. RMSNORM
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ─────────────────────────────────────────────
# 4. ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ─────────────────────────────────────────────

def precompute_freqs(dim: int, max_seq: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq)
    freqs = torch.outer(t, freqs)                       # (T, dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, T, H, head_dim)
    B, T, H, D = x.shape
    xc = torch.view_as_complex(x.float().reshape(B, T, H, D//2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(2)         # (1, T, 1, D/2)
    xr = torch.view_as_real(xc * freqs).reshape(B, T, H, D)
    return xr.type_as(x)


# ─────────────────────────────────────────────
# 5. GROUPED QUERY ATTENTION (GQA) + KV CACHE
# ─────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.dim // cfg.n_heads
        self.scale      = self.head_dim ** -0.5
        self.groups     = cfg.n_heads // cfg.n_kv_heads   # heads per KV group

        self.wq  = nn.Linear(cfg.dim, cfg.n_heads    * self.head_dim, bias=False)
        self.wk  = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv  = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo  = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim,    bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_pos: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, G, D = self.n_heads, self.groups, self.head_dim
        Hkv = self.n_kv_heads

        q = self.wq(x).view(B, T, H,   D)
        k = self.wk(x).view(B, T, Hkv, D)
        v = self.wv(x).view(B, T, Hkv, D)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # KV cache
        if use_cache:
            if self.cache_k is None:
                max_T = freqs.shape[0]
                self.cache_k = torch.zeros(B, max_T, Hkv, D, device=x.device, dtype=x.dtype)
                self.cache_v = torch.zeros(B, max_T, Hkv, D, device=x.device, dtype=x.dtype)
            self.cache_k[:, cache_pos:cache_pos+T] = k
            self.cache_v[:, cache_pos:cache_pos+T] = v
            k = self.cache_k[:, :cache_pos+T]
            v = self.cache_v[:, :cache_pos+T]

        # GQA: expand KV heads → Q heads
        k = k.repeat_interleave(G, dim=2)   # (B, S, H, D)
        v = v.repeat_interleave(G, dim=2)

        # (B, H, T, D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)                    # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.wo(out)

    def clear_cache(self):
        self.cache_k = self.cache_v = None


# ─────────────────────────────────────────────
# 6. SWIGLU FFN
# ─────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.ffn_mult)
        hidden = (hidden + 7) // 8 * 8      # multiple of 8
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)   # gate
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)   # down
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)   # up
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────
# 7. TRANSFORMER BLOCK
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn      = Attention(cfg)
        self.ffn       = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs, mask, use_cache, cache_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─────────────────────────────────────────────
# 8. FULL LLM
# ─────────────────────────────────────────────

class LLM(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg

        self.embed   = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.register_buffer(
            "freqs",
            precompute_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        mask = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        idx: torch.Tensor,              # (B, T)
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_pos: int = 0,
    ):
        B, T = idx.shape
        x = self.embed(idx)
        mask = self._causal_mask(T, idx.device) if not use_cache or T > 1 else None

        for layer in self.layers:
            x = layer(x, self.freqs, mask, use_cache, cache_pos)

        x = self.norm(x)
        logits = self.lm_head(x)           # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def clear_cache(self):
        for layer in self.layers:
            layer.attn.clear_cache()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# 9. LORA (LOW-RANK ADAPTATION)
# ─────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adapters."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank   = rank
        self.scale  = alpha / rank

        in_f, out_f = linear.in_features, linear.out_features
        self.A = nn.Parameter(torch.randn(rank, in_f)  * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))

        linear.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.A.T @ self.B.T) * self.scale


def apply_lora(model: LLM, rank: int = 8, alpha: float = 16.0) -> LLM:
    """Replace Q/V projections in every attention layer with LoRA versions."""
    for layer in model.layers:
        attn = layer.attn
        attn.wq = LoRALinear(attn.wq, rank, alpha)
        attn.wv = LoRALinear(attn.wv, rank, alpha)
    # freeze everything except LoRA params
    for name, p in model.named_parameters():
        if "lora" not in name.lower() and ".A" not in name and ".B" not in name:
            p.requires_grad_(False)
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = model.n_params()
    print(f"LoRA: {lora_params:,} trainable / {total_params:,} total "
          f"({100*lora_params/total_params:.2f}%)")
    return model


# ─────────────────────────────────────────────
# 10. DPO LOSS
# ─────────────────────────────────────────────

def dpo_loss(
    model: LLM,
    ref_model: LLM,
    chosen_ids:   torch.Tensor,   # (B, T)
    rejected_ids: torch.Tensor,   # (B, T)
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimisation loss.
    DPO paper: https://arxiv.org/abs/2305.18290
    """
    def log_probs(m, ids):
        logits, _ = m(ids[:, :-1])                     # (B, T-1, V)
        lp = F.log_softmax(logits, dim=-1)
        return lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)  # (B,)

    with torch.no_grad():
        ref_chosen   = log_probs(ref_model, chosen_ids)
        ref_rejected = log_probs(ref_model, rejected_ids)

    pi_chosen   = log_probs(model, chosen_ids)
    pi_rejected = log_probs(model, rejected_ids)

    log_ratio = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    return -F.logsigmoid(beta * log_ratio).mean()


# ─────────────────────────────────────────────
# 11. SPECULATIVE DECODING
# ─────────────────────────────────────────────

@torch.no_grad()
def speculative_decode(
    draft_model: LLM,
    target_model: LLM,
    prompt_ids: torch.Tensor,       # (1, T)
    max_new: int = 100,
    K: int = 4,                     # draft tokens per step
    temperature: float = 1.0,
    top_k: int = 50,
) -> list[int]:
    """
    Speculative decoding: draft model proposes K tokens,
    target model verifies in one forward pass.
    """
    device = prompt_ids.device
    generated = prompt_ids[0].tolist()

    def sample(logits, temp, k):
        logits = logits / max(temp, 1e-8)
        if k > 0:
            v, _ = logits.topk(k)
            logits[logits < v[-1]] = float("-inf")
        return torch.multinomial(F.softmax(logits, dim=-1), 1).item()

    while len(generated) - prompt_ids.shape[1] < max_new:
        ctx = torch.tensor([generated], device=device)

        # ── draft K tokens ───────────────────
        draft_tokens, draft_probs = [], []
        draft_ctx = ctx.clone()
        for _ in range(K):
            d_logits, _ = draft_model(draft_ctx)
            d_logits = d_logits[0, -1]
            d_probs  = F.softmax(d_logits / temperature, dim=-1)
            tok      = sample(d_logits.clone(), temperature, top_k)
            draft_tokens.append(tok)
            draft_probs.append(d_probs[tok].item())
            draft_ctx = torch.cat([draft_ctx, torch.tensor([[tok]], device=device)], dim=1)

        # ── verify with target ────────────────
        full_ctx = torch.cat([ctx, torch.tensor([draft_tokens], device=device)], dim=1)
        t_logits, _ = target_model(full_ctx)      # (1, T+K, V)

        accepted = []
        for i, tok in enumerate(draft_tokens):
            t_probs = F.softmax(t_logits[0, ctx.shape[1]+i-1] / temperature, dim=-1)
            accept_prob = min(1.0, (t_probs[tok] / (draft_probs[i] + 1e-8)).item())
            if torch.rand(1).item() < accept_prob:
                accepted.append(tok)
            else:
                # sample from corrected distribution
                corrected = (t_probs - draft_probs[i]).clamp(min=0)
                corrected /= corrected.sum()
                tok = torch.multinomial(corrected, 1).item()
                accepted.append(tok)
                break

        generated.extend(accepted)

        # if all K accepted, also sample one bonus token from target
        if len(accepted) == K:
            bonus_logits = t_logits[0, -1]
            bonus = sample(bonus_logits.clone(), temperature, top_k)
            generated.append(bonus)

    return generated[prompt_ids.shape[1]:]


# ─────────────────────────────────────────────
# 12. TRAINING LOOP
# ─────────────────────────────────────────────

class CosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr=0.0):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.total  = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_n = 0

    def step(self):
        self.step_n += 1
        n = self.step_n
        if n < self.warmup:
            lr = self.max_lr * n / self.warmup
        else:
            progress = (n - self.warmup) / max(1, self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def train(
    model: LLM,
    data: torch.Tensor,                 # flat token ids
    batch_size: int     = 8,
    seq_len: int        = 512,
    max_steps: int      = 1000,
    max_lr: float       = 3e-4,
    warmup_steps: int   = 100,
    grad_clip: float    = 1.0,
    device: str         = "cuda" if torch.cuda.is_available() else "cpu",
    log_every: int      = 50,
):
    model = model.to(device)
    data  = data.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1,
    )
    scheduler = CosineScheduler(optimizer, warmup_steps, max_steps, max_lr, max_lr * 0.1)

    model.train()
    for step in range(max_steps):
        # random batch
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[s:s+seq_len]   for s in starts])
        y = torch.stack([data[s+1:s+seq_len+1] for s in starts])

        # mixed precision
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device=="cuda"):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr = scheduler.step()

        if step % log_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

    return model


# ─────────────────────────────────────────────
# 13. GREEDY / SAMPLING INFERENCE
# ─────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: LLM,
    prompt_ids: list[int],
    max_new: int       = 200,
    temperature: float = 0.8,
    top_k: int         = 50,
    top_p: float       = 0.9,
    device: str        = "cpu",
) -> list[int]:
    model.eval()
    model.clear_cache()
    ids = torch.tensor([prompt_ids], device=device)
    generated = []

    # prefill
    _, _ = model(ids, use_cache=True, cache_pos=0)
    cache_pos = ids.shape[1]

    for _ in range(max_new):
        last = ids[:, -1:] if cache_pos == ids.shape[1] else torch.tensor([[generated[-1]]], device=device)
        logits, _ = model(last, use_cache=True, cache_pos=cache_pos)
        logits = logits[0, -1] / max(temperature, 1e-8)

        # top-k
        if top_k > 0:
            v, _ = logits.topk(top_k)
            logits[logits < v[-1]] = float("-inf")

        # top-p (nucleus)
        if top_p < 1.0:
            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_idx = probs.sort(descending=True)
            cum = sorted_p.cumsum(0)
            mask = cum - sorted_p > top_p
            sorted_p[mask] = 0.0
            probs = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_p)
            tok = torch.multinomial(probs, 1).item()
        else:
            tok = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

        generated.append(tok)
        cache_pos += 1

    return generated


# ─────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    cfg = LLMConfig(
        vocab_size=1024, dim=128, n_layers=2,
        n_heads=4, n_kv_heads=2, max_seq_len=256,
    )
    model = LLM(cfg)
    print(f"params: {model.n_params():,}")

    # forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, y)
    print(f"logits: {logits.shape}  loss: {loss.item():.4f}")

    # LoRA
    model_lora = apply_lora(model, rank=4, alpha=8.0)

    # quick training step
    data = torch.randint(0, cfg.vocab_size, (10_000,))
    model_lora = train(
        model_lora, data,
        batch_size=4, seq_len=64,
        max_steps=5, log_every=1,
        device=device,
    )

    # generation
    out = generate(model_lora, [1, 2, 3], max_new=20, device=device)
    print(f"generated: {out}")

    # BPE tokenizer
    tok = BPETokenizer()
    tok.train("hello world hello pytorch hello llm " * 200, vocab_size=300)
    enc = tok.encode("hello pytorch llm")
    dec = tok.decode(enc)
    print(f"BPE encode: {enc}")
    print(f"BPE decode: {dec!r}")

    print("All checks passed.")
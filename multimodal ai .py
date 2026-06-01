"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ADVANCED MULTIMODAL AI SYSTEM — PRODUCTION GRADE               ║
║  Supports: Text · Image · Audio · Video · PDF · Structured Data             ║
╚══════════════════════════════════════════════════════════════════════════════╝

ADVANTAGES OF MULTIMODAL AI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
✔ Processes multiple data types simultaneously (text, image, audio, video, docs)
✔ Richer context understanding — sees the full picture, not just one modality
✔ Reduces need for separate models/pipelines per input type
✔ Enables cross-modal reasoning (e.g., "describe what you hear in this image")
✔ Powers real-world applications: medical imaging, autonomous vehicles, VQA
✔ Improved accuracy by fusing signals across modalities
✔ Flexible — degrades gracefully when some modalities are missing

WORKFLOW:
━━━━━━━━
  [Raw Inputs]
       │
       ▼
  [Modality Detectors]  ←── Identify input type(s)
       │
       ▼
  [Modality Encoders]   ←── Text (BERT/GPT), Image (ViT/CLIP), Audio (Wav2Vec)
       │
       ▼
  [Fusion Layer]        ←── Cross-Attention / Concat / Weighted Sum
       │
       ▼
  [Multimodal Encoder]  ←── Unified representation
       │
       ▼
  [Task Head]           ←── Classification / Generation / VQA / Captioning
       │
       ▼
  [Output]              ←── Text, Labels, Scores, Generated Content
"""

# ─────────────────────────── IMPORTS ────────────────────────────────────────
from __future__ import annotations

import os
import io
import time
import base64
import logging
import asyncio
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# Optional heavy deps — gracefully skipped if not installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠  PyTorch not found. Fusion layers will run in simulation mode.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import anthropic          # Claude API for real multimodal calls
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ─────────────────────────── LOGGING ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MultimodalAI")


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS & DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

class Modality(str, Enum):
    TEXT    = "text"
    IMAGE   = "image"
    AUDIO   = "audio"
    VIDEO   = "video"
    PDF     = "pdf"
    TABULAR = "tabular"

class FusionStrategy(str, Enum):
    CONCAT          = "concat"           # Concatenate embeddings
    CROSS_ATTENTION = "cross_attention"  # Transformer cross-attention
    WEIGHTED_SUM    = "weighted_sum"     # Learnable weighted sum
    EARLY_FUSION    = "early_fusion"     # Fuse raw features
    LATE_FUSION     = "late_fusion"      # Fuse predictions

class TaskType(str, Enum):
    VQA              = "visual_question_answering"
    CAPTIONING       = "image_captioning"
    CLASSIFICATION   = "classification"
    GENERATION       = "text_generation"
    SUMMARIZATION    = "summarization"
    SENTIMENT        = "sentiment_analysis"
    OBJECT_DETECTION = "object_detection"
    CROSS_MODAL      = "cross_modal_retrieval"


@dataclass
class ModalityInput:
    """Container for a single modality's raw input."""
    modality: Modality
    data: Any                       # raw bytes, str, np.ndarray, PIL.Image, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None  # populated after encoding

    def __repr__(self) -> str:
        dtype = type(self.data).__name__
        return f"ModalityInput({self.modality.value}, dtype={dtype})"


@dataclass
class MultimodalRequest:
    """A full request carrying one or more modalities plus a task."""
    inputs: List[ModalityInput]
    task: TaskType
    query: str = ""
    fusion_strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    max_tokens: int = 512
    temperature: float = 0.7
    session_id: str = ""

    @property
    def modalities_present(self) -> List[Modality]:
        return [inp.modality for inp in self.inputs]


@dataclass
class MultimodalResponse:
    """Structured response from the multimodal pipeline."""
    output: str
    modalities_used: List[Modality]
    confidence: float
    latency_ms: float
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def pretty(self) -> str:
        mods = ", ".join(m.value for m in self.modalities_used)
        return (
            f"\n{'─'*60}\n"
            f"  OUTPUT        : {self.output}\n"
            f"  MODALITIES    : {mods}\n"
            f"  CONFIDENCE    : {self.confidence:.2%}\n"
            f"  LATENCY       : {self.latency_ms:.1f} ms\n"
            f"  TOKENS USED   : {self.token_count}\n"
            f"{'─'*60}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 2. MODALITY ENCODERS
# ════════════════════════════════════════════════════════════════════════════

class BaseEncoder:
    """Abstract encoder interface."""
    DIM = 768

    def encode(self, inp: ModalityInput) -> np.ndarray:
        raise NotImplementedError

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)


class TextEncoder(BaseEncoder):
    """
    Simulates a BERT/GPT-style text encoder.
    In production: replace with sentence-transformers or OpenAI embeddings.
    """
    DIM = 768

    def encode(self, inp: ModalityInput) -> np.ndarray:
        text = inp.data if isinstance(inp.data, str) else str(inp.data)
        # Deterministic pseudo-embedding based on character codes
        seed = sum(ord(c) for c in text[:256])
        rng  = np.random.default_rng(seed)
        vec  = rng.standard_normal(self.DIM).astype(np.float32)
        log.debug(f"TextEncoder: encoded {len(text)} chars → dim {self.DIM}")
        return self._normalize(vec)


class ImageEncoder(BaseEncoder):
    """
    Simulates a ViT / CLIP image encoder.
    In production: use `transformers.CLIPModel` or `timm`.
    """
    DIM = 768

    def encode(self, inp: ModalityInput) -> np.ndarray:
        if PIL_AVAILABLE and isinstance(inp.data, Image.Image):
            img_array = np.array(inp.data.resize((224, 224))).astype(np.float32)
            seed = int(img_array.mean() * 1000) % (2**31)
        elif isinstance(inp.data, (bytes, bytearray)):
            seed = sum(inp.data[:256]) % (2**31)
        else:
            seed = 42
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIM).astype(np.float32)
        log.debug(f"ImageEncoder: encoded image → dim {self.DIM}")
        return self._normalize(vec)


class AudioEncoder(BaseEncoder):
    """
    Simulates a Wav2Vec2 / Whisper audio encoder.
    In production: use `transformers.Wav2Vec2Model`.
    """
    DIM = 512

    def encode(self, inp: ModalityInput) -> np.ndarray:
        audio_data = inp.data
        if isinstance(audio_data, np.ndarray):
            seed = int(audio_data[:100].sum()) % (2**31)
        elif isinstance(audio_data, (bytes, bytearray)):
            seed = sum(audio_data[:256]) % (2**31)
        else:
            seed = 7
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIM).astype(np.float32)
        log.debug(f"AudioEncoder: encoded audio → dim {self.DIM}")
        return self._normalize(vec)


class VideoEncoder(BaseEncoder):
    """
    Simulates a VideoMAE / TimeSformer video encoder.
    In production: use `transformers.VideoMAEModel`.
    """
    DIM = 1024

    def encode(self, inp: ModalityInput) -> np.ndarray:
        seed = hash(str(inp.metadata)) % (2**31)
        rng  = np.random.default_rng(abs(seed))
        vec  = rng.standard_normal(self.DIM).astype(np.float32)
        log.debug(f"VideoEncoder: encoded video → dim {self.DIM}")
        return self._normalize(vec)


class PDFEncoder(BaseEncoder):
    """Encodes PDF documents via text extraction + TextEncoder."""
    DIM = 768

    def __init__(self):
        self._text_enc = TextEncoder()

    def encode(self, inp: ModalityInput) -> np.ndarray:
        # In production: use pdfplumber or PyMuPDF to extract text
        if isinstance(inp.data, (bytes, bytearray)):
            text = inp.data[:2000].decode("utf-8", errors="ignore")
        else:
            text = str(inp.data)[:2000]
        proxy = ModalityInput(modality=Modality.TEXT, data=text)
        return self._text_enc.encode(proxy)


class TabularEncoder(BaseEncoder):
    """Encodes structured/tabular data (CSV, JSON rows, DataFrames)."""
    DIM = 256

    def encode(self, inp: ModalityInput) -> np.ndarray:
        seed = hash(str(inp.data)[:512]) % (2**31)
        rng  = np.random.default_rng(abs(seed))
        vec  = rng.standard_normal(self.DIM).astype(np.float32)
        log.debug(f"TabularEncoder: encoded tabular data → dim {self.DIM}")
        return self._normalize(vec)


# ════════════════════════════════════════════════════════════════════════════
# 3. FUSION LAYER
# ════════════════════════════════════════════════════════════════════════════

class ModalityFusion:
    """
    Fuses embeddings from multiple modalities into a single unified vector.

    Strategies:
      • CONCAT          — simple concatenation (baseline)
      • WEIGHTED_SUM    — learnable weights per modality
      • CROSS_ATTENTION — simulates transformer cross-attention
      • EARLY/LATE      — conceptual stubs
    """
    MODALITY_WEIGHTS: Dict[Modality, float] = {
        Modality.TEXT:    1.0,
        Modality.IMAGE:   0.9,
        Modality.AUDIO:   0.7,
        Modality.VIDEO:   0.8,
        Modality.PDF:     0.85,
        Modality.TABULAR: 0.75,
    }

    def fuse(
        self,
        embeddings: Dict[Modality, np.ndarray],
        strategy: FusionStrategy,
    ) -> np.ndarray:
        if not embeddings:
            raise ValueError("No embeddings to fuse.")

        log.info(f"Fusing {len(embeddings)} modalities with strategy={strategy.value}")

        if strategy == FusionStrategy.CONCAT:
            return self._concat(embeddings)
        elif strategy == FusionStrategy.WEIGHTED_SUM:
            return self._weighted_sum(embeddings)
        elif strategy == FusionStrategy.CROSS_ATTENTION:
            return self._cross_attention(embeddings)
        else:
            # EARLY / LATE fusion — fallback to weighted sum
            return self._weighted_sum(embeddings)

    def _concat(self, embeddings: Dict[Modality, np.ndarray]) -> np.ndarray:
        vecs = list(embeddings.values())
        fused = np.concatenate(vecs, axis=-1)
        return fused / (np.linalg.norm(fused) + 1e-9)

    def _weighted_sum(self, embeddings: Dict[Modality, np.ndarray]) -> np.ndarray:
        target_dim = max(v.shape[-1] for v in embeddings.values())
        result = np.zeros(target_dim, dtype=np.float32)
        total_w = 0.0
        for mod, vec in embeddings.items():
            w = self.MODALITY_WEIGHTS.get(mod, 0.5)
            # Pad or truncate to target_dim
            if vec.shape[-1] < target_dim:
                vec = np.pad(vec, (0, target_dim - vec.shape[-1]))
            elif vec.shape[-1] > target_dim:
                vec = vec[:target_dim]
            result += w * vec
            total_w += w
        return (result / total_w) / (np.linalg.norm(result / total_w) + 1e-9)

    def _cross_attention(self, embeddings: Dict[Modality, np.ndarray]) -> np.ndarray:
        """
        Simulated cross-attention:
          Q = text embedding (or first available)
          K/V = all other embeddings
        """
        vecs = list(embeddings.values())
        keys = np.stack([v[:min(v.shape[0], vecs[0].shape[0])] for v in vecs])

        # Softmax attention scores
        q = vecs[0]
        scores = np.array([np.dot(q[:k.shape[0]], k) for k in keys])
        attn = np.exp(scores - scores.max())
        attn /= attn.sum()

        # Weighted combination
        dim = vecs[0].shape[0]
        result = np.zeros(dim, dtype=np.float32)
        for a, v in zip(attn, vecs):
            result += a * v[:dim]

        return result / (np.linalg.norm(result) + 1e-9)


# ════════════════════════════════════════════════════════════════════════════
# 4. TASK HEADS
# ════════════════════════════════════════════════════════════════════════════

class TaskHead:
    """Maps fused representation + task → final output string."""

    TASK_PROMPTS: Dict[TaskType, str] = {
        TaskType.VQA:              "Answer the following question about the visual content:",
        TaskType.CAPTIONING:       "Generate a detailed, descriptive caption for the provided image:",
        TaskType.CLASSIFICATION:   "Classify the content into the most relevant category:",
        TaskType.GENERATION:       "Generate creative content based on the provided context:",
        TaskType.SUMMARIZATION:    "Provide a concise, comprehensive summary of the content:",
        TaskType.SENTIMENT:        "Analyze the sentiment and emotional tone of the content:",
        TaskType.OBJECT_DETECTION: "Identify and describe all objects present in the visual content:",
        TaskType.CROSS_MODAL:      "Find and describe cross-modal relationships in the content:",
    }

    def build_prompt(self, request: MultimodalRequest) -> str:
        task_prefix = self.TASK_PROMPTS.get(request.task, "Process the following:")
        mods = ", ".join(m.value for m in request.modalities_present)
        prompt = (
            f"{task_prefix}\n\n"
            f"Input modalities: {mods}\n"
            f"User query: {request.query or 'No specific query provided.'}\n"
        )
        # Append text content directly if present
        for inp in request.inputs:
            if inp.modality == Modality.TEXT:
                prompt += f"\nText content:\n{inp.data}\n"
        return prompt


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN MULTIMODAL PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class MultimodalAIPipeline:
    """
    End-to-end multimodal AI pipeline:

      Input → Detect → Encode → Fuse → Task Head → LLM → Output

    Supports both real Claude API calls and simulation mode.
    """

    ENCODERS: Dict[Modality, type] = {
        Modality.TEXT:    TextEncoder,
        Modality.IMAGE:   ImageEncoder,
        Modality.AUDIO:   AudioEncoder,
        Modality.VIDEO:   VideoEncoder,
        Modality.PDF:     PDFEncoder,
        Modality.TABULAR: TabularEncoder,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        use_real_api: bool = True,
    ):
        self.model        = model
        self.use_real_api = use_real_api and ANTHROPIC_AVAILABLE
        self.fusion       = ModalityFusion()
        self.task_head    = TaskHead()
        self._encoders: Dict[Modality, BaseEncoder] = {}
        self._call_log: List[Dict] = []

        # Claude client
        if self.use_real_api:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self.client = anthropic.Anthropic(api_key=key) if key else None
            if not self.client:
                log.warning("No API key found — switching to simulation mode.")
                self.use_real_api = False

        log.info(
            f"MultimodalAIPipeline ready │ model={model} │ "
            f"api={'real' if self.use_real_api else 'simulated'}"
        )

    # ── Encoder registry ──────────────────────────────────────────────────

    def _get_encoder(self, modality: Modality) -> BaseEncoder:
        if modality not in self._encoders:
            enc_cls = self.ENCODERS.get(modality)
            if enc_cls is None:
                raise ValueError(f"No encoder registered for modality: {modality}")
            self._encoders[modality] = enc_cls()
        return self._encoders[modality]

    # ── Core pipeline ─────────────────────────────────────────────────────

    def process(self, request: MultimodalRequest) -> MultimodalResponse:
        """Synchronous pipeline entry point."""
        t0 = time.perf_counter()

        # Step 1 — Encode each modality
        log.info(f"Step 1/4 │ Encoding {len(request.inputs)} input(s)...")
        embeddings: Dict[Modality, np.ndarray] = {}
        for inp in request.inputs:
            enc = self._get_encoder(inp.modality)
            inp.embedding = enc.encode(inp)
            embeddings[inp.modality] = inp.embedding
            log.info(f"  ✔ {inp.modality.value:10s} → embedding shape: {inp.embedding.shape}")

        # Step 2 — Fuse embeddings
        log.info(f"Step 2/4 │ Fusing with strategy: {request.fusion_strategy.value}")
        if len(embeddings) > 1:
            fused = self.fusion.fuse(embeddings, request.fusion_strategy)
        else:
            fused = list(embeddings.values())[0]
        log.info(f"  ✔ Fused embedding shape: {fused.shape}")

        # Step 3 — Build prompt
        log.info(f"Step 3/4 │ Building prompt for task: {request.task.value}")
        prompt = self.task_head.build_prompt(request)

        # Step 4 — Generate output
        log.info(f"Step 4/4 │ Generating response...")
        if self.use_real_api and self.client:
            output, token_count = self._call_claude_api(request, prompt)
        else:
            output, token_count = self._simulate_output(request, fused, prompt)

        latency = (time.perf_counter() - t0) * 1000
        confidence = float(np.clip(np.abs(fused).mean() * 10, 0.6, 0.99))

        resp = MultimodalResponse(
            output=output,
            modalities_used=request.modalities_present,
            confidence=confidence,
            latency_ms=latency,
            token_count=token_count,
            metadata={"fusion_dim": fused.shape[0], "task": request.task.value},
        )
        self._call_log.append({"request": request.task.value, "latency_ms": latency})
        return resp

    async def process_async(self, request: MultimodalRequest) -> MultimodalResponse:
        """Async wrapper for concurrent processing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, request)

    # ── Claude API call ───────────────────────────────────────────────────

    def _call_claude_api(
        self, request: MultimodalRequest, text_prompt: str
    ) -> Tuple[str, int]:
        """Calls Claude API with multimodal content blocks."""
        content: List[Dict] = []

        for inp in request.inputs:
            if inp.modality == Modality.TEXT:
                content.append({"type": "text", "text": inp.data})

            elif inp.modality == Modality.IMAGE:
                if PIL_AVAILABLE and isinstance(inp.data, Image.Image):
                    buf = io.BytesIO()
                    inp.data.save(buf, format="JPEG")
                    b64 = base64.standard_b64encode(buf.getvalue()).decode()
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    })
                elif isinstance(inp.data, (bytes, bytearray)):
                    b64 = base64.standard_b64encode(inp.data).decode()
                    media = inp.metadata.get("media_type", "image/jpeg")
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media, "data": b64},
                    })

            elif inp.modality == Modality.PDF:
                if isinstance(inp.data, (bytes, bytearray)):
                    b64 = base64.standard_b64encode(inp.data).decode()
                    content.append({
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
                    })

        # Append the task prompt
        content.append({"type": "text", "text": text_prompt})

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            output = message.content[0].text
            tokens = message.usage.input_tokens + message.usage.output_tokens
            return output, tokens
        except Exception as e:
            log.error(f"Claude API error: {e}")
            return f"[API Error] {e}", 0

    # ── Simulation mode ───────────────────────────────────────────────────

    def _simulate_output(
        self,
        request: MultimodalRequest,
        fused: np.ndarray,
        prompt: str,
    ) -> Tuple[str, int]:
        """Rich simulated output for demo / offline use."""
        mods = ", ".join(m.value for m in request.modalities_present)
        sim_outputs = {
            TaskType.VQA: (
                f"Based on the visual and textual context ({mods}), the answer is: "
                f"The content shows clear evidence of the queried concept with high "
                f"confidence. The multimodal fusion captured complementary features "
                f"across {len(request.inputs)} input stream(s)."
            ),
            TaskType.CAPTIONING: (
                "A richly detailed scene captured across multiple sensory modalities. "
                "The visual composition reveals intricate spatial relationships, while "
                "accompanying textual cues provide semantic grounding for the depicted objects."
            ),
            TaskType.CLASSIFICATION: (
                f"Classification result: Category_A (confidence: 87.3%) | "
                f"Modalities used: {mods} | "
                f"The cross-modal features strongly align with the primary class."
            ),
            TaskType.GENERATION: (
                "Generated content synthesized from multimodal context: "
                "The fusion of textual semantics and visual features produces a coherent "
                "narrative that bridges both modalities seamlessly."
            ),
            TaskType.SUMMARIZATION: (
                f"Summary ({mods}): The provided content discusses key themes across "
                f"{len(request.inputs)} modality/ies. Core topics identified include "
                f"the central subject matter, supporting evidence, and contextual "
                f"background that collectively form a comprehensive understanding."
            ),
            TaskType.SENTIMENT: (
                "Sentiment: POSITIVE (score: 0.82) │ "
                "Emotion: Confident, Informative │ "
                "The multimodal signals agree on a generally positive and constructive tone."
            ),
            TaskType.OBJECT_DETECTION: (
                "Detected entities: [Object_1 @ (0.12, 0.34, 0.56, 0.78), conf=0.94] "
                "[Object_2 @ (0.45, 0.12, 0.89, 0.67), conf=0.87] "
                "[Object_3 @ (0.03, 0.51, 0.42, 0.99), conf=0.79]"
            ),
            TaskType.CROSS_MODAL: (
                f"Cross-modal alignment score: 0.91 │ "
                f"The {mods} inputs share strong semantic overlap. "
                f"Key bridge concepts: context, content, meaning, structure."
            ),
        }
        output = sim_outputs.get(
            request.task,
            f"[Simulation] Processed {mods} with task={request.task.value}."
        )
        tokens = len(output.split())
        return output, tokens

    # ── Batch processing ──────────────────────────────────────────────────

    def process_batch(
        self, requests: List[MultimodalRequest], parallel: bool = False
    ) -> List[MultimodalResponse]:
        """Process multiple requests, optionally in parallel via asyncio."""
        if parallel:
            async def _run_all():
                tasks = [self.process_async(r) for r in requests]
                return await asyncio.gather(*tasks)
            return asyncio.run(_run_all())
        return [self.process(r) for r in requests]

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def auto_detect_modality(data: Any, hint: str = "") -> Modality:
        """Automatically detect the modality of raw data."""
        hint_lower = hint.lower()
        if any(ext in hint_lower for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
            return Modality.IMAGE
        if any(ext in hint_lower for ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]):
            return Modality.AUDIO
        if any(ext in hint_lower for ext in [".mp4", ".avi", ".mov", ".mkv"]):
            return Modality.VIDEO
        if ".pdf" in hint_lower:
            return Modality.PDF
        if any(ext in hint_lower for ext in [".csv", ".tsv", ".xlsx", ".json"]):
            return Modality.TABULAR
        if PIL_AVAILABLE and isinstance(data, Image.Image):
            return Modality.IMAGE
        if isinstance(data, str):
            return Modality.TEXT
        if isinstance(data, np.ndarray) and data.ndim in (2, 3):
            return Modality.IMAGE
        return Modality.TEXT

    def stats(self) -> Dict:
        """Return call statistics."""
        if not self._call_log:
            return {"calls": 0}
        latencies = [c["latency_ms"] for c in self._call_log]
        return {
            "total_calls": len(self._call_log),
            "avg_latency_ms": round(np.mean(latencies), 2),
            "min_latency_ms": round(np.min(latencies), 2),
            "max_latency_ms": round(np.max(latencies), 2),
        }


# ════════════════════════════════════════════════════════════════════════════
# 6. CONVENIENCE BUILDER
# ════════════════════════════════════════════════════════════════════════════

class MultimodalRequestBuilder:
    """Fluent builder for constructing MultimodalRequest objects."""

    def __init__(self):
        self._inputs: List[ModalityInput] = []
        self._task = TaskType.GENERATION
        self._query = ""
        self._strategy = FusionStrategy.CROSS_ATTENTION
        self._max_tokens = 512
        self._temperature = 0.7

    def add_text(self, text: str, **meta) -> "MultimodalRequestBuilder":
        self._inputs.append(ModalityInput(Modality.TEXT, text, meta))
        return self

    def add_image(self, image: Any, **meta) -> "MultimodalRequestBuilder":
        self._inputs.append(ModalityInput(Modality.IMAGE, image, meta))
        return self

    def add_audio(self, audio: Any, **meta) -> "MultimodalRequestBuilder":
        self._inputs.append(ModalityInput(Modality.AUDIO, audio, meta))
        return self

    def add_pdf(self, pdf_bytes: bytes, **meta) -> "MultimodalRequestBuilder":
        self._inputs.append(ModalityInput(Modality.PDF, pdf_bytes, meta))
        return self

    def add_tabular(self, data: Any, **meta) -> "MultimodalRequestBuilder":
        self._inputs.append(ModalityInput(Modality.TABULAR, data, meta))
        return self

    def task(self, t: TaskType) -> "MultimodalRequestBuilder":
        self._task = t; return self

    def query(self, q: str) -> "MultimodalRequestBuilder":
        self._query = q; return self

    def fusion(self, s: FusionStrategy) -> "MultimodalRequestBuilder":
        self._strategy = s; return self

    def max_tokens(self, n: int) -> "MultimodalRequestBuilder":
        self._max_tokens = n; return self

    def build(self) -> MultimodalRequest:
        if not self._inputs:
            raise ValueError("At least one input modality is required.")
        return MultimodalRequest(
            inputs=self._inputs,
            task=self._task,
            query=self._query,
            fusion_strategy=self._strategy,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )


# ════════════════════════════════════════════════════════════════════════════
# 7. DEMO — RUN THIS FILE DIRECTLY
# ════════════════════════════════════════════════════════════════════════════

def run_demo():
    print("\n" + "═" * 65)
    print("  ADVANCED MULTIMODAL AI — DEMO RUN")
    print("═" * 65)

    pipeline = MultimodalAIPipeline(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model="claude-opus-4-5",
        use_real_api=True,      # Falls back to simulation if no key found
    )

    # ── Demo 1: Text + Image (VQA) ────────────────────────────────────────
    print("\n📌 Demo 1: Visual Question Answering (Text + Image)")
    req1 = (
        MultimodalRequestBuilder()
        .add_text("A photo of a mountain landscape at sunset.")
        .add_image(b"\xff\xd8\xff" + b"\x00" * 100, media_type="image/jpeg")
        .task(TaskType.VQA)
        .query("What colors dominate the image and what emotions do they evoke?")
        .fusion(FusionStrategy.CROSS_ATTENTION)
        .build()
    )
    resp1 = pipeline.process(req1)
    print(resp1.pretty())

    # ── Demo 2: Text + Audio (Sentiment) ─────────────────────────────────
    print("\n📌 Demo 2: Multimodal Sentiment Analysis (Text + Audio)")
    audio_signal = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
    req2 = (
        MultimodalRequestBuilder()
        .add_text("The product exceeded all my expectations! Absolutely fantastic.")
        .add_audio(audio_signal, sample_rate=16000)
        .task(TaskType.SENTIMENT)
        .query("Analyze the combined sentiment from voice tone and text.")
        .fusion(FusionStrategy.WEIGHTED_SUM)
        .build()
    )
    resp2 = pipeline.process(req2)
    print(resp2.pretty())

    # ── Demo 3: PDF + Text (Summarization) ───────────────────────────────
    print("\n📌 Demo 3: Document + Text Summarization (PDF + Text)")
    fake_pdf = b"%PDF-1.4\nThis is a research paper about climate change impacts."
    req3 = (
        MultimodalRequestBuilder()
        .add_pdf(fake_pdf)
        .add_text("Focus on economic consequences.")
        .task(TaskType.SUMMARIZATION)
        .query("Summarize key findings with emphasis on economic impact.")
        .fusion(FusionStrategy.CONCAT)
        .build()
    )
    resp3 = pipeline.process(req3)
    print(resp3.pretty())

    # ── Demo 4: Batch processing ──────────────────────────────────────────
    print("\n📌 Demo 4: Batch Processing (3 requests)")
    reqs = [
        MultimodalRequestBuilder()
        .add_text(f"Sample text input number {i}.")
        .task(TaskType.CLASSIFICATION)
        .query(f"Classify input {i}")
        .build()
        for i in range(3)
    ]
    responses = pipeline.process_batch(reqs)
    for i, r in enumerate(responses):
        print(f"  Batch [{i+1}] → {r.output[:80]}... | {r.latency_ms:.1f}ms")

    # ── Stats ─────────────────────────────────────────────────────────────
    print("\n📊 Pipeline Statistics:")
    stats = pipeline.stats()
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    run_demo()
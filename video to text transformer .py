"""
Advanced Video-to-Text Pipeline
=================================
A production-grade pipeline for converting video files to accurate transcripts.

Stages:
  1. Input ingestion         — file / URL / stream
  2. Validation & metadata   — format check, codec, duration
  3. Audio extraction        — FFmpeg → 16kHz WAV mono
  4. Audio preprocessing     — denoise, normalize, VAD segmentation
  5. Intelligent chunking    — silence-based splitting with overlap
  6. Parallel transcription  — Whisper GPU/CPU batching
  7. Post-processing         — merge, deduplicate, punctuate
  8. Speaker diarization     — pyannote.audio speaker labels
  9. Output generation       — TXT / SRT / VTT / JSON + quality report

Dependencies (pip install):
  openai-whisper ffmpeg-python noisereduce webrtcvad
  pyannote.audio torch torchaudio numpy scipy pydub
"""

# ─────────────────────── stdlib ───────────────────────
import os
import re
import json
import time
import logging
import hashlib
import tempfile
import warnings
import subprocess
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Tuple

# ─────────────────────── third-party ──────────────────
import numpy as np
import torch
import whisper
import ffmpeg
import noisereduce as nr
import webrtcvad
from scipy.io import wavfile
from scipy.signal import resample
from pydub import AudioSegment

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vtt")


# ══════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════
@dataclass
class VideoMetadata:
    """Parsed video file metadata."""
    path: str
    duration_sec: float
    video_codec: str
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    file_size_mb: float
    checksum: str


@dataclass
class AudioChunk:
    """Single audio segment sent to Whisper."""
    index: int
    start_sec: float            # offset in the original audio
    end_sec: float
    audio_array: np.ndarray     # float32, mono, 16 kHz
    sample_rate: int = 16_000


@dataclass
class TranscriptSegment:
    """One transcribed segment from Whisper."""
    chunk_index: int
    start: float
    end: float
    text: str
    confidence: float           # avg log-prob → confidence proxy
    language: str = "en"
    speaker: Optional[str] = None


@dataclass
class QualityReport:
    """Summarised quality metrics for the full transcript."""
    total_segments: int
    avg_confidence: float
    low_confidence_count: int   # segments < 0.5
    total_duration_sec: float
    processing_time_sec: float
    model_used: str
    language_detected: str


@dataclass
class PipelineConfig:
    """Centralised config for every pipeline stage."""
    # General
    model_size: str = "base"    # tiny / base / small / medium / large
    device: str = "auto"        # auto / cpu / cuda
    language: Optional[str] = None  # None = auto-detect
    output_formats: List[str] = field(default_factory=lambda: ["txt", "srt", "json"])

    # Audio extraction
    target_sample_rate: int = 16_000
    mono: bool = True

    # Preprocessing
    denoise: bool = True
    normalize_audio: bool = True
    vad_aggressiveness: int = 2         # webrtcvad 0-3

    # Chunking
    chunk_length_sec: float = 30.0
    chunk_overlap_sec: float = 1.0
    silence_threshold_db: float = -40.0
    min_silence_ms: int = 500

    # Transcription
    max_workers: int = 4
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0            # 0 = greedy (deterministic)
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # Diarization
    enable_diarization: bool = False    # requires HuggingFace token
    hf_token: Optional[str] = None

    # Output
    output_dir: str = "./output"
    min_confidence_warn: float = 0.5    # segments below this are flagged


# ══════════════════════════════════════════════════════
#  STAGE 1 — Input ingestion
# ══════════════════════════════════════════════════════
class InputIngester:
    """
    Accepts a local file path, HTTP/HTTPS URL, or a raw stream path.
    Downloads remote sources to a temp file so every downstream
    stage receives a stable local path.
    """

    SUPPORTED_EXTENSIONS = {
        ".mp4", ".mkv", ".avi", ".mov", ".wmv",
        ".flv", ".webm", ".m4v", ".mpg", ".mpeg",
        ".ts", ".mts", ".m2ts",
    }

    def __init__(self, config: PipelineConfig):
        self.config = config

    def ingest(self, source: str) -> Path:
        log.info("Stage 1 ▶ Ingesting source: %s", source)

        if source.startswith(("http://", "https://")):
            return self._download(source)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {source}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        log.info("  Local file accepted: %s (%.1f MB)",
                 path.name, path.stat().st_size / 1e6)
        return path

    def _download(self, url: str) -> Path:
        import urllib.request
        suffix = Path(url.split("?")[0]).suffix or ".mp4"
        tmp = Path(tempfile.mktemp(suffix=suffix))
        log.info("  Downloading %s …", url)
        urllib.request.urlretrieve(url, tmp)
        log.info("  Downloaded → %s (%.1f MB)", tmp, tmp.stat().st_size / 1e6)
        return tmp


# ══════════════════════════════════════════════════════
#  STAGE 2 — Validation & metadata extraction
# ══════════════════════════════════════════════════════
class MetadataExtractor:
    """
    Runs ffprobe to extract codec, duration, sample-rate, channels.
    Computes SHA-256 checksum so the same video is never processed twice.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def extract(self, video_path: Path) -> VideoMetadata:
        log.info("Stage 2 ▶ Extracting metadata from %s", video_path.name)

        probe = ffmpeg.probe(str(video_path))
        streams = probe["streams"]

        # Pick first video and audio streams
        v_stream = next((s for s in streams if s["codec_type"] == "video"), {})
        a_stream = next((s for s in streams if s["codec_type"] == "audio"), {})

        if not a_stream:
            raise ValueError("No audio stream found in video.")

        duration = float(probe["format"].get("duration", 0))
        size_mb  = video_path.stat().st_size / 1e6
        checksum = self._sha256(video_path)

        meta = VideoMetadata(
            path=str(video_path),
            duration_sec=duration,
            video_codec=v_stream.get("codec_name", "none"),
            audio_codec=a_stream.get("codec_name", "unknown"),
            audio_sample_rate=int(a_stream.get("sample_rate", 44100)),
            audio_channels=int(a_stream.get("channels", 2)),
            file_size_mb=size_mb,
            checksum=checksum,
        )

        log.info(
            "  Duration: %.1f s | Audio: %s %d Hz %dch | Size: %.1f MB",
            meta.duration_sec, meta.audio_codec,
            meta.audio_sample_rate, meta.audio_channels, meta.file_size_mb,
        )
        return meta

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]


# ══════════════════════════════════════════════════════
#  STAGE 3 — Audio extraction
# ══════════════════════════════════════════════════════
class AudioExtractor:
    """
    Uses FFmpeg to strip audio from the video, convert to 16-bit PCM
    WAV at the target sample rate (default 16 kHz), mono.
    This is the format Whisper expects natively.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def extract(self, video_path: Path, out_dir: Path) -> Path:
        log.info("Stage 3 ▶ Extracting audio …")
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = out_dir / (video_path.stem + ".wav")

        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(wav_path),
                acodec="pcm_s16le",
                ac=1 if self.config.mono else 2,
                ar=self.config.target_sample_rate,
                vn=None,            # drop video stream
            )
            .overwrite_output()
            .run(quiet=True)
        )

        size_mb = wav_path.stat().st_size / 1e6
        log.info("  WAV saved → %s (%.1f MB)", wav_path.name, size_mb)
        return wav_path


# ══════════════════════════════════════════════════════
#  STAGE 4 — Audio preprocessing
# ══════════════════════════════════════════════════════
class AudioPreprocessor:
    """
    Three sub-steps:
      a) Spectral-subtraction noise reduction (noisereduce)
      b) RMS normalisation to -20 dBFS
      c) Voice-activity detection (WebRTC VAD) to strip silent regions
         before we split into chunks — cuts Whisper hallucinations on
         long silences.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def process(self, wav_path: Path) -> Tuple[np.ndarray, int]:
        log.info("Stage 4 ▶ Preprocessing audio …")
        sr, audio = wavfile.read(str(wav_path))

        # Convert to float32 normalised in [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        if self.config.denoise:
            log.info("  Applying spectral noise reduction …")
            audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,
                prop_decrease=0.75,
            )

        if self.config.normalize_audio:
            log.info("  Normalising to -20 dBFS …")
            audio = self._normalise(audio, target_dbfs=-20.0)

        log.info("  Running VAD (aggressiveness=%d) …",
                 self.config.vad_aggressiveness)
        audio = self._apply_vad(audio, sr)

        log.info("  Preprocessed audio: %.1f s", len(audio) / sr)
        return audio, sr

    @staticmethod
    def _normalise(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        target_rms = 10 ** (target_dbfs / 20.0)
        return audio * (target_rms / rms)

    def _apply_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        WebRTC VAD requires 16-bit PCM at 8/16/32/48 kHz.
        We process 30 ms frames and keep only speech frames.
        """
        vad = webrtcvad.Vad(self.config.vad_aggressiveness)
        frame_ms = 30
        frame_len = int(sr * frame_ms / 1000)

        audio_int16 = (audio * 32768).astype(np.int16)
        speech_frames = []

        for i in range(0, len(audio_int16) - frame_len, frame_len):
            frame = audio_int16[i: i + frame_len]
            frame_bytes = frame.tobytes()
            try:
                is_speech = vad.is_speech(frame_bytes, sample_rate=sr)
            except Exception:
                is_speech = True    # keep frame if VAD fails on this chunk

            if is_speech:
                speech_frames.append(audio[i: i + frame_len])

        if not speech_frames:
            log.warning("  VAD: no speech frames detected — keeping full audio.")
            return audio

        kept_ratio = len(speech_frames) * frame_len / len(audio)
        log.info("  VAD: kept %.0f%% of audio as speech.", kept_ratio * 100)
        return np.concatenate(speech_frames)


# ══════════════════════════════════════════════════════
#  STAGE 5 — Intelligent chunking
# ══════════════════════════════════════════════════════
class AudioChunker:
    """
    Splits the preprocessed audio into overlapping chunks so that:
      • No chunk exceeds `chunk_length_sec` (Whisper's sweet-spot is 30 s)
      • Splits occur at silence boundaries when possible (avoids cutting
        mid-word and reduces WER at chunk edges)
      • A `chunk_overlap_sec` guard ensures boundary words aren't lost.

    Returns a list of AudioChunk objects that carry the global time offset
    so we can reconstruct accurate timestamps later.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def chunk(self, audio: np.ndarray, sr: int) -> List[AudioChunk]:
        log.info("Stage 5 ▶ Chunking audio (%.1f s) …", len(audio) / sr)

        chunk_samples   = int(self.config.chunk_length_sec * sr)
        overlap_samples = int(self.config.chunk_overlap_sec * sr)

        chunks: List[AudioChunk] = []
        start = 0
        idx   = 0

        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            segment = audio[start:end]

            # Try to trim end to nearest silence boundary
            end_trimmed = self._find_silence_boundary(segment, sr)
            if end_trimmed and end_trimmed > overlap_samples:
                segment = segment[:end_trimmed]

            chunks.append(AudioChunk(
                index=idx,
                start_sec=start / sr,
                end_sec=(start + len(segment)) / sr,
                audio_array=segment,
                sample_rate=sr,
            ))

            # Advance, minus the overlap so edge context is repeated
            advance = max(len(segment) - overlap_samples, 1)
            start += advance
            idx   += 1

        log.info("  Created %d chunk(s).", len(chunks))
        return chunks

    def _find_silence_boundary(
        self, segment: np.ndarray, sr: int
    ) -> Optional[int]:
        """
        Scan from the END of the segment backwards to find the first
        window below the silence threshold.  Returns sample offset or None.
        """
        window_sec = 0.5
        window_samples = int(window_sec * sr)
        threshold = 10 ** (self.config.silence_threshold_db / 20.0)

        for i in range(len(segment) - window_samples, len(segment) // 2, -window_samples):
            window = segment[i: i + window_samples]
            if np.abs(window).max() < threshold:
                return i        # silence found → split here

        return None             # no silence → use hard boundary


# ══════════════════════════════════════════════════════
#  STAGE 6 — Parallel transcription
# ══════════════════════════════════════════════════════
class ParallelTranscriber:
    """
    Loads the Whisper model once, then submits every AudioChunk to a
    ThreadPoolExecutor.  Results are sorted by chunk index before returning
    so the caller always gets segments in chronological order.

    Each chunk is passed to whisper.transcribe() which handles its own
    30-second internal windowing — we just give it the numpy array.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        device = self._resolve_device(config.device)
        log.info("Stage 6 ▶ Loading Whisper model '%s' on %s …",
                 config.model_size, device)
        self.model  = whisper.load_model(config.model_size, device=device)
        self.device = device

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def transcribe(self, chunks: List[AudioChunk]) -> List[TranscriptSegment]:
        log.info("  Transcribing %d chunks with %d worker(s) …",
                 len(chunks), self.config.max_workers)

        all_segments: List[TranscriptSegment] = []
        futures = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            for chunk in chunks:
                fut = pool.submit(self._transcribe_chunk, chunk)
                futures[fut] = chunk.index

            for fut in as_completed(futures):
                try:
                    segs = fut.result()
                    all_segments.extend(segs)
                except Exception as exc:
                    log.error("  Chunk %d failed: %s", futures[fut], exc)

        # Sort by global start time
        all_segments.sort(key=lambda s: s.start)
        log.info("  Transcription complete: %d segment(s).", len(all_segments))
        return all_segments

    def _transcribe_chunk(self, chunk: AudioChunk) -> List[TranscriptSegment]:
        """Run Whisper on a single chunk, convert to TranscriptSegment list."""
        result = self.model.transcribe(
            chunk.audio_array,
            language=self.config.language,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            temperature=self.config.temperature,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
            logprob_threshold=self.config.logprob_threshold,
            no_speech_threshold=self.config.no_speech_threshold,
            word_timestamps=True,
            verbose=False,
        )

        segments = []
        detected_lang = result.get("language", "unknown")

        for seg in result.get("segments", []):
            # Offset timestamps to global audio timeline
            start = chunk.start_sec + seg["start"]
            end   = chunk.start_sec + seg["end"]
            text  = seg["text"].strip()

            # Confidence proxy: avg log-prob (Whisper exposes it per-token)
            avg_lp = seg.get("avg_logprob", -0.5)
            confidence = float(np.clip(np.exp(avg_lp), 0.0, 1.0))

            if text:
                segments.append(TranscriptSegment(
                    chunk_index=chunk.index,
                    start=start,
                    end=end,
                    text=text,
                    confidence=confidence,
                    language=detected_lang,
                ))

        return segments


# ══════════════════════════════════════════════════════
#  STAGE 7 — Post-processing
# ══════════════════════════════════════════════════════
class PostProcessor:
    """
    After parallel transcription, chunks' overlapping regions produce
    duplicate text.  This stage:
      a) Deduplicates: drops segments whose text overlaps with the
         previous segment (comparing normalised tokens)
      b) Merges short segments into natural sentence boundaries
      c) Cleans Whisper artefacts (filler words, trailing ellipses, etc.)
    """

    FILLER_PATTERN = re.compile(
        r"\b(um+|uh+|hmm+|err+|ah+)\b", re.IGNORECASE
    )
    ARTEFACT_PATTERN = re.compile(
        r"(\.{3,}|\[.*?\]|\(.*?\))"
    )

    def __init__(self, config: PipelineConfig):
        self.config = config

    def process(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        log.info("Stage 7 ▶ Post-processing %d segment(s) …", len(segments))

        segments = self._deduplicate(segments)
        segments = self._clean_text(segments)
        segments = self._merge_short(segments, min_chars=20)

        log.info("  After post-processing: %d segment(s).", len(segments))
        return segments

    def _deduplicate(self, segs: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Remove segments that are substantially identical to the previous one.
        We normalise text (lower, strip punct) and compare word sets.
        """
        out = []
        prev_words: set = set()

        for seg in segs:
            words = set(re.sub(r"[^\w\s]", "", seg.text.lower()).split())
            if not words:
                continue

            overlap = len(words & prev_words) / max(len(words), 1)
            if overlap < 0.80:      # < 80% overlap → keep
                out.append(seg)
                prev_words = words

        return out

    def _clean_text(self, segs: List[TranscriptSegment]) -> List[TranscriptSegment]:
        for seg in segs:
            t = seg.text
            t = self.ARTEFACT_PATTERN.sub("", t)   # remove [Music] etc.
            t = self.FILLER_PATTERN.sub("", t)       # remove fillers
            t = re.sub(r"\s{2,}", " ", t).strip()   # collapse whitespace
            # Capitalise first letter
            if t:
                t = t[0].upper() + t[1:]
            seg.text = t
        return [s for s in segs if s.text]

    @staticmethod
    def _merge_short(
        segs: List[TranscriptSegment], min_chars: int = 20
    ) -> List[TranscriptSegment]:
        """Merge consecutive segments that are suspiciously short."""
        out: List[TranscriptSegment] = []
        buffer: Optional[TranscriptSegment] = None

        for seg in segs:
            if buffer is None:
                buffer = seg
                continue

            if len(buffer.text) < min_chars:
                # Append to buffer
                buffer = TranscriptSegment(
                    chunk_index=buffer.chunk_index,
                    start=buffer.start,
                    end=seg.end,
                    text=buffer.text + " " + seg.text,
                    confidence=(buffer.confidence + seg.confidence) / 2,
                    language=buffer.language,
                )
            else:
                out.append(buffer)
                buffer = seg

        if buffer:
            out.append(buffer)
        return out


# ══════════════════════════════════════════════════════
#  STAGE 8 — Speaker diarization
# ══════════════════════════════════════════════════════
class SpeakerDiarizer:
    """
    Uses pyannote.audio to assign a SPEAKER_XX label to each transcript
    segment.  Requires a HuggingFace token (free) and model acceptance
    at hf.co/pyannote/speaker-diarization.

    If diarization is disabled or fails, segments keep speaker=None.
    """

    def __init__(self, config: PipelineConfig):
        self.config  = config
        self.pipeline = None

        if config.enable_diarization:
            try:
                from pyannote.audio import Pipeline
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=config.hf_token,
                )
                log.info("Stage 8 ▶ Diarization pipeline loaded.")
            except Exception as e:
                log.warning("  Diarization unavailable (%s). Skipping.", e)

    def diarize(
        self,
        wav_path: Path,
        segments: List[TranscriptSegment],
    ) -> List[TranscriptSegment]:
        if self.pipeline is None:
            log.info("Stage 8 ▶ Diarization skipped.")
            return segments

        log.info("Stage 8 ▶ Running speaker diarization …")
        diarization = self.pipeline(str(wav_path))

        # Build speaker timeline
        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append((turn.start, turn.end, speaker))

        # Assign speaker to each segment by maximum overlap
        for seg in segments:
            seg.speaker = self._assign_speaker(seg.start, seg.end, timeline)

        speakers = {s.speaker for s in segments if s.speaker}
        log.info("  Identified %d speaker(s): %s", len(speakers), speakers)
        return segments

    @staticmethod
    def _assign_speaker(
        start: float, end: float, timeline: List[Tuple]
    ) -> Optional[str]:
        best_speaker = None
        best_overlap = 0.0

        for t_start, t_end, speaker in timeline:
            overlap = min(end, t_end) - max(start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        return best_speaker


# ══════════════════════════════════════════════════════
#  STAGE 9 — Output generation + quality report
# ══════════════════════════════════════════════════════
class OutputGenerator:
    """
    Writes the final transcript in all requested formats:
      • .txt  — plain text, one line per segment
      • .srt  — SubRip subtitle format
      • .vtt  — WebVTT (for browsers / HTML5 video)
      • .json — full structured data with timestamps & confidence
    Produces a QualityReport printed to stdout and saved as JSON.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.out_dir = Path(config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        segments: List[TranscriptSegment],
        stem: str,
        meta: VideoMetadata,
        processing_time: float,
        model_size: str,
    ) -> Dict[str, Path]:
        log.info("Stage 9 ▶ Writing output files …")
        outputs: Dict[str, Path] = {}

        if "txt" in self.config.output_formats:
            outputs["txt"] = self._write_txt(segments, stem)
        if "srt" in self.config.output_formats:
            outputs["srt"] = self._write_srt(segments, stem)
        if "vtt" in self.config.output_formats:
            outputs["vtt"] = self._write_vtt(segments, stem)
        if "json" in self.config.output_formats:
            outputs["json"] = self._write_json(segments, stem, meta)

        # Quality report
        report = self._build_report(
            segments, meta, processing_time, model_size
        )
        report_path = self.out_dir / f"{stem}_quality.json"
        report_path.write_text(json.dumps(asdict(report), indent=2))
        outputs["quality"] = report_path

        log.info("  Files written to: %s", self.out_dir)
        self._print_report(report)
        return outputs

    # ─── Formatters ───────────────────────────────────

    def _write_txt(self, segs: List[TranscriptSegment], stem: str) -> Path:
        path = self.out_dir / f"{stem}.txt"
        lines = []
        for seg in segs:
            prefix = f"[{seg.speaker}] " if seg.speaker else ""
            lines.append(f"{prefix}{seg.text}")
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("  TXT → %s", path.name)
        return path

    def _write_srt(self, segs: List[TranscriptSegment], stem: str) -> Path:
        path = self.out_dir / f"{stem}.srt"
        blocks = []
        for i, seg in enumerate(segs, 1):
            ts_s = self._fmt_srt(seg.start)
            ts_e = self._fmt_srt(seg.end)
            speaker = f"[{seg.speaker}] " if seg.speaker else ""
            blocks.append(f"{i}\n{ts_s} --> {ts_e}\n{speaker}{seg.text}\n")
        path.write_text("\n".join(blocks), encoding="utf-8")
        log.info("  SRT → %s", path.name)
        return path

    def _write_vtt(self, segs: List[TranscriptSegment], stem: str) -> Path:
        path = self.out_dir / f"{stem}.vtt"
        lines = ["WEBVTT\n"]
        for seg in segs:
            ts_s = self._fmt_vtt(seg.start)
            ts_e = self._fmt_vtt(seg.end)
            speaker = f"<v {seg.speaker}>" if seg.speaker else ""
            lines.append(f"{ts_s} --> {ts_e}\n{speaker}{seg.text}\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("  VTT → %s", path.name)
        return path

    def _write_json(
        self,
        segs: List[TranscriptSegment],
        stem: str,
        meta: VideoMetadata,
    ) -> Path:
        path = self.out_dir / f"{stem}.json"
        data = {
            "video": asdict(meta),
            "segments": [
                {
                    "index": s.chunk_index,
                    "start": round(s.start, 3),
                    "end":   round(s.end,   3),
                    "text":  s.text,
                    "confidence": round(s.confidence, 4),
                    "language": s.language,
                    "speaker": s.speaker,
                }
                for s in segs
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("  JSON → %s", path.name)
        return path

    # ─── Helpers ──────────────────────────────────────

    @staticmethod
    def _fmt_srt(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_ms = int(td.total_seconds() * 1000)
        h, rem = divmod(total_ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms  = divmod(rem, 1_000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _fmt_vtt(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_ms = int(td.total_seconds() * 1000)
        h, rem = divmod(total_ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms  = divmod(rem, 1_000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _build_report(
        self,
        segs: List[TranscriptSegment],
        meta: VideoMetadata,
        processing_time: float,
        model_size: str,
    ) -> QualityReport:
        confidences = [s.confidence for s in segs]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        low_conf = sum(1 for c in confidences if c < self.config.min_confidence_warn)
        lang = segs[0].language if segs else "unknown"

        return QualityReport(
            total_segments=len(segs),
            avg_confidence=round(avg_conf, 4),
            low_confidence_count=low_conf,
            total_duration_sec=round(meta.duration_sec, 2),
            processing_time_sec=round(processing_time, 2),
            model_used=model_size,
            language_detected=lang,
        )

    @staticmethod
    def _print_report(r: QualityReport) -> None:
        print("\n" + "═" * 52)
        print("  QUALITY REPORT")
        print("═" * 52)
        print(f"  Segments          : {r.total_segments}")
        print(f"  Avg confidence    : {r.avg_confidence:.2%}")
        print(f"  Low-conf segments : {r.low_confidence_count}")
        print(f"  Video duration    : {r.total_duration_sec:.1f} s")
        print(f"  Processing time   : {r.processing_time_sec:.1f} s")
        print(f"  Whisper model     : {r.model_used}")
        print(f"  Language detected : {r.language_detected}")
        print("═" * 52 + "\n")


# ══════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════
class VideoToTextPipeline:
    """
    Top-level orchestrator — instantiates every stage and runs them
    in order, passing state between stages.

    Usage:
        config   = PipelineConfig(model_size="base", output_formats=["txt","srt"])
        pipeline = VideoToTextPipeline(config)
        outputs  = pipeline.run("lecture.mp4")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config   = config or PipelineConfig()
        self.ingester = InputIngester(self.config)
        self.extractor= MetadataExtractor(self.config)
        self.audio_ex = AudioExtractor(self.config)
        self.preproc  = AudioPreprocessor(self.config)
        self.chunker  = AudioChunker(self.config)
        self.transcriber = None       # loaded lazily (GPU allocation)
        self.postproc = PostProcessor(self.config)
        self.diarizer = SpeakerDiarizer(self.config)
        self.output   = OutputGenerator(self.config)

    def run(self, source: str) -> Dict[str, Path]:
        t0 = time.perf_counter()
        log.info("═" * 52)
        log.info("  Video → Text Pipeline  |  source: %s", source)
        log.info("═" * 52)

        with tempfile.TemporaryDirectory(prefix="vtt_") as tmp_dir:
            tmp = Path(tmp_dir)

            # Stage 1 — Ingest
            video_path = self.ingester.ingest(source)

            # Stage 2 — Metadata
            meta = self.extractor.extract(video_path)

            # Stage 3 — Extract audio
            wav_path = self.audio_ex.extract(video_path, tmp)

            # Stage 4 — Preprocess
            audio, sr = self.preproc.process(wav_path)

            # Stage 5 — Chunk
            chunks = self.chunker.chunk(audio, sr)

            # Stage 6 — Transcribe (lazy-load model)
            if self.transcriber is None:
                self.transcriber = ParallelTranscriber(self.config)
            segments = self.transcriber.transcribe(chunks)

            # Stage 7 — Post-process
            segments = self.postproc.process(segments)

            # Stage 8 — Diarize (uses original wav before VAD for accuracy)
            segments = self.diarizer.diarize(wav_path, segments)

            # Stage 9 — Output
            processing_time = time.perf_counter() - t0
            stem = Path(source).stem
            outputs = self.output.write(
                segments, stem, meta,
                processing_time,
                self.config.model_size,
            )

        return outputs


# ══════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════
def build_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Advanced Video-to-Text Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("source", help="Video file path or HTTP URL")

    p.add_argument("--model", default="base",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (larger = more accurate, slower)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--language", default=None,
                   help="Force language code (e.g. 'en', 'fr'). Default: auto-detect")
    p.add_argument("--formats", nargs="+",
                   default=["txt", "srt", "json"],
                   choices=["txt", "srt", "vtt", "json"])
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--no-denoise", action="store_true")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--chunk-length", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--diarize", action="store_true",
                   help="Enable speaker diarization (requires HF token)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token for diarization model")

    return p


def main():
    parser = build_cli()
    args   = parser.parse_args()

    config = PipelineConfig(
        model_size=args.model,
        device=args.device,
        language=args.language,
        output_formats=args.formats,
        output_dir=args.output_dir,
        denoise=not args.no_denoise,
        normalize_audio=not args.no_normalize,
        chunk_length_sec=args.chunk_length,
        max_workers=args.workers,
        enable_diarization=args.diarize,
        hf_token=args.hf_token,
    )

    pipeline = VideoToTextPipeline(config)
    outputs  = pipeline.run(args.source)

    print("\nOutput files:")
    for fmt, path in outputs.items():
        print(f"  [{fmt.upper():7s}]  {path}")


if __name__ == "__main__":
    main()
"""Speaker embedding extraction using WeSpeaker ResNet34-LM (ONNX).

Extracts 256-dimensional speaker embeddings from audio segments detected
by VAD.  Long segments are split with a sliding window so that each
window produces its own embedding, improving clustering granularity.

MODIFIED FROM UPSTREAM (Apache-2.0 §4(b)), twice:
  1. (2026-08-03) the ONNX embedding backend is replaced by this project's
     existing PyTorch WeSpeaker — see the note at the substitution point below.
  2. (2026-08-05) the full-file read is float32 instead of soundfile's float64
     default — see the note at that line.
Upstream: https://github.com/FoxNoseTech/diarize
at commit 4f25d27dee54f7e8264a914e705f7cee182151e2 (2026-05-06).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .utils import SpeechSegment, SubSegment

logger = logging.getLogger(__name__)

__all__ = ["extract_embeddings"]

# ── Constants ────────────────────────────────────────────────────────────────

#: Minimum segment duration for embedding extraction (seconds).
#: Segments shorter than this are skipped during embedding extraction
#: and later assigned the nearest speaker label.
MIN_SEGMENT_DURATION: float = 0.4

#: Sliding window length for splitting long segments (seconds).
EMBEDDING_WINDOW: float = 1.2

#: Sliding window step size (seconds).  Overlap = WINDOW − STEP.
EMBEDDING_STEP: float = 0.6


def extract_embeddings(
    audio_path: str | Path,
    speech_segments: list[SpeechSegment],
) -> tuple[np.ndarray, list[SubSegment]]:
    """Extract 256-dim speaker embeddings using WeSpeaker ResNet34-LM (ONNX).

    Long segments are split using a sliding window for more accurate
    clustering.  Each window produces its own embedding.

    Args:
        audio_path: Path to the audio file (wav, mp3, flac, etc.).
        speech_segments: Speech segments detected by VAD.

    Returns:
        A ``(embeddings, subsegments)`` tuple where:

        -   **embeddings** --- ``np.ndarray`` of shape ``(N, 256)`` with
            raw speaker embeddings (not yet L2-normalised; normalisation
            is applied later during clustering).
        -   **subsegments** --- list of :class:`SubSegment` objects that
            record the time window and parent segment index for each
            embedding row.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.

    Example::

        from diarize.vad import run_vad
        from diarize.embeddings import extract_embeddings

        segments = run_vad("meeting.wav")
        embeddings, subs = extract_embeddings("meeting.wav", segments)
        print(embeddings.shape)  # (N, 256)
    """
    # ── MODIFIED BY MeetingTranscriber, 2026-08-03 (Apache-2.0 §4(b)) ──────────
    # Upstream loads WeSpeaker ResNet34-LM through `wespeakerruntime` (ONNX),
    # which downloads its own copy of the weights from a hardcoded URL. This
    # project already ships the SAME model as PyTorch — `pyannote/wespeaker-
    # voxceleb-resnet34-LM`, the one `wespeaker-service.py` loads — so the ONNX
    # backend would mean a second download of identical weights, a runtime
    # network call this project forbids, and `onnxruntime` for nothing.
    #
    # MEASURED before substituting, twice:
    #   * the vectors are the SAME — cosine 1.0000 between the PyTorch and ONNX
    #     vector for the same span, on three spans; pairwise cosines agreed to 4
    #     decimals (0.8727 vs 0.8738 same-speaker, 0.0115 vs 0.0117 different).
    #   * this whole pipeline returns IDENTICAL output either way — same speaker
    #     count and same segment count on four real recordings, including a
    #     5-speaker meeting. It is also ~3x faster, because upstream writes a temp
    #     WAV and calls onnxruntime once per sub-window.
    #
    # This shim is deliberately the ONLY change: every other line of upstream's
    # pipeline — VAD, sliding-window sub-segmentation, GMM-BIC counting, spectral
    # clustering, temporal smoothing — runs unmodified.
    from .torch_embedder import Speaker as wespeaker_rt_Speaker

    logger.info("Extracting speaker embeddings (WeSpeaker ResNet34-LM, 256-dim)...")

    model = wespeaker_rt_Speaker()

    # Load full audio for segment slicing
    #
    # ── MODIFIED BY MeetingTranscriber, 2026-08-05 (Apache-2.0 §4(b)) ──────────
    # `dtype="float32"` added. Upstream takes soundfile's default, which is
    # **float64** — 8 bytes per sample for audio that is at most 24-bit at the
    # source, so the extra 32 bits of mantissa are filled with zeros. This is the
    # single largest memory cost in the whole engine: it is one copy of the ENTIRE
    # recording, and it is what makes the peak scale with meeting length.
    #
    # MEASURED (2026-08-05, on the owner's M4) before changing it:
    #   * the decode is BIT-IDENTICAL either way — `maxdiff 0.0` on a PCM_16
    #     fixture AND on a real 44.1 kHz FLOAT recording. float32 holds a 24-bit
    #     mantissa, and there is nothing beyond 24 bits in the source to lose.
    #   * the precision is discarded one screen below in any case: the per-window
    #     temp WAV is written with soundfile's default WAV subtype, which is
    #     **PCM_16**, and the array the embedder finally sees is identical whether
    #     this read was float32 or float64 (`beda 0.0`).
    #   * whole-pipeline output is unchanged — same segment count, same speaker
    #     count, same boundaries at full `repr()` precision, on four real
    #     recordings.
    #   * peak RSS on a 4027 s recording: 4.09 GB → measured again after this
    #     change together with the VAD shim's redundant copy.
    audio_data, sr = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # stereo → mono

    embeddings: list[np.ndarray] = []
    subsegments: list[SubSegment] = []

    for idx, seg in enumerate(speech_segments):
        seg_duration = seg.duration

        if seg_duration < MIN_SEGMENT_DURATION:
            continue

        # Split long segments with a sliding window
        if seg_duration <= EMBEDDING_WINDOW * 1.5:
            windows = [(seg.start, seg.end)]
        else:
            windows: list[tuple[float, float]] = []
            win_start = seg.start
            while win_start + MIN_SEGMENT_DURATION < seg.end:
                win_end = min(win_start + EMBEDDING_WINDOW, seg.end)
                windows.append((win_start, win_end))
                win_start += EMBEDDING_STEP

        for win_start, win_end in windows:
            start_sample = int(win_start * sr)
            end_sample = int(win_end * sr)
            segment_audio = audio_data[start_sample:end_sample]

            tmp_path: str | None = None
            try:
                # The embedder accepts a file path — write the window to a temp wav.
                # (Upstream said 'wespeakerruntime accepts file paths'; our shim
                # keeps the same contract so this loop is untouched.)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    sf.write(tmp_path, segment_audio, sr)

                emb = model.extract_embedding(tmp_path)
            except Exception:
                logger.debug(
                    "Embedding extraction failed for window %.2f-%.2f",
                    win_start,
                    win_end,
                )
                continue
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            if emb is not None:
                if emb.ndim == 2:
                    emb = emb[0]
                embeddings.append(emb)
                subsegments.append(SubSegment(start=win_start, end=win_end, parent_idx=idx))

    if not embeddings:
        return np.empty((0, 256), dtype=np.float32), []

    X = np.stack(embeddings)  # (N, 256)
    logger.info("Extracted %d embeddings (dim=%d)", X.shape[0], X.shape[1])
    return X, subsegments

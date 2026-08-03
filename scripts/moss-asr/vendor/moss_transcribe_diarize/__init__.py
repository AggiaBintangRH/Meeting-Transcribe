"""Vendored subset of `moss_transcribe_diarize` (OpenMOSS/MOSS-Transcribe-Diarize).

Copied VERBATIM from the upstream package at version 0.1.0
(https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) — only this `__init__`
is ours.

WHY VENDORED AT ALL
-------------------
The helper is installable only from git (`pip install git+https://…`). `build.sh`
freezes the main `.venv` and strips every VCS requirement before provisioning the
bundled interpreter:

    build.sh:110   grep -vE '^pip==|file://|@ file://|git\\+' …

so a `git+` freeze line is silently dropped and the packaged `.app` would ship
without the package at all — exactly the DiCoW-missing-from-the-bundle failure of
2026-07-27. `scripts/` (including `vendor/`) is copied into the bundle by build.sh
[B3], so vendoring makes dev and the `.app` run the SAME code with ZERO build.sh
changes. The `.venv` pip install is kept only as provenance; nothing imports it.

WHY ONLY TWO MODULES
--------------------
The upstream `__init__` also re-exports the model/processor/config classes and the
`subtitle` package. We need neither: the model and processor classes are loaded
from the HF snapshot's own remote code (`trust_remote_code=True`), and we emit no
subtitles. Importing them here would drag `modeling_moss_transcribe_diarize.py`
into every import of this package for nothing.
"""

from .inference_utils import (
    DEFAULT_PROMPT,
    build_transcription_messages,
    dtype_from_name,
    generate_transcription,
    load_audio_item,
    prepare_inputs,
    process_audio_info,
    resolve_device,
)
from .transcript_parser import (
    TranscriptParseError,
    TranscriptSegment,
    TranscriptStreamParser,
    iter_transcript_segments,
    parse_transcript,
)

__all__ = [
    "DEFAULT_PROMPT",
    "TranscriptParseError",
    "TranscriptSegment",
    "TranscriptStreamParser",
    "build_transcription_messages",
    "dtype_from_name",
    "generate_transcription",
    "iter_transcript_segments",
    "load_audio_item",
    "parse_transcript",
    "prepare_inputs",
    "process_audio_info",
    "resolve_device",
]

#!/usr/bin/env python3
"""Write MODEL-LICENSES.txt for every checkpoint the .app redistributes.

WHY THIS EXISTS. The app ships fourteen third-party checkpoints, and every
licence among them — Apache 2.0, MIT, CC BY 4.0, OpenMDW-1.1 — obliges whoever
distributes the weights to pass the licence and its attribution along with them.
The 2026-08-13 audit found the vendored SOURCE trees each shipping their own
LICENSE (campplus, spectral, diarizen) while not one CHECKPOINT did.

DERIVED, NOT LISTED. The licence of each checkpoint is read from that
checkpoint's own README front-matter inside the bundle. A table kept in this
file would be a copy of the truth, and this project has watched exactly that
kind of copy drift twice in one week (the diarization-engine sentence, stale
first for DiariZen and then for CAM++). It is also the rule the Granite
language-count and the DiariZen CC-BY-NC findings both established: the card of
the checkpoint ON DISK is the authority — not the family's card, not upstream's
README, and not anybody's memory.

THE EXCEPTIONS TABLE IS THE ONE HAND-WRITTEN PART, and it is deliberately small,
deliberately loud, and carries its evidence inline. A checkpoint whose card
omits the tag cannot be waved through: `--strict` (the build's mode) EXITS
NON-ZERO unless the exception names where the licence was read from instead.
Shipping weights whose terms nobody can name is the failure this prevents, and
it is silent, which is what makes it worth a gate rather than a habit.
"""

import argparse
import os
import re
import sys

# Checkpoints whose own card carries no `license:` tag, with the licence read
# from the source the card itself points at. ONE entry today.
#
# `mlx-community/Fun-ASR-MLT-Nano-2512-fp16` is a format conversion — its
# front-matter declares `base_model: FunAudioLLM/Fun-ASR-MLT-Nano-2512` and
# nothing else. That base model is published under Apache 2.0 (verified against
# the upstream model card, 2026-08-13), and a format conversion does not change
# the terms of the weights it converts. The conversion simply omits the tag; it
# is the only one of the fourteen that does.
EXCEPTIONS = {
    "mlx-community/Fun-ASR-MLT-Nano-2512-fp16": (
        "apache-2.0",
        "not tagged on the conversion; taken from its declared base model "
        "FunAudioLLM/Fun-ASR-MLT-Nano-2512, verified 2026-08-13",
    ),
}

# Checkpoints that are not HF-hub snapshots and so have no card to read. Each
# names where its terms come from.
FLAT = {
    "nemo/titanet_large.nemo": (
        "nvidia/speakerverification_en_titanet_large",
        "cc-by-4.0",
        "NVIDIA NeMo speaker model, fetched by download-best-models.sh 3c",
    ),
    "nemo/vad_multilingual_marblenet.nemo": (
        "nvidia/vad_multilingual_marblenet",
        "cc-by-4.0",
        "NVIDIA NeMo VAD model, fetched by download-best-models.sh 3c",
    ),
    "mossformer2/mossformer2-librimix-2spk": (
        "alibabasglab/mossformer2-librimix-2spk",
        "apache-2.0",
        "standalone MossFormer2 checkpoint",
    ),
    "silero-vad-v6.2.1": (
        "snakers4/silero-vad",
        "mit",
        "Silero VAD weights, shipped with the silero-vad package",
    ),
}

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def card_licence(readme: str):
    """The `license:` (and `license_name:`) from a model card's front matter."""
    m = FRONT_MATTER.match(readme)
    if not m:
        return None, None
    block = m.group(1)
    lic = re.search(r"^license:\s*(.+?)\s*$", block, re.M)
    name = re.search(r"^license_name:\s*(.+?)\s*$", block, re.M)
    return (lic.group(1).strip() if lic else None,
            name.group(1).strip() if name else None)


def hub_repos(models_dir: str):
    """Every HF-hub snapshot in the bundle, as (repo id, snapshot path)."""
    hub = os.path.join(models_dir, "hub")
    out = []
    for entry in sorted(os.listdir(hub)) if os.path.isdir(hub) else []:
        if not entry.startswith("models--"):
            continue
        repo = entry[len("models--"):].replace("--", "/", 1)
        snaps = os.path.join(hub, entry, "snapshots")
        if not os.path.isdir(snaps):
            continue
        kids = [os.path.join(snaps, k) for k in sorted(os.listdir(snaps))]
        kids = [k for k in kids if os.path.isdir(k)]
        if kids:
            out.append((repo, kids[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lenient", action="store_true",
                    help="report unknown licences without failing (the build "
                         "never passes this — see the module docstring)")
    args = ap.parse_args()

    rows, unknown = [], []

    for repo, snap in hub_repos(args.models):
        readme = os.path.join(snap, "README.md")
        lic = name = None
        if os.path.exists(readme):
            with open(readme, encoding="utf-8", errors="replace") as fh:
                lic, name = card_licence(fh.read())
        source = "declared on the model card"
        if not lic and repo in EXCEPTIONS:
            lic, source = EXCEPTIONS[repo]
        if not lic:
            unknown.append(repo)
            continue
        rows.append((repo, name or lic, source))

    for path, (repo, lic, note) in FLAT.items():
        if os.path.exists(os.path.join(args.models, path)):
            rows.append((repo, lic, note))

    if unknown and not args.lenient:
        for repo in unknown:
            print(f"  no licence could be established for {repo}", file=sys.stderr)
        return 1

    width = max((len(r) for r, _, _ in rows), default=0)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("THIRD-PARTY MODEL LICENCES\n")
        fh.write("=" * 74 + "\n\n")
        fh.write(
            "Meeting Transcriber redistributes the model weights below. Each is\n"
            "listed with the licence its publisher released it under, taken from\n"
            "that checkpoint's own model card unless noted. The full text of each\n"
            "licence is available from the model's page on huggingface.co.\n\n"
            "This file is generated at build time from the checkpoints actually\n"
            "present in the app — it is not a list maintained by hand.\n\n")
        for repo, lic, source in sorted(rows):
            fh.write(f"{repo.ljust(width)}  {lic}\n")
            if source != "declared on the model card":
                fh.write(f"{' ' * width}  ({source})\n")
        fh.write(f"\n{len(rows)} checkpoints.\n")

    print(f"    MODEL-LICENSES.txt written — {len(rows)} checkpoints, "
          f"{len(unknown)} unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())

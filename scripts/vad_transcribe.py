"""VAD-segmented transcription for mlx-whisper (Mac) or faster-whisper (PC).

Fine-tuned Japanese models such as anime-whisper were trained on short single
lines and never emit timestamp tokens, so whisper's long-form mode returns one
30-second cue per window. Instead, Silero VAD finds each stretch of speech,
each stretch is transcribed on its own, and the VAD times become the subtitle
timings.

Writes <output-dir>/<video name>.srt, the same contract as the whisper CLIs,
and prints "[start --> end] text" lines that the server's progress parser reads.

Requires, in the backend's venv:
  --engine mlx: mlx-whisper, silero-vad
  --engine ct2: faster-whisper (bundles its own Silero VAD)
"""

import argparse
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

SAMPLE_RATE = 16000

# Long cues are split at Japanese punctuation so each subtitle stays readable
SPLIT_CHARS = 40
SPLIT_RE = re.compile(r"(?<=[、。!?！？…])")


def fmt(t, sep=","):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def split_cue(start, end, text):
    """Split a long cue at punctuation, dividing time by character count."""
    if len(text) <= SPLIT_CHARS:
        return [(start, end, text)]
    parts, cur = [], ""
    for piece in filter(None, SPLIT_RE.split(text)):
        if cur and len(cur) + len(piece) > SPLIT_CHARS:
            parts.append(cur)
            cur = ""
        cur += piece
    if cur:
        parts.append(cur)
    total = sum(len(p) for p in parts)
    cues, t = [], start
    for p in parts:
        dur = (end - start) * len(p) / total
        cues.append((t, t + dur, p))
        t += dur
    return cues


def load_audio_ffmpeg(path):
    """16 kHz mono float32 via the ffmpeg CLI (the app already requires it).
    Avoids faster-whisper's PyAV decoder, which breaks with some PyAV versions."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path, "-f", "s16le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-loglevel", "error", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        # ffmpeg's own last line ("Invalid data found when processing input")
        # becomes the item's error in the app, instead of a Python traceback
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        sys.exit(f"ERROR: ffmpeg could not read {path}: {lines[-1] if lines else f'exit code {proc.returncode}'}")
    return np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0


def mlx_engine(args):
    """Returns (audio, vad_fn, transcribe_fn) backed by mlx-whisper + silero-vad."""
    import mlx_whisper
    import torch
    from mlx_whisper.audio import load_audio
    from silero_vad import get_speech_timestamps, load_silero_vad

    audio = np.array(load_audio(args.audio), dtype=np.float32)  # 16 kHz mono via ffmpeg
    vad = load_silero_vad()

    def find_speech():
        return get_speech_timestamps(
            torch.from_numpy(audio), vad, sampling_rate=SAMPLE_RATE,
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_ms,
            min_silence_duration_ms=args.vad_min_silence_ms,
            max_speech_duration_s=args.vad_max_segment_s,
            speech_pad_ms=args.vad_speech_pad_ms,
        )

    def transcribe(chunk):
        result = mlx_whisper.transcribe(
            chunk, path_or_hf_repo=args.model, language=args.language, task=args.task,
            temperature=args.temperature, initial_prompt=args.initial_prompt or None,
            condition_on_previous_text=False, without_timestamps=True,
            fp16=args.fp16.lower() == "true", verbose=None,
        )
        return "".join(s["text"] for s in result["segments"])

    return audio, find_speech, transcribe


def ct2_engine(args):
    """Returns (audio, vad_fn, transcribe_fn) backed by faster-whisper (CTranslate2)."""
    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = load_audio_ffmpeg(args.audio)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    def find_speech():
        return get_speech_timestamps(audio, VadOptions(
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_ms,
            min_silence_duration_ms=args.vad_min_silence_ms,
            max_speech_duration_s=args.vad_max_segment_s,
            speech_pad_ms=args.vad_speech_pad_ms,
        ))

    def transcribe(chunk):
        segments, _ = model.transcribe(
            chunk, language=args.language, task=args.task, beam_size=args.beam_size,
            temperature=args.temperature, initial_prompt=args.initial_prompt or None,
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
            # Against "ああああ…" loops on long cries or shouts (e.g. 1.5 / 3)
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        return "".join(s.text for s in segments)

    return audio, find_speech, transcribe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--engine", choices=["mlx", "ct2"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--initial-prompt", default=None)
    # Defaults: tuned "balanced" Silero values (the app passes its own)
    ap.add_argument("--vad-threshold", type=float, default=0.28)
    ap.add_argument("--vad-min-speech-ms", type=int, default=100)
    ap.add_argument("--vad-min-silence-ms", type=int, default=300)
    ap.add_argument("--vad-max-segment-s", type=float, default=15.0)
    ap.add_argument("--vad-speech-pad-ms", type=int, default=400)
    # mlx only
    ap.add_argument("--fp16", default="True")
    # ct2 only
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    args = ap.parse_args()

    audio, find_speech, transcribe = (mlx_engine if args.engine == "mlx" else ct2_engine)(args)
    segments = find_speech()
    print(f"VAD: {len(segments)} speech segments", flush=True)

    cues = []
    for seg in segments:
        start, end = seg["start"] / SAMPLE_RATE, seg["end"] / SAMPLE_RATE
        text = transcribe(audio[seg["start"]:seg["end"]]).strip()
        print(f"[{fmt(start, '.')} --> {fmt(end, '.')}] {text}", flush=True)
        if text:
            cues.extend(split_cue(start, end, text))

    out = Path(args.output_dir) / (Path(args.audio).stem + ".srt")
    with open(out, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt(s)} --> {fmt(e)}\n{t}\n\n")
    print(f"Wrote {len(cues)} cues to {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

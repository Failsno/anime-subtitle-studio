# Mac Mini (M5) setup

This app now supports two whisper backends, picked via `config.json`'s
`"backend"` field:

- `ctranslate2` — whisper-ctranslate2, CUDA or CPU only (used on the PC).
- `mlx` — mlx-whisper, uses the Mac's GPU via Apple's MLX/Metal framework.

CTranslate2 has no Metal/MPS support, so on the Mac Mini we use `mlx-whisper`
to actually get GPU acceleration out of the M5's unified memory.

## 1. Get the code onto the Mac

From this PC, push/copy the repo (excluding `node_modules` and your local
`config.json`, which is machine-specific) to the Mac — e.g. via a git remote,
or `rsync`/AirDrop/USB. Once it's on the Mac:

```bash
cd anime-subtitle-studio
npm install
```

## 2. Install Node (if not already installed)

```bash
brew install node
```

## 3. Install ffmpeg (needed for progress estimation via ffprobe)

```bash
brew install ffmpeg
```

## 4. Install mlx-whisper

Use a Python virtual environment so it doesn't collide with system Python:

```bash
python3 -m venv ~/mlx-whisper-env
source ~/mlx-whisper-env/bin/activate
pip install -U mlx-whisper silero-vad
```

Confirm the CLI is on your PATH while the venv is active:

```bash
which mlx_whisper
```

Since `npm start`/`npm run dev` launch `node`, which then spawns
`mlx_whisper` as a subprocess, the venv needs to be **activated in the same
shell** you launch the Node server from — or set `mlxExecutable` in
`config.json` to the venv's absolute path, e.g.:

```
~/mlx-whisper-env/bin/mlx_whisper
```

That way it works regardless of which shell/venv state launched `node`.

## 5. First run — let it create config.json

```bash
npm start
```

On first run (no existing `config.json`), the server detects macOS and
writes defaults with `"backend": "mlx"` already set, watch folder under
`~/Videos/Japanese`, and model `mlx-community/whisper-large-v3-mlx`.

Open `http://localhost:3939`, then in **Settings**:
- Set **Watch folder** to wherever your `.mp4` files live on the Mac.
- Confirm **Backend** is `mlx-whisper (Apple Silicon GPU)`.
- Set **mlx_whisper executable** to the venv path from step 4 if you didn't
  activate the venv in the launching shell.
- Adjust `--language` / `--initial_prompt` / etc. as needed — these carry
  over independently of the PC's whisper-ctranslate2 settings.

## 6. anime-whisper model (recommended)

[anime-whisper](https://huggingface.co/litagin/anime-whisper) transcribes
Japanese anime dialogue much better than stock whisper. Convert it to
MLX once with Apple's script:

```bash
source ~/mlx-whisper-env/bin/activate
curl -LO https://raw.githubusercontent.com/ml-explore/mlx-examples/main/whisper/convert.py
python convert.py --torch-name-or-path litagin/anime-whisper --mlx-path ~/models/anime-whisper-mlx --dtype float16
# mlx-whisper 0.4.x loads weights.safetensors; newer convert.py writes model.safetensors
mv ~/models/anime-whisper-mlx/model.safetensors ~/models/anime-whisper-mlx/weights.safetensors
```

Then in **Settings** set mlx `--model` to the full path
(`/Users/<you>/models/anime-whisper-mlx`), keep **VAD segmentation** `True`
(anime-whisper has no timestamps and needs it), and leave `--initial-prompt`
blank. Expect about 1.5 GB on disk. In testing, 59 seconds of speech
transcribed in about 3 seconds on the M5.

Optional alternative, whisper-ja-1.5B (about 3 GB, slower, similar accuracy):

```bash
python convert.py --torch-name-or-path efwkjn/whisper-ja-1.5B --mlx-path ~/models/whisper-ja-1.5B-mlx --dtype float16
mv ~/models/whisper-ja-1.5B-mlx/model.safetensors ~/models/whisper-ja-1.5B-mlx/weights.safetensors
```

## 6b. Stock model download

The first job will auto-download the model from Hugging Face
(`mlx-community/whisper-large-v3-mlx`, a few GB) into `~/.cache/huggingface`.
That download only happens once.

## 7. Local translation with llama-server

On the Mac, translation runs locally through llama.cpp's `llama-server`
(OpenAI-compatible API) instead of OpenRouter:

```bash
brew install llama.cpp
llama-server -m ~/models/<your-model>.gguf --port 9931 -ngl 99 -c 16384 --jinja --reasoning-budget 0
```

- `--port 9931` is llama.cpp's upcoming default port; set it explicitly so
  older and newer builds behave the same.
- `--reasoning-budget 0` turns off "thinking" on Qwen3-style models, which is
  much faster for subtitle translation.
- 48GB of unified memory comfortably fits a ~30B-class Qwen model at Q4–Q5
  alongside mlx-whisper.

Then in **Settings → Translation**: choose **Two-stage**, click the
**llama.cpp (local)** preset (`http://127.0.0.1:9931/v1`), and leave the API
key and model blank. Fresh macOS installs already default to this URL.

llama-server must be running before the queue reaches the translation step;
if it isn't, the item errors with a connection failure and the `.ja.srt` is
kept, so **Retry** later only re-runs the translation.

Or let the app start it: in **Settings → llama-server auto-start** set the
executable (`llama-server`, or its full path), the `.gguf` model and the extra
args above (without `-m`/`--port`). The app starts it when a video first needs
translating, waits up to **Model load wait** (default 15 s) for the model, and
**Shut down server** stops it.

## Notes

- `config.json` is per-machine and not meant to be shared between the PC and
  the Mac — each keeps its own backend/model/paths. If you do sync the repo
  via git, keep `config.json` out of version control (or `.gitignore` it) so
  each machine's settings don't clobber each other.
- mlx-whisper has no built-in VAD filter, so the app does its own:
  `scripts/vad_transcribe.py --engine mlx` (Silero VAD) runs when **VAD segmentation** is
  `True`, using the `python` in the same venv as `mlx_whisper`. Set it to
  `False` to use the plain `mlx_whisper` CLI.
- To stop the app, use the gear icon (top left) → **Shut down server**. That
  stops any running job and frees the port; closing the Terminal window now
  does the same. Port 3939 can be changed in Settings (applies on reboot).
- The PC equivalent of this guide is [`WINDOWS_CUDA_SETUP.md`](WINDOWS_CUDA_SETUP.md).
- Progress estimation (ffprobe duration → estimated segment count) works the
  same on both backends since mlx-whisper's verbose segment-timestamp output
  (`[00:00:00.000 --> 00:00:04.320] ...`) matches the same format the
  progress parser already expects.

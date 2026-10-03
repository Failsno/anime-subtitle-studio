# Windows 11 PC (NVIDIA RTX 5090) setup

On the PC the app uses the `ctranslate2` backend: `whisper-ctranslate2` /
faster-whisper on the GPU via CUDA for transcription, and a **local
llama-server** (llama.cpp) on the same GPU for translation, which the app
starts and stops by itself. The whole pipeline runs on this machine, offline;
the Mac isn't needed. OpenRouter (a cloud API) is an alternative for
translation (step 7).

```
video.mp4 ──▶ VAD + anime-whisper (CUDA) ──▶ video.ja.srt ──▶ llama-server (CUDA) ──▶ video.en.srt
```

> Tested end to end on the RTX 5090 (anime-whisper + a 27B Qwen model). Run
> `npm test` once after setting up (step 9) to confirm it on your machine.

All commands below are for **PowerShell**. `$HOME` is `C:\Users\<you>`.

## 1. Get the code onto the PC

Clone or pull the repo (your `config.json` is gitignored, so pulling won't
touch it):

```powershell
cd $HOME\Documents\GitHub\anime-subtitle-studio
git pull
npm install
```

## 2. Install Node, Python and ffmpeg (if not already installed)

```powershell
winget install OpenJS.NodeJS.LTS
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```

Open a new terminal afterwards so they're on PATH, then check:

```powershell
node -v; python --version; ffmpeg -version | Select-Object -First 1
```

ffmpeg is required: `ffprobe` estimates progress, the Concat tab uses
`ffmpeg`, and the VAD script decodes audio with it.

## 3. NVIDIA driver and CUDA 12 libraries

If `whisper-ctranslate2` already runs on the GPU on this PC, skip this step.

- **Driver:** a current GeForce driver (RTX 50-series needs R570 or newer).
- **CUDA libraries:** CTranslate2 needs **cuBLAS for CUDA 12** and **cuDNN 9
  for CUDA 12**. Install the CUDA Toolkit **12.8** (the first 12.x with
  Blackwell support) and cuDNN 9 from NVIDIA, and make sure their `bin` folders
  are on PATH. The [faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu)
  also links prebuilt Windows DLLs as an alternative.
- **Don't use CUDA 13.** CTranslate2 is built against CUDA 12 and fails to
  load on 13 (`libcublas.so.12` / `cublas64_12.dll` not found).

## 4. Python environment for whisper

Use a virtual environment so the app always finds the right Python:

```powershell
python -m venv $HOME\whisper-env
& $HOME\whisper-env\Scripts\pip install -U whisper-ctranslate2 "ctranslate2>=4.8.1" "av<16"
& $HOME\whisper-env\Scripts\whisper-ctranslate2 --help | Select-Object -First 1
```

Pin **`av<16`**: faster-whisper 1.2.1 crashes decoding audio with PyAV 19
(`open() got an unexpected keyword argument 'metadata_errors'`). PyAV 15.1
works.

The app runs `scripts\vad_transcribe.py` with the `python.exe` that sits next
to `whisper-ctranslate2.exe` (here `$HOME\whisper-env\Scripts\python.exe`), so
faster-whisper must be importable from that Python. The venv takes care of
that. If you installed whisper-ctranslate2 somewhere else, set **Settings →
Python (VAD script)** to the matching `python.exe`.

## 5. Models

Local models live in `$HOME\models\<name>`. In Settings you can write that as
`~/models/<name>`; the app expands `~` to your home folder.

The conversion tools need Transformers and PyTorch (CPU-only is fine; they're
only used for converting):

```powershell
& $HOME\whisper-env\Scripts\pip install "transformers<5" torch
```

### anime-whisper (recommended)

[anime-whisper](https://huggingface.co/litagin/anime-whisper) (MIT) is
fine-tuned on anime and visual-novel dialogue and transcribes laughs, gasps and
other non-speech vocalizations properly instead of hallucinating on them. Convert it
to CTranslate2 format once (about 1.5 GB):

```powershell
& $HOME\whisper-env\Scripts\ct2-transformers-converter --model litagin/anime-whisper --output_dir $HOME\models\anime-whisper-ct2 --copy_files preprocessor_config.json --quantization float16
& $HOME\whisper-env\Scripts\hf download openai/whisper-large-v3 tokenizer.json --local-dir $HOME\models\anime-whisper-ct2
```

**Don't skip the second command.** anime-whisper's repo has no `tokenizer.json`.
Without one, faster-whisper falls back to a tokenizer with a different
vocabulary, every special token is off by one, and the output is gibberish in
random languages. Third-party anime-whisper CT2 uploads may have the same
problem, so convert it yourself.

anime-whisper has two other quirks:
- It **never outputs timestamps**, so it must run with **VAD segmentation**
  (the app transcribes each stretch of speech separately). With it off,
  whisper-ctranslate2 crashes building word timestamps.
- Keep **--initial_prompt blank**. Its author says prompts make it hallucinate.

Picking anime-whisper in the Settings dropdown handles both.

### Optional: whisper-ja-1.5B

[whisper-ja-1.5B](https://huggingface.co/efwkjn/whisper-ja-1.5B) is a full
large-v3 fine-tune: about 3 GB and roughly twice as slow, with accuracy roughly
level with anime-whisper on its author's benchmarks. Its model card doesn't
state a license. [TransWithAI](https://huggingface.co/TransWithAI/whisper-ja-1.5B-ct2)
publishes it already converted (with `tokenizer.json`):

```powershell
& $HOME\whisper-env\Scripts\hf download TransWithAI/whisper-ja-1.5B-ct2 --local-dir $HOME\models\whisper-ja-1.5B-ct2
```

### Optional: whisper-ja-anime-v0.3 (experimental)

[whisper-ja-anime-v0.3](https://huggingface.co/efwkjn/whisper-ja-anime-v0.3) is
a large-v3-turbo fine-tune with its own, smaller Japanese vocabulary (about
1.5 GB converted). Treat it as experimental. It ships
`tokenizer.json`, so one conversion is enough:

```powershell
& $HOME\whisper-env\Scripts\ct2-transformers-converter --model efwkjn/whisper-ja-anime-v0.3 --output_dir $HOME\models\whisper-ja-anime-v0.3-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization float16
```

### Optional: kotoba-whisper v2.0 (fast general Japanese)

```powershell
& $HOME\whisper-env\Scripts\hf download kotoba-tech/kotoba-whisper-v2.0-faster --local-dir $HOME\models\kotoba-whisper-v2.0-faster
```

kotoba-whisper **v2.1** and **v2.2** are this same model (their weights are
v2.0's, stored at higher precision) plus punctuation and speaker-diarization
steps that only run inside Hugging Face's Python pipeline, not in
faster-whisper. For this app, v2.0-faster is the one to use.

### Which model?

Measured on this PC (RTX 5090) with VAD segmentation, on the test clip (59 s,
known transcript) and a 125-minute video:

| Model | Test clip errors | 125-min video | Lines found |
|---|---|---|---|
| anime-whisper (default) | 0.8 % (none real) | 50 s | 186 |
| whisper-ja-1.5B | 0.9 % (none real) | 60 s | 177 |
| whisper-ja-anime-v0.3 | 6.5 % (one misheard line) | 27 s | 165 |
| kotoba-whisper v2.0 | 3.0 % (one meaning flip: うん → ううん) | 27 s | 177 |

anime-whisper stays the default. If you need hard-to-hear lines, compare
another model with **EN↻** on a video you know.

If `hf download` stalls at the end of a large file, stop it, delete the
model folder's `.cache` subfolder and run it again.

Stock `large-v3` / `large-v2` need no setup: whisper-ctranslate2 downloads them
on first use.

## 6. First run and Settings

```powershell
npm start
```

Open `http://localhost:3939`, then **Settings**:

| Setting | Value |
|---|---|
| Backend | `whisper-ctranslate2 (NVIDIA CUDA / CPU)` |
| Whisper executable | `C:\Users\<you>\whisper-env\Scripts\whisper-ctranslate2.exe` |
| Python (VAD script) | blank (auto-detected next to the executable) |
| --model | choose **anime-whisper** in the dropdown (turns VAD segmentation on and clears the initial prompt) |
| --device | `cuda` |
| --compute_type | **`int8_float16`** |
| --language / --task | `ja` / anything (two-stage mode always transcribes) |

**If your `config.json` predates this update, change `--compute_type` to
`int8_float16`** (it was `float32`). On RTX 50-series cards, `float16` can make
CTranslate2 decode some 30-second windows into garbage that carries over to the
next ones: one RTX 5070 test produced 0 subtitles on float16 and 62 on
int8_float16.
That needs CTranslate2 4.8.1 or newer (4.6.2 turned INT8 off on these cards),
which the `pip install` line in step 4 pins. Compare both on your own clips
with `npm test` (step 9) if you like: `$env:WT_TEST_CT2_COMPUTE = "float16"`.

Hover the **ⓘ** next to --model for how to install any other model. Custom
models must be CTranslate2 folders; whisper-ctranslate2's `--model` only
accepts stock names, so the app passes anything else as `--model_directory`.

## 7. Translation with a local llama-server

The 32 GB card runs a ~27–30B model (Q4–Q6, about 20 GB) alongside whisper.
Translation stays on the PC: no API key, no per-request cost, no content
filter beyond the model's own.

**Install llama.cpp** with GPU support:

```powershell
winget install llama.cpp
llama-server --list-devices
```

The device list must show `CUDA0: NVIDIA GeForce RTX 5090`. If it lists only
the CPU, download a CUDA build instead (`llama-…-bin-win-cuda-12…-x64.zip`
from llama.cpp's GitHub releases), unzip it, e.g. to `C:\llama-cuda`, and use
that `llama-server.exe` below.

**Get a model**: a `.gguf` file. For Japanese anime dialogue, pick an
instruction-tuned Qwen or Gemma 27B at Q4_K_M–Q6_K; a smaller model is faster
but misses more nuance, and any line it drops stays in Japanese. Download with
`hf download <repo> <file>.gguf` (it lands in the Hugging Face cache, which
the app scans) or save it under `$HOME\models`.

**In Settings → Translation (LLM):**

1. Mode: **Two-stage**.
2. Click the **llama.cpp (local)** preset (`http://127.0.0.1:9931/v1`). No API
   key or model id is needed.
3. Under **llama-server auto-start**: set the executable (e.g.
   `C:\llama-cuda\llama-server.exe`), pick your `.gguf` in the **Model**
   dropdown, and keep the default extra args
   (`-ngl 99 -c 16384 --jinja --reasoning-budget 0`; the port comes from the
   Base URL).
4. Click **Save & test translation**. The first test starts llama-server and
   loads the model; it should end with a translated sample line.
5. Leave **Style prompt** blank to use the built-in one (shown as the
   placeholder), or write your own.

llama-server starts when a video first reaches translation and the app waits
up to **Model load wait** (default 15 s; raise it if a big model loads more
slowly, e.g. right after a reboot). **Shut down server** in the gear menu stops
llama-server too; **Reboot server** keeps the model loaded. **Currently running**
in Settings shows the loaded model.

Then set the watch folder and press **Start**. Each video gets
`video.srt` (English subtitles) next to it, which video players load
automatically. With **Keep only English subtitles** unticked you get
`video.ja.srt` (Japanese transcript) and `video.en.srt` instead. The 📂 button
on a queue item opens its folder with the subtitles selected. If translation fails or you press Stop, the `.ja.srt` is kept and
the next run or **Retry** only redoes the translation (**EN↻** needs the
`.ja.srt` kept).

### Comparing models

The **Model** dropdown lists the `.gguf` files already on the PC (Hugging
Face cache and `$HOME\models`). To compare two models, untick **Keep only
English subtitles**, translate a video, pick the other model, Save, and press
**EN↻**: llama-server restarts with the new model first. EN↻ overwrites
`video.en.srt`, so copy it aside (or use the 👁 subtitle preview) to compare.

### Running llama-server yourself

Leave the auto-start executable blank and start it in its own terminal before
pressing **Start**:

```powershell
llama-server -m $HOME\models\<model>.gguf --port 9931 -ngl 99 -c 16384 --jinja --reasoning-budget 0
```

### Alternative: OpenRouter

A cloud API instead of the local model: no VRAM needed, but pay-per-use and
the provider's content rules apply.

1. Click the **OpenRouter** preset (`https://openrouter.ai/api/v1`).
2. Paste your API key. It's stored only in `config.json` on this PC and never
   sent to the browser. Alternatively, leave it blank and put
   `OPENROUTER_API_KEY=...` in a `.env` file in the repo folder (copy
   `.env.example`; it's gitignored) or set it as an environment variable.
3. Enter a model id. Lines a model drops show up in the log as missing lines
   and are re-requested, then left in Japanese.
4. **Save & test translation** to check the key and model.

## 8. Concat tab

Uses the same `ffmpeg` from step 2. Pick videos in playback order, name the
output, and press Concat. It's a stream copy (no re-encode, no GPU), so all
parts must share codec, resolution and frame rate.

## 9. Run the tests

`npm test` runs end-to-end tests against real servers with temporary configs
(your `config.json` isn't touched), using a short Japanese test clip. Tests
whose requirements are missing are skipped, with the reason shown.

```powershell
$env:WT_TEST_CT2_EXE = "$HOME\whisper-env\Scripts\whisper-ctranslate2.exe"
npm test
```

The translation tests use `http://127.0.0.1:9931/v1` and skip if nothing
answers there, so start llama-server first (e.g. **Save & test translation**
in Settings). To test against OpenRouter instead, also set
`$env:WT_TEST_LLM_URL = "https://openrouter.ai/api/v1"`,
`$env:WT_TEST_LLM_KEY` and `$env:WT_TEST_LLM_MODEL`.

The CT2 test uses `~/models/anime-whisper-ct2` on `cuda` / `int8_float16` by default
(`WT_TEST_CT2_MODEL`, `WT_TEST_CT2_DEVICE`, `WT_TEST_CT2_COMPUTE` override
them). Point `WT_TEST_CT2_MODEL` at another folder to check a new model. A
few tests need a POSIX shell or signals and skip on Windows, as does the Mac
(mlx) backend test.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cublas64_12.dll` / `libcublas.so.12` not found | CUDA 12 libraries missing or not on PATH (step 3). CUDA 13 alone isn't enough. |
| `open() got an unexpected keyword argument 'metadata_errors'` | PyAV too new: `pip install "av<16"` in the venv. |
| Transcript is gibberish or in random languages | The CT2 model folder has no `tokenizer.json` (step 5). |
| Few or no subtitles, some windows garbled (RTX 50-series) | `--compute_type float16` on Blackwell; use `int8_float16` with CTranslate2 ≥ 4.8.1 (step 6). |
| Subtitles are 30 seconds long, or the job errors in `model.align` | VAD segmentation is off for a model without timestamps (anime-whisper). |
| `ModuleNotFoundError: faster_whisper` in the log | The VAD script is running with the wrong Python; set **Python (VAD script)**. |
| `llama-server didn't finish loading within 15s` | The model is still loading (it keeps going in the background): press **Retry**, or raise **Model load wait** in Settings. |
| `llama-server exited … while loading` | The log shows llama-server's last lines: usually a wrong `.gguf` path, or not enough VRAM (pick a smaller quant or lower `-c`). |
| Translation is slow, or GPU memory barely rises | llama-server is a CPU build; `llama-server --list-devices` must show the RTX 5090 (step 7). |
| Item errors with `HTTP 401` / `402` | OpenRouter key missing or out of credit. |
| Lines left in Japanese in `.en.srt` | The model dropped or refused them three times; try another model, then **EN↻**. |

## Notes

- To stop the app, use the gear icon (top left) → **Shut down server**. It
  stops any running job and frees the port. Port 3939 can be changed in
  Settings (applies on reboot).
- `config.json` is per-machine and gitignored, so the PC and the Mac keep
  separate backends, models, paths and translation endpoints.
- The Mac equivalent of this guide is [`APPLESILICON_SETUP.md`](APPLESILICON_SETUP.md).
- The **How to use** panel at the top of the app (under the tabs) shows a condensed version
  of these steps for both machines.

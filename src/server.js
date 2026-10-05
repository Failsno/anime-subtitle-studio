const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const chokidar = require('chokidar');
const { spawn, execFileSync } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

// ─── Config ────────────────────────────────────────────────────────────────
// Optional .env (e.g. OPENROUTER_API_KEY); real environment variables win
try { process.loadEnvFile(path.join(__dirname, '..', '.env')); } catch {}

const CONFIG_PATH = process.env.WT_CONFIG || path.join(__dirname, '..', 'config.json');

const DEFAULT_WHISPER_ARGS = {
  model: 'large-v3',
  device: 'cuda',
  // int8_float16, not float16: float16 garbles output on RTX 50-series (Blackwell)
  // (no subtitles at all on some cards), and int8 works there again since
  // CTranslate2 4.8.1
  compute_type: 'int8_float16',
  output_format: 'srt',
  language: 'ja',
  task: 'translate',
  vad_filter: 'True',
  // Speech detection: tuned "balanced" Silero values. They catch short and
  // quiet lines (sighs, gasps, whispered asides) the old 0.3 / 200 / 1000 / 200
  // settings skipped, and give shorter, tighter subtitles
  vad_threshold: '0.28',
  vad_min_speech_duration_ms: '100',
  vad_min_silence_duration_ms: '300',
  suppress_blank: 'True',
  beam_size: '4',
  temperature: '0.0',
  no_speech_threshold: '0.4',
  word_timestamps: 'True',
  initial_prompt: '',
  // Off by default on CUDA: turn on for models without timestamps (anime-whisper)
  vad_segmentation: 'False',
  vad_max_segment_s: '15',
  // VAD segmentation only (scripts/vad_transcribe.py)
  vad_speech_pad_ms: '400',
  repetition_penalty: '1.0', // > 1 discourages "ああああ…" loops
  no_repeat_ngram_size: '0'  // 0 = off
};

// mlx-whisper (Apple Silicon / MLX GPU backend) has a different CLI surface
// than whisper-ctranslate2 — no device/compute_type/VAD flags, models are
// referenced as Hugging Face repo ids (or local folders) rather than size names.
// vad_segmentation runs scripts/vad_transcribe.py instead of the CLI.
const DEFAULT_MLX_ARGS = {
  model: 'mlx-community/whisper-large-v3-mlx',
  language: 'ja',
  task: 'translate',
  temperature: '0.0',
  no_speech_threshold: '0.4',
  word_timestamps: 'True',
  fp16: 'True',
  initial_prompt: '',
  vad_segmentation: 'True',
  // Same Silero detector as the PC, so the same values
  vad_threshold: '0.28',
  vad_min_speech_duration_ms: '100',
  vad_min_silence_duration_ms: '300',
  vad_max_segment_s: '15',
  vad_speech_pad_ms: '400'
};

const VAD_SCRIPT = path.join(__dirname, '..', 'scripts', 'vad_transcribe.py');

// Second stage: whisper transcribes Japanese, then an OpenAI-compatible chat
// endpoint translates the subtitle file. Each machine keeps its own endpoint
// in its own config.json — OpenRouter on the PC, local llama-server on the Mac.
const DEFAULT_STYLE_PROMPT = [
  'You translate Japanese anime dialogue into English subtitles.',
  "Write natural, colloquial English that fits each character's voice and personality.",
  'Translate faithfully: do not summarize, soften or add commentary.',
  'Keep lines short enough to read as subtitles.',
  'Keep honorifics (-san, -kun, -chan, -senpai, -sensei) and names in their usual romanized form.',
  'Render interjections and reactions briefly (e.g. "Eh?!", "Huh?", "Ah!") rather than describing them.',
  'Use the surrounding lines to keep speakers, pronouns and tone consistent.'
].join(' ');

function defaultTranslation() {
  return {
    enabled: false,
    baseUrl: process.platform === 'darwin' ? 'http://127.0.0.1:9931/v1' : 'https://openrouter.ai/api/v1',
    apiKey: '',
    model: '',
    batchSize: '30',
    contextLines: '5',
    temperature: '0.3',
    stylePrompt: '',
    englishOnly: true, // delete video.ja.srt once video.en.srt is written
    // Auto-start a local llama-server (blank executable = don't manage it)
    llamaExecutable: '',
    llamaModel: '',
    llamaArgs: '-ngl 99 -c 16384 --jinja --reasoning-budget 0',
    llamaLoadWait: '15' // seconds to wait for the model to load
  };
}

function defaultWatchFolder() {
  if (process.platform === 'win32') return 'C:\\Videos\\Japanese';
  return path.join(os.homedir(), 'Videos', 'Japanese');
}

function loadConfig() {
  if (fs.existsSync(CONFIG_PATH)) {
    try {
      const cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
      // Back-compat: configs written before the mlx backend existed have
      // neither field — treat them as the CUDA/CTranslate2 setup they always were.
      if (!cfg.backend) cfg.backend = 'ctranslate2';
      cfg.whisperArgs = { ...DEFAULT_WHISPER_ARGS, ...cfg.whisperArgs };
      // Merge so settings added later (e.g. vad_*) get their defaults
      cfg.mlxArgs = { ...DEFAULT_MLX_ARGS, ...cfg.mlxArgs };
      cfg.translation = { ...defaultTranslation(), ...cfg.translation };
      return cfg;
    } catch (e) {
      console.warn('config.json is malformed, falling back to defaults:', e.message);
    }
  }
  // Fresh install: default to the backend that actually works on this OS.
  // CTranslate2 has no Metal/MPS support, so macOS gets mlx-whisper instead.
  const backend = process.platform === 'darwin' ? 'mlx' : 'ctranslate2';
  const defaults = {
    watchFolder: defaultWatchFolder(),
    backend,
    whisperExecutable: 'whisper-ctranslate2',
    mlxExecutable: 'mlx_whisper',
    whisperArgs: { ...DEFAULT_WHISPER_ARGS },
    mlxArgs: { ...DEFAULT_MLX_ARGS },
    translation: defaultTranslation()
  };
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(defaults, null, 2));
  return defaults;
}

function saveConfig(cfg) {
  try {
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2));
  } catch (e) {
    console.error('Failed to save config.json:', e.message);
  }
}

// Config as sent to the browser: the API key never leaves the server, the UI
// only learns whether one is set (a blank field on save keeps the old key).
function publicConfig() {
  const { apiKey, ...translation } = config.translation;
  return {
    watchFolder: config.watchFolder,
    backend: config.backend,
    whisperExecutable: config.whisperExecutable,
    mlxExecutable: config.mlxExecutable,
    pythonExecutable: config.pythonExecutable || '',
    autoStart: !!config.autoStart,
    port: parseInt(config.port, 10) || null, // configured; null = default
    runningPort: PORT,
    nextPort: nextPort(),
    whisperArgs: config.whisperArgs,
    mlxArgs: config.mlxArgs,
    translation: { ...translation, apiKeySet: !!apiKey, defaultStylePrompt: DEFAULT_STYLE_PROMPT }
  };
}

// ─── State ──────────────────────────────────────────────────────────────────
let config = loadConfig();

// queue items: { id, filePath, outputDir, status: 'queued'|'running'|'done'|'error', progress: 0-100, segment, totalSegments, log }
let queue = [];
let isRunning = false;
let isPaused = false;
let currentProcess = null;
let watcher = null;
let nextId = 1;

function makeItem(filePath) {
  return {
    id: nextId++,
    filePath,
    outputDir: path.dirname(filePath),
    status: 'queued',
    progress: 0,
    segment: 0,
    totalSegments: 0,
    stage: null, // 'transcribe' | 'translate' while running
    startedAt: null, stageStartedAt: null, finishedAt: null, // ms timestamps for the UI
    log: []
  };
}

// Back to queued, as Retry / Re-translate / Retry errors do
function resetItem(item) {
  Object.assign(item, {
    status: 'queued', progress: 0, segment: 0, totalSegments: 0,
    startedAt: null, stageStartedAt: null, finishedAt: null, log: []
  });
}

function setStage(item, stage) {
  item.stage = stage;
  item.stageStartedAt = Date.now();
}

// Last error-looking log line of a failed item, shown under its name
function lastError(item) {
  if (item.status !== 'error') return undefined;
  const errs = item.log.filter(l => /error|exception|traceback|failed|cannot|not found/i.test(l));
  return (errs.length ? errs[errs.length - 1] : item.log[item.log.length - 1]) || 'failed (no log output)';
}

function queueHas(filePath) {
  return queue.some(i => i.filePath === filePath);
}

// Subtitle files written next to each video. Single-stage (whisper's own
// translate task) writes video.srt; two-stage writes video.ja.srt (whisper
// transcript) and video.en.srt (LLM translation).
function subtitlePaths(filePath) {
  const base = path.join(path.dirname(filePath), path.basename(filePath, path.extname(filePath)));
  return { single: base + '.srt', ja: base + '.ja.srt', en: base + '.en.srt' };
}

function isDone(filePath) {
  const p = subtitlePaths(filePath);
  return fs.existsSync(p.single) || fs.existsSync(p.en);
}

function itemSnapshot(i) {
  return {
    id: i.id, filePath: i.filePath, name: path.basename(i.filePath),
    outputDir: i.outputDir, status: i.status, progress: i.progress,
    segment: i.segment, totalSegments: i.totalSegments, stage: i.stage,
    hasJa: fs.existsSync(subtitlePaths(i.filePath).ja),
    startedAt: i.startedAt, stageStartedAt: i.stageStartedAt, finishedAt: i.finishedAt,
    lastError: lastError(i)
  };
}

// ─── WebSocket broadcast ────────────────────────────────────────────────────
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

wss.on('connection', ws => {
  // Send current state immediately so a reconnecting client is up to date
  ws.send(JSON.stringify({
    type: 'state',
    isRunning,
    isPaused,
    queue: queue.map(itemSnapshot)
  }));
  ws.send(JSON.stringify({ type: 'concat', job: concatSnapshot() }));
});

app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public'), { etag: false, maxAge: 0 }));

function broadcast(msg) {
  const data = JSON.stringify(msg);
  wss.clients.forEach(c => { if (c.readyState === WebSocket.OPEN) c.send(data); });
}

function broadcastState() {
  broadcast({
    type: 'state',
    isRunning,
    isPaused,
    queue: queue.map(itemSnapshot)
  });
}

// ─── File watcher ────────────────────────────────────────────────────────────
function startWatcher(folder) {
  if (watcher) watcher.close();
  if (!fs.existsSync(folder)) {
    console.warn(`Watch folder does not exist: ${folder}`);
    return;
  }
  watcher = chokidar.watch(folder, {
    persistent: true,
    ignoreInitial: false,
    depth: 10,
    ignored: /(^|[/\\])(System Volume Information|\$Recycle\.Bin|\.git)[/\\]/,
    awaitWriteFinish: { stabilityThreshold: 2000, pollInterval: 500 }
  });
  watcher.on('error', err => console.warn('Watcher error (ignored):', err.message));
  // Files reported before 'ready' are the startup scan; only later ones auto-start
  let scanned = false;
  watcher.on('ready', () => { scanned = true; });
  watcher.on('add', filePath => {
    if (path.extname(filePath).toLowerCase() === '.mp4' && !queueHas(filePath)) {
      const item = makeItem(filePath);
      if (isDone(filePath)) {
        item.status = 'done';
        item.progress = 100;
      }
      queue.push(item);
      // Not while paused: that's the user's call, and Resume picks it up
      if (scanned && config.autoStart && item.status === 'queued' && !isPaused) startQueue();
      broadcastState();
    }
  });
  watcher.on('unlink', filePath => {
    queue = queue.filter(i => i.filePath !== filePath || i.status === 'running');
    broadcastState();
  });
}

startWatcher(config.watchFolder);

// ─── Whisper runner ──────────────────────────────────────────────────────────
// Stock whisper names the CLIs can download themselves; anything else is a
// local model folder (e.g. ~/models/anime-whisper-ct2).
const STOCK_MODELS = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en',
  'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo', 'turbo',
  'distil-large-v2', 'distil-large-v3', 'distil-large-v3.5', 'distil-medium.en', 'distil-small.en'];

function expandHome(p) {
  return /^~(?=$|[\\/])/.test(p) ? path.join(os.homedir(), p.slice(1)) : p;
}

function isTrue(v) {
  return String(v).toLowerCase() === 'true';
}

// CTranslate2 refuses float16-based compute types on CPU instead of falling
// back, so the GPU default (int8_float16) must not break --device cpu
function ct2ComputeType(w) {
  if (String(w.device).toLowerCase() !== 'cpu') return w.compute_type;
  return { int8_float16: 'int8', int8_bfloat16: 'int8', float16: 'float32', bfloat16: 'float32' }[w.compute_type] || w.compute_type;
}

function buildCtranslate2Args(item, outputDir, task) {
  const w = config.whisperArgs;
  // whisper-ctranslate2's --model only accepts stock names; custom models go
  // through --model_directory
  const model = STOCK_MODELS.includes(w.model)
    ? ['--model', w.model]
    : ['--model_directory', expandHome(w.model)];
  const args = [
    item.filePath,
    '--output_dir', outputDir,
    ...model,
    '--device', w.device,
    '--compute_type', ct2ComputeType(w),
    '--output_format', w.output_format,
    '--language', w.language,
    '--task', task,
    '--vad_filter', w.vad_filter,
    '--vad_threshold', String(w.vad_threshold),
    '--vad_min_speech_duration_ms', String(w.vad_min_speech_duration_ms),
    '--vad_min_silence_duration_ms', String(w.vad_min_silence_duration_ms),
    '--suppress_blank', w.suppress_blank,
    '--beam_size', String(w.beam_size),
    '--temperature', String(w.temperature),
    '--no_speech_threshold', String(w.no_speech_threshold),
    '--word_timestamps', w.word_timestamps,
  ];
  if (w.initial_prompt) args.push('--initial_prompt', w.initial_prompt);
  return args;
}

// mlx-whisper's CLI (Apple Silicon / Metal GPU backend) uses hyphenated flags
// and has no --device/--compute_type/--vad_* options — it always runs on the
// GPU via MLX, and doesn't have a built-in VAD filter.
function buildMlxArgs(item, outputDir, task) {
  const w = config.mlxArgs;
  const args = [
    item.filePath,
    '--output-dir', outputDir,
    '--output-format', 'srt',
    '--model', expandHome(w.model),
    '--language', w.language,
    '--task', task,
    '--temperature', String(w.temperature),
    '--no-speech-threshold', String(w.no_speech_threshold),
    '--word-timestamps', w.word_timestamps,
    '--fp16', w.fp16,
    '--verbose', 'True',
  ];
  if (w.initial_prompt) args.push('--initial-prompt', w.initial_prompt);
  return args;
}

// VAD segmentation (scripts/vad_transcribe.py): Silero VAD finds speech, each
// stretch is transcribed alone, and the VAD times become the cue timings.
// Needed for models without timestamp tokens (anime-whisper), and cuts
// hallucinations on non-speech for any model.
function buildVadArgs(item, outputDir, task) {
  const mlx = config.backend === 'mlx';
  const w = mlx ? config.mlxArgs : config.whisperArgs;
  const args = [
    VAD_SCRIPT,
    item.filePath,
    '--engine', mlx ? 'mlx' : 'ct2',
    '--output-dir', outputDir,
    '--model', expandHome(w.model),
    '--language', w.language,
    '--task', task,
    '--temperature', String(w.temperature),
    '--vad-threshold', String(w.vad_threshold),
    '--vad-min-speech-ms', String(w.vad_min_speech_duration_ms),
    '--vad-min-silence-ms', String(w.vad_min_silence_duration_ms),
    '--vad-max-segment-s', String(w.vad_max_segment_s),
    '--vad-speech-pad-ms', String(w.vad_speech_pad_ms || 400),
  ];
  if (mlx) args.push('--fp16', w.fp16);
  else args.push('--device', w.device, '--compute-type', ct2ComputeType(w), '--beam-size', String(w.beam_size),
    // faster-whisper only; mlx-whisper has no equivalent
    '--repetition-penalty', String(w.repetition_penalty || 1.0),
    '--no-repeat-ngram-size', String(w.no_repeat_ngram_size || 0));
  if (w.initial_prompt) args.push('--initial-prompt', w.initial_prompt);
  return args;
}

// The VAD script must run in the backend's Python environment. Use the
// configured pythonExecutable, else the python next to the backend executable
// (venv bin/ or Scripts\, or the Python install dir above Scripts\ on Windows).
function findPython(backendExe) {
  if (config.pythonExecutable) return expandHome(config.pythonExecutable);
  const exe = expandHome(backendExe);
  if (path.isAbsolute(exe)) {
    const dir = path.dirname(exe);
    const candidates = process.platform === 'win32'
      ? [path.join(dir, 'python.exe'), path.join(dir, '..', 'python.exe')]
      : [path.join(dir, 'python3'), path.join(dir, 'python')];
    const found = candidates.find(c => fs.existsSync(c));
    if (found) return found;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

function getRunConfig(item, outputDir, task) {
  const mlx = config.backend === 'mlx';
  const w = mlx ? config.mlxArgs : config.whisperArgs;
  const backendExe = mlx ? (config.mlxExecutable || 'mlx_whisper') : (config.whisperExecutable || 'whisper-ctranslate2');
  task = task || w.task;
  if (isTrue(w.vad_segmentation)) {
    return { executable: findPython(backendExe), args: buildVadArgs(item, outputDir, task) };
  }
  return {
    executable: expandHome(backendExe),
    args: mlx ? buildMlxArgs(item, outputDir, task) : buildCtranslate2Args(item, outputDir, task)
  };
}

// Parse progress from whisper-ctranslate2 stderr/stdout
// It emits lines like: "Detected language: Japanese" and segment timestamps like "[00:00.000 --> 00:04.320]"
// We count segment lines to approximate progress. Total segments are estimated from duration via ffprobe if available.
function parseProgress(line, item) {
  // VAD script announces the exact number of segments up front
  const vadMatch = line.match(/^VAD: (\d+) speech segments/);
  if (vadMatch) {
    item.totalSegments = parseInt(vadMatch[1], 10);
    return true;
  }
  // Segment timestamp pattern: [HH:MM:SS.mmm --> HH:MM:SS.mmm] or [MM:SS.mmm --> MM:SS.mmm]
  const segMatch = line.match(/^\[[\d:\.]+\s*-->\s*([\d:\.]+)\]/);
  if (segMatch) {
    item.segment++;
    // If we know total segments, compute progress
    if (item.totalSegments > 0) {
      item.progress = Math.min(99, Math.round((item.segment / item.totalSegments) * 100));
    } else {
      // Fall back: just increment, cap at 95 until done
      item.progress = Math.min(95, item.segment);
    }
    return true;
  }
  // whisper-ctranslate2 also prints percentage in some builds: "  x%|..."
  const pctMatch = line.match(/^\s*(\d+)%\|/);
  if (pctMatch) {
    item.progress = Math.min(99, parseInt(pctMatch[1], 10));
    return true;
  }
  return false;
}

async function getDurationSeconds(filePath) {
  return new Promise(resolve => {
    const ff = spawn('ffprobe', [
      '-v', 'error', '-select_streams', 'v:0',
      '-show_entries', 'format=duration',
      '-of', 'default=noprint_wrappers=1:nokey=1',
      filePath
    ]);
    let out = '';
    const timer = setTimeout(() => { ff.kill(); resolve(0); }, 5000);
    ff.stdout.on('data', d => out += d);
    ff.on('close', () => {
      clearTimeout(timer);
      const secs = parseFloat(out.trim());
      resolve(isNaN(secs) ? 0 : secs);
    });
    ff.on('error', () => { clearTimeout(timer); resolve(0); });
  });
}

function logItem(item, line) {
  item.log.push(line);
  if (item.log.length > 2000) item.log.shift();
  broadcast({ type: 'log', id: item.id, line });
}

function broadcastProgress(item) {
  broadcast({ type: 'progress', id: item.id, progress: item.progress, segment: item.segment, totalSegments: item.totalSegments, stage: item.stage });
}

// Runs the whisper CLI for one item, writing its .srt into outputDir.
// Resolves true if the .srt was written, false otherwise (crash, Stop, delete).
function runWhisper(item, outputDir, task) {
  return new Promise(resolve => {
    // Estimate total segments (approx 1 segment per 5 seconds of audio)
    getDurationSeconds(item.filePath).then(secs => {
      if (secs > 0 && !item.totalSegments) item.totalSegments = Math.ceil(secs / 5);
    });

    const { executable, args } = getRunConfig(item, outputDir, task);
    console.log('Running:', executable, args.join(' '));

    // A .js "executable" (stand-ins for demos/tests) runs with this Node, since
    // Windows can't spawn scripts directly
    const [cmd, cmdArgs] = /\.[cm]?js$/i.test(executable)
      ? [process.execPath, [executable, ...args]] : [executable, args];
    currentProcess = spawn(cmd, cmdArgs, {
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' }
    });

    let lastProgressBroadcast = 0;

    function handleLine(line) {
      line = line.trim();
      if (!line) return;
      const progressed = parseProgress(line, item);
      if (progressed) {
        const now = Date.now();
        if (now - lastProgressBroadcast >= 1000) {
          lastProgressBroadcast = now;
          broadcastProgress(item);
        }
      }
      // Always send to log so transcripts stream through visibly
      logItem(item, line);
    }

    let stdoutBuf = '';
    let stderrBuf = '';

    // Simulate terminal \r behaviour: split on \n for real newlines, then within
    // each segment keep only the text after the last \r (tqdm overwrites in place).
    function processChunk(buf, newData) {
      buf += newData;
      const lines = buf.split('\n');
      buf = lines.pop(); // keep incomplete last line
      lines.forEach(line => {
        // Honour carriage-return overwrite: only the content after the last \r matters.
        // Drop the \r of a Windows \r\n first, or every line would come out empty.
        const parts = line.replace(/\r$/, '').split('\r');
        const visible = parts[parts.length - 1];
        handleLine(visible);
      });
      return buf;
    }

    // Decode as streams so a Japanese character split across chunks isn't garbled
    currentProcess.stdout.setEncoding('utf8');
    currentProcess.stderr.setEncoding('utf8');

    currentProcess.stdout.on('data', data => {
      stdoutBuf = processChunk(stdoutBuf, data);
    });

    currentProcess.stderr.on('data', data => {
      stderrBuf = processChunk(stderrBuf, data);
    });

    let finished = false;

    // Safety timeout: kill the process if it runs longer than 4 hours
    const MAX_RUNTIME_MS = 4 * 60 * 60 * 1000;
    const runtimeTimeout = setTimeout(() => {
      if (!finished && currentProcess) {
        console.warn(`Job timed out after 4 hours: ${item.filePath}`);
        currentProcess.kill('SIGTERM');
      }
    }, MAX_RUNTIME_MS);

    currentProcess.on('close', code => {
      clearTimeout(runtimeTimeout);
      if (stdoutBuf.trim()) handleLine(stdoutBuf);
      if (stderrBuf.trim()) handleLine(stderrBuf);
      currentProcess = null;
      if (finished) return;
      finished = true;
      // Success means "the SRT exists", not the exit code: whisper-ctranslate2
      // sometimes exits non-zero after writing it (Unicode print errors), and
      // exits 0 after swallowing a crash without writing anything.
      resolve(findSrt(outputDir, item.filePath) !== null);
    });

    currentProcess.on('error', err => {
      clearTimeout(runtimeTimeout);
      logItem(item, `ERROR: ${err.message}`);
      currentProcess = null;
      if (finished) return;
      finished = true;
      resolve(false);
    });
  });
}

function findSrt(dir, filePath) {
  const p = path.join(dir, path.basename(filePath, path.extname(filePath)) + '.srt');
  return fs.existsSync(p) ? p : null;
}

// ─── Translation (OpenAI-compatible chat endpoint) ──────────────────────────
// Works against OpenRouter and llama.cpp's llama-server alike.
let currentAbort = null; // AbortController for the in-flight translation

function parseSrt(text) {
  return text.replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').trim().split(/\n\s*\n/).map(block => {
    const lines = block.split('\n');
    const t = lines.findIndex(l => l.includes('-->'));
    if (t < 0) return null;
    return { time: lines[t].trim(), text: lines.slice(t + 1).join(' ').trim() };
  }).filter(c => c && c.text);
}

function formatSrt(cues) {
  return cues.map((c, i) => `${i + 1}\n${c.time}\n${c.text}\n`).join('\n');
}

const FORMAT_RULES = [
  'Input lines look like "[id] Japanese text". Translate ONLY the lines under "Translate these lines".',
  'Reply with exactly one line per id, in the form "[id] English text", in the same order.',
  'Never merge, split or skip ids. Output nothing else: no notes, no headings, no code fences.'
].join(' ');

function buildTranslationMessages(cues, ids, done, contextLines) {
  const first = ids[0];
  const last = ids[ids.length - 1];
  const parts = [];
  const before = [];
  for (let i = Math.max(0, first - contextLines); i < first; i++) {
    if (done[i] !== undefined) before.push(`[${i + 1}] ${cues[i].text} => ${done[i]}`);
  }
  if (before.length) parts.push('Previous lines (context only, already translated; do not output):\n' + before.join('\n'));
  parts.push('Translate these lines:\n' + ids.map(i => `[${i + 1}] ${cues[i].text}`).join('\n'));
  const after = cues.slice(last + 1, last + 1 + contextLines).map((c, k) => `[${last + 2 + k}] ${c.text}`);
  if (after.length) parts.push('Upcoming lines (context only; do not output):\n' + after.join('\n'));

  const t = config.translation;
  return [
    { role: 'system', content: `${t.stylePrompt || DEFAULT_STYLE_PROMPT}\n\n${FORMAT_RULES}` },
    { role: 'user', content: parts.join('\n\n') }
  ];
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function chatCompletion(messages, signal) {
  const t = config.translation;
  const apiKey = t.apiKey || process.env.OPENROUTER_API_KEY || '';
  const url = t.baseUrl.replace(/\/+$/, '') + '/chat/completions';
  const body = { messages, temperature: parseFloat(t.temperature) || 0 };
  if (t.model) body.model = t.model;

  for (let attempt = 0; ; attempt++) {
    // Per-request controller: aborted by Stop/delete or a 10 minute timeout
    const req = new AbortController();
    const onAbort = () => req.abort();
    signal.addEventListener('abort', onAbort);
    const timer = setTimeout(onAbort, 10 * 60 * 1000);
    let res;
    try {
      res = await fetchJson(url, apiKey, body, req.signal);
    } finally {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    }
    if (res.ok) return res.content;
    // Rate limits and server hiccups are worth waiting out; anything else
    // (bad key, unknown model) won't fix itself.
    if ((res.status === 429 || res.status >= 500) && attempt < 4) {
      await sleep(2000 * 2 ** attempt);
      continue;
    }
    throw new Error(`HTTP ${res.status} from ${url}: ${res.error}`);
  }
}

async function fetchJson(url, apiKey, body, signal) {
  const res = await fetch(url, {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      'X-Title': 'Subtitle Studio'
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) return { ok: false, status: res.status, error: (await res.text()).slice(0, 300) };
  const data = await res.json();
  const msg = data.choices && data.choices[0] && data.choices[0].message;
  // Reasoning models may wrap their thinking in <think> tags inside content
  return { ok: true, content: String((msg && msg.content) || '').replace(/<think>[\s\S]*?<\/think>/g, '') };
}

async function translateSrt(item, jaPath, enPath) {
  const t = config.translation;
  if (!t.baseUrl) throw new Error('Translation base URL is not set (Settings → Translation)');
  const cues = parseSrt(fs.readFileSync(jaPath, 'utf8'));
  const batchSize = Math.max(1, parseInt(t.batchSize, 10) || 30);
  const contextLines = Math.max(0, parseInt(t.contextLines, 10) || 0);
  const done = new Array(cues.length);

  setStage(item, 'translate');
  item.segment = 0;
  item.totalSegments = cues.length;
  item.progress = 0;
  broadcastState();
  logItem(item, `Translating ${cues.length} lines via ${t.baseUrl}${t.model ? ` (${t.model})` : ''}`);

  const abort = new AbortController();
  currentAbort = abort;
  try {
    for (let start = 0; start < cues.length; start += batchSize) {
      let pending = [];
      for (let i = start; i < Math.min(start + batchSize, cues.length); i++) pending.push(i);

      // Ask again for any ids the model dropped; after 3 tries, keep the
      // Japanese line so the subtitle timing stays intact.
      for (let attempt = 1; attempt <= 3 && pending.length; attempt++) {
        const reply = await chatCompletion(buildTranslationMessages(cues, pending, done, contextLines), abort.signal);
        for (const line of reply.split('\n')) {
          const m = line.match(/^\s*\[(\d+)\]\s*(.*\S)\s*$/);
          if (!m) continue;
          const i = parseInt(m[1], 10) - 1;
          if (pending.includes(i) && done[i] === undefined) {
            done[i] = m[2];
            logItem(item, `[${i + 1}] ${cues[i].text}  →  ${m[2]}`);
          }
        }
        pending = pending.filter(i => done[i] === undefined);
        if (pending.length) logItem(item, `WARNING: ${pending.length} line(s) missing from reply (attempt ${attempt}/3)${reply.trim() ? '' : ': empty reply'}`);
      }
      pending.forEach(i => { done[i] = cues[i].text; });

      item.segment = Math.min(start + batchSize, cues.length);
      item.progress = Math.min(99, Math.round((item.segment / cues.length) * 100));
      broadcastProgress(item);
    }
  } finally {
    if (currentAbort === abort) currentAbort = null;
  }

  const tmp = enPath + '.part';
  fs.writeFileSync(tmp, formatSrt(cues.map((c, i) => ({ time: c.time, text: done[i] }))), 'utf8');
  fs.renameSync(tmp, enPath);
}

// ─── Local llama-server ──────────────────────────────────────────────────────
// With translation.llamaExecutable set and a local base URL, a translation
// starts llama-server if nothing answers there, and Shut down stops it.
// Detached so a Restart leaves the model loaded (on Windows a normal child dies
// with us), with output in a file because a pipe would break when we exit.
let llamaProc = null;

// host/port of the translation endpoint when it's on this machine, else null
function localEndpoint() {
  let url;
  try { url = new URL(config.translation.baseUrl); } catch (e) { return null; }
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) return null;
  const host = url.hostname === 'localhost' ? '127.0.0.1' : url.hostname.replace(/^\[|\]$/g, '');
  const port = parseInt(url.port, 10) || (url.protocol === 'https:' ? 443 : 80);
  const origin = `${url.protocol}//${url.host}`;
  return { host, port, origin, health: `${origin}/health` };
}

// The endpoint, if the app should start/stop llama-server there
function llamaTarget() {
  const t = config.translation;
  return t.enabled && t.llamaExecutable ? localEndpoint() : null;
}

const llamaLogPath = port => path.join(os.tmpdir(), `whisper-translate-llama-${port}.log`);

// GET a JSON path on the endpoint: { status, body } or null if nothing answers
async function llamaGet(ep, p) {
  try {
    const res = await fetch(ep.origin + p, { signal: AbortSignal.timeout(2000) });
    return { status: res.status, body: await res.json().catch(() => null) };
  } catch (e) {
    return null;
  }
}

// The .gguf llama-server has loaded (from /props), or '' if it doesn't say
async function llamaLoadedModel(ep) {
  const props = await llamaGet(ep, '/props');
  return (props && props.body && props.body.model_path) || '';
}

const samePath = (a, b) => {
  const norm = p => path.resolve(expandHome(p));
  return process.platform === 'win32' ? norm(a).toLowerCase() === norm(b).toLowerCase() : norm(a) === norm(b);
};

async function llamaHealthy(target) {
  try {
    return (await fetch(target.health, { signal: AbortSignal.timeout(2000) })).ok;
  } catch (e) {
    return false;
  }
}

// Split "a \"b c\" d" the way a shell would, minus escapes
const splitArgs = s => (s.match(/"[^"]*"|\S+/g) || []).map(a => a.replace(/^"(.*)"$/, '$1'));

// The command llama-server would be started with for this endpoint
function llamaCommand(target) {
  const t = config.translation;
  const args = [...splitArgs(t.llamaArgs || '')];
  if (t.llamaModel) args.push('-m', expandHome(t.llamaModel));
  args.push('--host', target.host, '--port', String(target.port));
  return { exe: expandHome(t.llamaExecutable), args };
}
const commandKey = c => JSON.stringify([c.exe, c.args]);

function startLlama(item, target) {
  const cmd = llamaCommand(target);
  logItem(item, `Starting llama-server: ${cmd.exe} ${cmd.args.join(' ')}`);
  const out = fs.openSync(llamaLogPath(target.port), 'w');
  try {
    llamaProc = spawn(cmd.exe, cmd.args, { detached: true, windowsHide: true, stdio: ['ignore', out, out] });
  } finally {
    fs.closeSync(out);
  }
  const proc = llamaProc;
  proc.key = commandKey(cmd); // to notice when Settings change the command
  proc.port = target.port;
  // A failed spawn only emits 'error' (no 'exit'), so forget it here too, or
  // a corrected path in Settings would never be tried
  proc.on('error', e => { proc.spawnError = e; if (llamaProc === proc) llamaProc = null; });
  proc.on('exit', () => { if (llamaProc === proc) llamaProc = null; });
  proc.unref();
}

// Kill the llama-server this app started (if any); returns the port it served
function stopLlamaChild() {
  if (!llamaProc) return null;
  const port = llamaProc.port;
  try { process.kill(llamaProc.pid); } catch (e) { /* already gone */ }
  llamaProc = null;
  return port;
}

function llamaLogTail(port, n = 8) {
  try {
    return fs.readFileSync(llamaLogPath(port), 'utf8').trim().split(/\r?\n/).slice(-n);
  } catch (e) {
    return [];
  }
}

// Make sure the translation endpoint is up, starting llama-server if needed.
// Waits at most llamaLoadWait seconds; a model still loading after that keeps
// loading, so a later Retry can use it.
async function ensureLlama(item) {
  const target = llamaTarget();
  if (!target) return;
  const health = await llamaGet(target, '/health');
  // Something answers but isn't ready: llama-server still loading (503)
  let answering = !!health;
  if (health && health.status === 200) {
    // Settings may name another model than the loaded one: reload between videos
    const loaded = await llamaLoadedModel(target);
    const wanted = config.translation.llamaModel;
    if (!wanted || !loaded || samePath(loaded, wanted)) return;
    logItem(item, `Switching llama-server model: ${path.basename(loaded)} → ${path.basename(wanted)}`);
    await killLlama(target);
    answering = false;
  }
  // Our llama-server was started with other settings (port, executable, args):
  // replace it rather than wait on an endpoint it doesn't serve
  if (llamaProc && llamaProc.key !== commandKey(llamaCommand(target))) {
    logItem(item, 'llama-server settings changed; restarting it');
    if (stopLlamaChild() === target.port) {
      for (let i = 0; i < 40 && await llamaGet(target, '/health'); i++) await sleep(250);
      answering = false;
    }
  }
  // A llama-server that is still loading is waited for, not started twice
  if (!answering && !llamaProc) startLlama(item, target);
  const proc = llamaProc; // null when waiting on one started outside the app
  const waitS = Math.max(1, parseFloat(config.translation.llamaLoadWait) || 15);
  logItem(item, `Waiting up to ${waitS}s for llama-server to load the model...`);

  const abort = new AbortController();
  currentAbort = abort;
  try {
    const deadline = Date.now() + waitS * 1000;
    while (Date.now() < deadline) {
      await sleep(500);
      if (abort.signal.aborted) { const e = new Error('cancelled'); e.name = 'AbortError'; throw e; }
      if (proc && proc.spawnError) throw new Error(`could not start llama-server: ${proc.spawnError.message}`);
      if (proc && proc.exitCode !== null) {
        llamaLogTail(target.port).forEach(l => logItem(item, `llama-server: ${l}`));
        throw new Error(`llama-server exited (code ${proc.exitCode}) while loading; see the lines above`);
      }
      if (await llamaHealthy(target)) { logItem(item, 'llama-server is ready'); return; }
    }
  } finally {
    if (currentAbort === abort) currentAbort = null;
  }
  throw new Error(`llama-server didn't finish loading within ${waitS}s. It keeps loading in the background: press Retry, or raise Settings → Translation → Model load wait`);
}

// What the local endpoint is serving, for Settings. llama-server answers
// /health with 503 while loading; /props has the model path and context size.
async function llamaStatus() {
  const ep = localEndpoint();
  if (!ep) return { local: false };
  const getJson = p => llamaGet(ep, p);
  const health = await getJson('/health');
  const status = {
    local: true, url: ep.origin,
    autoStart: !!config.translation.llamaExecutable,
    startedByApp: !!llamaProc,
    state: !health ? (llamaProc ? 'starting' : 'stopped') : health.status === 200 ? 'ready' : health.status === 503 ? 'loading' : 'error'
  };
  if (status.state !== 'ready') return status;
  const [props, models] = await Promise.all([getJson('/props'), getJson('/v1/models')]);
  const p = props && props.body || {};
  const m = models && models.body && models.body.data && models.body.data[0];
  status.model = p.model_path ? path.basename(p.model_path) : (m && m.id) || '';
  status.ctx = (p.default_generation_settings && p.default_generation_settings.n_ctx) || p.n_ctx || null;
  // A model picked in Settings loads on the next translation (see ensureLlama)
  const wanted = config.translation.llamaModel;
  if (status.autoStart && wanted && p.model_path && !samePath(p.model_path, wanted)) status.nextModel = path.basename(wanted);
  return status;
}

// .gguf files a llama-server could load: the Hugging Face cache, ~/models and
// the selected model's folder. Skips vision projectors (mmproj), embedding
// models and all but the first part of split models (sizes are summed).
function listGgufModels() {
  const hfHub = process.env.HF_HUB_CACHE
    || path.join(process.env.HF_HOME || path.join(os.homedir(), '.cache', 'huggingface'), 'hub');
  const roots = [hfHub, path.join(os.homedir(), 'models')];
  if (config.translation.llamaModel) roots.push(path.dirname(expandHome(config.translation.llamaModel)));
  const found = new Map(); // real path → model
  const walk = (dir, depth) => {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (e) { return; }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory() && e.name !== 'blobs' && depth < 5) { walk(full, depth + 1); continue; }
      if (!/\.gguf$/i.test(e.name) || /^mmproj|embed/i.test(e.name)) continue;
      const split = e.name.match(/^(.*)-(\d{5})-of-(\d{5})\.gguf$/i);
      if (split && split[2] !== '00001') continue;
      let real, size;
      try {
        real = fs.realpathSync(full);
        size = split
          ? fs.readdirSync(dir).filter(n => n.startsWith(split[1] + '-') && n.endsWith(`-of-${split[3]}.gguf`))
              .reduce((sum, n) => sum + fs.statSync(path.join(dir, n)).size, 0)
          : fs.statSync(full).size;
      } catch (err) { continue; } // dangling cache symlink
      if (!found.has(real)) found.set(real, { name: e.name.replace(/\.gguf$/i, ''), path: full, size });
    }
  };
  roots.forEach(r => walk(r, 0));
  return [...found.values()].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
}

// PID listening on a TCP port, and its executable name (best effort)
function portOwner(port) {
  try {
    if (process.platform === 'win32') {
      const row = execFileSync('netstat', ['-ano', '-p', 'TCP'], { encoding: 'utf8', timeout: 5000 })
        .split(/\r?\n/).map(l => l.trim().split(/\s+/))
        .find(c => c[3] === 'LISTENING' && c[1].endsWith(`:${port}`));
      if (!row) return null;
      const pid = parseInt(row[4], 10);
      const csv = execFileSync('tasklist', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'], { encoding: 'utf8', timeout: 5000 });
      return { pid, name: (csv.match(/^"([^"]+)"/) || [])[1] || '' };
    }
    const pid = parseInt(execFileSync('lsof', ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN', '-t'], { encoding: 'utf8', timeout: 5000 }), 10);
    if (!pid) return null;
    const name = path.basename(execFileSync('ps', ['-p', String(pid), '-o', 'comm='], { encoding: 'utf8', timeout: 5000 }).trim());
    return { pid, name };
  } catch (e) {
    return null;
  }
}

// Stop the llama-server we started, or one already running on the endpoint's
// port, but only if it is the configured llama-server executable
function stopLlamaAt(target) {
  stopLlamaChild();
  const owner = portOwner(target.port);
  const exeName = n => path.basename(n).toLowerCase().replace(/\.exe$/, '');
  const wanted = [exeName(config.translation.llamaExecutable), 'llama-server'];
  if (owner && owner.pid !== process.pid && wanted.includes(exeName(owner.name))) {
    console.log(`  Stopping llama-server (pid ${owner.pid})`);
    try { process.kill(owner.pid); } catch (e) { /* already gone */ }
  }
}

// Shut down (synchronous: we're about to exit). The llama-server we started is
// stopped even if Settings no longer use it (single-stage, remote endpoint)
function stopLlama() {
  stopLlamaChild();
  const target = llamaTarget();
  if (target) stopLlamaAt(target);
}

// Model switch: stop it and wait until the port is free for the new one
async function killLlama(target) {
  stopLlamaAt(target);
  for (let i = 0; i < 40 && await llamaGet(target, '/health'); i++) await sleep(250);
}

// ─── Job runner ──────────────────────────────────────────────────────────────
function moveFile(from, to) {
  // copy + unlink rather than rename: the temp dir may be on another volume
  fs.copyFileSync(from, to);
  fs.unlinkSync(from);
}

async function processItem(item) {
  if (!config.translation.enabled) {
    // Single stage: whisper's own translate task writes video.srt in place
    setStage(item, 'transcribe');
    return runWhisper(item, item.outputDir);
  }

  // Two stage. An existing video.ja.srt means transcription already happened
  // (earlier run, or Re-translate), so go straight to translation.
  const subs = subtitlePaths(item.filePath);
  if (!fs.existsSync(subs.ja)) {
    setStage(item, 'transcribe');
    broadcastState();
    // Write to a temp dir so whisper's video.srt never lands next to the video
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'whisper-job-'));
    try {
      const ok = await runWhisper(item, tmpDir, 'transcribe');
      const srt = findSrt(tmpDir, item.filePath);
      if (!ok || !srt) return false;
      moveFile(srt, subs.ja);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  } else {
    logItem(item, `Using existing transcript: ${subs.ja}`);
  }
  if (!isRunning || !queue.includes(item)) return false; // stopped or removed meanwhile

  try {
    await ensureLlama(item);
    await translateSrt(item, subs.ja, subs.en);
    if (config.translation.englishOnly) {
      // One subtitle per video, named so players pick it up: video.srt.
      // Rename first: if it fails (video.srt locked by a player), the
      // transcript must still be there so Retry only translates.
      const replaced = fs.existsSync(subs.single);
      fs.renameSync(subs.en, subs.single); // replaces an older video.srt
      logItem(item, `Renamed ${path.basename(subs.en)} → ${path.basename(subs.single)}`
        + (replaced ? ' (replaced the existing one)' : ''));
      fs.rmSync(subs.ja, { force: true });
      logItem(item, `Deleted ${path.basename(subs.ja)} (Settings → Keep only English subtitles)`);
    }
    return true;
  } catch (e) {
    logItem(item, `ERROR: translation failed: ${e.name === 'AbortError' ? 'cancelled' : e.message}`);
    return false;
  }
}

function runNext() {
  if (isPaused || !isRunning) return;
  if (queue.some(i => i.status === 'running')) return; // already processing one

  const item = queue.find(i => i.status === 'queued');
  if (!item) {
    isRunning = false;
    broadcastState();
    return;
  }

  item.status = 'running';
  item.progress = 0;
  item.segment = 0;
  item.totalSegments = 0;
  item.startedAt = Date.now();
  item.finishedAt = null;
  broadcastState();

  processItem(item).catch(e => {
    logItem(item, `ERROR: ${e.message}`);
    return false;
  }).then(ok => {
    item.stage = null;
    item.status = ok ? 'done' : 'error';
    item.finishedAt = Date.now();
    if (ok) item.progress = 100;
    broadcastState();
    if (exitAfterJob) return shutdown();
    if (!isPaused && isRunning) setTimeout(runNext, 500);
  });
}

function stopCurrentJob() {
  if (currentProcess) {
    currentProcess.kill('SIGTERM');
    currentProcess = null;
  }
  if (currentAbort) currentAbort.abort();
}

// ─── Concat runner ───────────────────────────────────────────────────────────
// One concat job at a time, independent of the whisper queue. Uses ffmpeg's
// concat demuxer with stream copy (no re-encode), so inputs must share codecs.
const VIDEO_EXTS = ['.mp4', '.mkv', '.mov', '.m4v', '.ts', '.avi', '.webm', '.flv'];

// concatJob: { status: 'running'|'done'|'error'|'cancelled', files, outputPath, progress, error, log,
//              deleteOriginals, deleted, deleteErrors }
let concatJob = null;
let concatProcess = null;

function concatSnapshot() {
  if (!concatJob) return null;
  const { status, files, outputPath, progress, error, deleteOriginals, deleted, deleteErrors } = concatJob;
  return { status, files, outputPath, progress, error, deleteOriginals, deleted, deleteErrors };
}

// "Delete originals after concat": only called once ffmpeg succeeded and the
// output is a non-empty file. Skips a video the queue is transcribing right now.
function deleteConcatInputs(job) {
  job.deleted = [];
  job.deleteErrors = [];
  let size = 0;
  try { size = fs.statSync(job.outputPath).size; } catch (e) { /* missing */ }
  if (!size) { job.deleteErrors.push('output file is missing or empty, kept the originals'); return; }
  job.files.forEach(f => {
    if (queue.some(i => i.filePath === f && i.status === 'running')) {
      job.deleteErrors.push(`${path.basename(f)}: being transcribed, kept`);
      return;
    }
    try {
      fs.unlinkSync(f);
      job.deleted.push(f);
    } catch (e) {
      job.deleteErrors.push(`${path.basename(f)}: ${e.message}`);
    }
  });
}

function broadcastConcat() {
  broadcast({ type: 'concat', job: concatSnapshot() });
}

// Concat demuxer list-file quoting: wrap in single quotes, and a literal quote
// becomes '\'' (close quote, escaped quote, reopen quote).
function concatListLine(filePath) {
  const p = filePath.replace(/\\/g, '/').replace(/'/g, `'\\''`);
  return `file '${p}'`;
}

async function startConcat(files, outputPath, deleteOriginals = false) {
  const listPath = path.join(os.tmpdir(), `whisper-concat-${Date.now()}.txt`);
  fs.writeFileSync(listPath, files.map(concatListLine).join('\n') + '\n', 'utf8');

  concatJob = { status: 'running', files, outputPath, progress: 0, error: null, log: [], deleteOriginals };
  broadcastConcat();

  const durations = await Promise.all(files.map(getDurationSeconds));
  const totalUs = durations.reduce((a, b) => a + b, 0) * 1e6;
  const job = concatJob;
  if (job.cancelRequested) {
    // Cancelled while ffprobe was still measuring — ffmpeg never started
    try { fs.unlinkSync(listPath); } catch (e) { /* already gone */ }
    job.status = 'cancelled';
    broadcastConcat();
    return;
  }

  const args = [
    '-hide_banner', '-nostats', '-n',
    '-f', 'concat', '-safe', '0',
    '-i', listPath,
    '-c', 'copy',
    '-progress', 'pipe:1',
    outputPath
  ];
  console.log('Running: ffmpeg', args.join(' '));
  concatProcess = spawn('ffmpeg', args);

  let lastBroadcast = 0;
  let outBuf = '';
  concatProcess.stdout.on('data', data => {
    outBuf += data.toString();
    const lines = outBuf.split(/\r?\n/);
    outBuf = lines.pop();
    lines.forEach(line => {
      const m = line.match(/^out_time_(?:us|ms)=(\d+)/);
      if (m && totalUs > 0) {
        job.progress = Math.min(99, Math.round((parseInt(m[1], 10) / totalUs) * 100));
        const now = Date.now();
        if (now - lastBroadcast >= 500) { lastBroadcast = now; broadcastConcat(); }
      }
    });
  });
  concatProcess.stderr.on('data', data => {
    data.toString().split(/\r?\n/).forEach(l => {
      if (!l.trim()) return;
      job.log.push(l);
      if (job.log.length > 200) job.log.shift();
    });
  });

  const finish = (status, error) => {
    if (job.status !== 'running') return;
    concatProcess = null;
    try { fs.unlinkSync(listPath); } catch (e) { /* already gone */ }
    // Don't leave a truncated/unplayable file behind
    if (status !== 'done') { try { fs.unlinkSync(outputPath); } catch (e) { /* never created */ } }
    job.status = status;
    job.error = error || null;
    if (status === 'done') {
      job.progress = 100;
      if (job.deleteOriginals) deleteConcatInputs(job);
    }
    broadcastConcat();
  };

  concatProcess.on('close', code => {
    if (job.cancelRequested) return finish('cancelled');
    if (code === 0) return finish('done');
    finish('error', job.log.slice(-5).join('\n') || `ffmpeg exited with code ${code}`);
  });
  concatProcess.on('error', err => finish('error', `Could not start ffmpeg: ${err.message}`));
}

// ─── REST API ────────────────────────────────────────────────────────────────
app.get('/api/state', (req, res) => {
  res.json({
    isRunning, isPaused,
    config: publicConfig(),
    concat: concatSnapshot(),
    queue: queue.map(itemSnapshot)
  });
});

function startQueue() {
  if (shuttingDown) return;
  isPaused = false;
  isRunning = true;
  runNext(); // no-op if a job is already running
}

app.post('/api/start', (req, res) => {
  startQueue();
  broadcastState();
  res.json({ ok: true });
});

app.post('/api/pause', (req, res) => {
  // Only a running queue can pause; a stale pause would make Resume a no-op
  // and keep auto-start from picking up new videos
  if (isRunning) isPaused = true;
  broadcastState();
  res.json({ ok: true });
});

app.post('/api/resume', (req, res) => {
  if (isPaused) {
    isPaused = false;
    if (isRunning) runNext();
  }
  broadcastState();
  res.json({ ok: true });
});

app.post('/api/stop', (req, res) => {
  isRunning = false;
  isPaused = false;
  stopCurrentJob();
  broadcastState();
  res.json({ ok: true });
});

app.delete('/api/queue/:id', (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) return res.status(400).json({ error: 'invalid id' });
  const item = queue.find(i => i.id === id);
  if (!item) return res.status(404).json({ error: 'not found' });
  queue = queue.filter(i => i.id !== id);
  if (item.status === 'running') stopCurrentJob();
  broadcastState();
  res.json({ ok: true });
});

app.post('/api/queue/:id/retry', (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) return res.status(400).json({ error: 'invalid id' });
  const item = queue.find(i => i.id === id);
  if (!item) return res.status(404).json({ error: 'not found' });
  if (item.status !== 'error') return res.status(400).json({ error: 'only error items can be retried' });
  resetItem(item);
  broadcastState();
  res.json({ ok: true });
});

// ─── Open folder ─────────────────────────────────────────────────────────────
// Shows a file selected in Explorer / Finder (Linux: opens its folder). Takes
// queue item ids, never paths, and requires a JSON body, so a cross-site form
// POST can't trigger it. WT_NO_REVEAL (tests) only reports the path.
// Opening the folder is what matters; the selection is a bonus (Explorer
// skips it when the folder changed in the last few seconds).
function revealInFileManager(file) {
  if (process.env.WT_NO_REVEAL) return;
  const opts = { detached: true, stdio: 'ignore' };
  let child;
  if (process.platform === 'win32') {
    // explorer wants /select,"path" as one verbatim argument
    child = spawn('explorer.exe', [`/select,"${file}"`], { ...opts, windowsVerbatimArguments: true });
  } else if (process.platform === 'darwin') {
    child = spawn('open', ['-R', file], opts);
  } else {
    child = spawn('xdg-open', [path.dirname(file)], opts);
  }
  child.on('error', e => console.warn('Could not open the file manager:', e.message));
  child.unref();
}

function requireJson(req, res) {
  if (req.is('application/json')) return true;
  res.status(400).json({ error: 'send a JSON body' });
  return false;
}

// The file to show for a video: its subtitles if there are any, else the video
function bestFileFor(item) {
  const p = subtitlePaths(item.filePath);
  return [p.en, p.single, p.ja].find(f => fs.existsSync(f)) || item.filePath;
}

app.post('/api/queue/:id/reveal', (req, res) => {
  if (!requireJson(req, res)) return;
  const item = queue.find(i => i.id === parseInt(req.params.id, 10));
  if (!item) return res.status(404).json({ error: 'not found' });
  const file = bestFileFor(item);
  revealInFileManager(file);
  res.json({ ok: true, path: file });
});

app.post('/api/concat/reveal', (req, res) => {
  if (!requireJson(req, res)) return;
  if (!concatJob || concatJob.status !== 'done' || !fs.existsSync(concatJob.outputPath)) {
    return res.status(400).json({ error: 'no finished concat output' });
  }
  revealInFileManager(concatJob.outputPath);
  res.json({ ok: true, path: concatJob.outputPath });
});

app.post('/api/queue/retry-errors', (req, res) => {
  const failed = queue.filter(i => i.status === 'error');
  failed.forEach(resetItem);
  broadcastState();
  res.json({ ok: true, count: failed.length });
});

// Subtitles of one item for the preview: cues of video.ja.srt, video.en.srt
// and (single-stage) video.srt, each null if the file isn't there
app.get('/api/subs/:id', (req, res) => {
  const item = queue.find(i => i.id === parseInt(req.params.id, 10));
  if (!item) return res.status(404).json({ error: 'not found' });
  const p = subtitlePaths(item.filePath);
  const read = f => {
    try { return parseSrt(fs.readFileSync(f, 'utf8')); } catch (e) { return null; }
  };
  res.json({ name: path.basename(item.filePath), ja: read(p.ja), en: read(p.en), srt: read(p.single) });
});

// Re-run only the translation stage from the existing video.ja.srt
app.post('/api/queue/:id/retranslate', (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) return res.status(400).json({ error: 'invalid id' });
  const item = queue.find(i => i.id === id);
  if (!item) return res.status(404).json({ error: 'not found' });
  if (item.status === 'running' || item.status === 'queued') return res.status(400).json({ error: 'item is already queued or running' });
  if (!config.translation.enabled) return res.status(400).json({ error: 'Turn on translation in Settings first' });
  if (!fs.existsSync(subtitlePaths(item.filePath).ja)) return res.status(400).json({ error: 'no Japanese transcript (.ja.srt) for this file' });
  resetItem(item);
  broadcastState();
  res.json({ ok: true });
});

app.post('/api/queue/clear-done', (req, res) => {
  queue = queue.filter(i => i.status !== 'done');
  broadcastState();
  res.json({ ok: true });
});

app.post('/api/queue/reorder', (req, res) => {
  const { ids } = req.body; // array of ids in new order
  if (!Array.isArray(ids)) return res.status(400).json({ error: 'ids required' });
  const map = Object.fromEntries(queue.map(i => [i.id, i]));
  const reordered = ids.map(id => map[id]).filter(Boolean);
  const rest = queue.filter(i => !ids.includes(i.id));
  queue = [...reordered, ...rest];
  broadcastState();
  res.json({ ok: true });
});

app.get('/api/log/:id', (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) return res.status(400).json({ error: 'invalid id' });
  const item = queue.find(i => i.id === id);
  if (!item) return res.status(404).json({ error: 'not found' });
  res.json({ log: item.log });
});

app.get('/api/llama/status', async (req, res) => {
  res.json(await llamaStatus());
});

app.get('/api/llama/models', (req, res) => {
  res.json({ models: listGgufModels() });
});

// A bare command name (no folder) is looked up on PATH, like spawn() would
function onPath(name) {
  const exts = process.platform === 'win32' ? ['', ...(process.env.PATHEXT || '.EXE;.CMD;.BAT').split(';')] : [''];
  return (process.env.PATH || '').split(path.delimiter).filter(Boolean)
    .some(dir => exts.some(ext => fs.existsSync(path.join(dir, name + ext))));
}

// Settings ✓/✗ marks: { key: path } → { key: exists }
app.post('/api/check-paths', (req, res) => {
  const paths = (req.body && req.body.paths) || {};
  const out = {};
  for (const [key, p] of Object.entries(paths)) {
    if (typeof p !== 'string' || !p.trim()) { out[key] = null; continue; }
    const v = p.trim();
    out[key] = /[\\/]/.test(v) || v.startsWith('~') ? fs.existsSync(expandHome(v)) : onPath(v);
  }
  res.json(out);
});

// Settings → Save & test translation: one sample line through the saved
// endpoint, starting llama-server first if it's set up to auto-start
const SAMPLE_LINE = 'ちょっと待って、まだ準備できてないよ！';
app.post('/api/translation/test', async (req, res) => {
  if (queue.some(i => i.status === 'running')) return res.status(409).json({ ok: false, error: 'A job is running; test when the queue is idle' });
  const probe = { id: 0, log: [] }; // ensureLlama logs here
  const started = Date.now();
  try {
    if (!config.translation.enabled) throw new Error('Translation mode is single-stage; choose Two-stage first');
    await ensureLlama(probe);
    const messages = buildTranslationMessages([{ text: SAMPLE_LINE }], [0], [], 0);
    const reply = await chatCompletion(messages, new AbortController().signal);
    const m = reply.match(/^\s*\[1\]\s*(.*\S)\s*$/m);
    res.json({ ok: true, japanese: SAMPLE_LINE, english: m ? m[1] : reply.trim(), ms: Date.now() - started, log: probe.log });
  } catch (e) {
    res.json({ ok: false, error: e.message, log: probe.log });
  }
});

app.post('/api/config', (req, res) => {
  const { watchFolder, backend, whisperExecutable, mlxExecutable, pythonExecutable, port, whisperArgs, mlxArgs, translation } = req.body;
  if (port !== undefined && port !== '' && port !== null) {
    const p = parseInt(port, 10);
    if (!(p >= 1024 && p <= 65535)) return res.status(400).json({ error: 'Port must be a number from 1024 to 65535' });
    config.port = p;
  } else if (port === '') {
    delete config.port; // back to the default
  }
  if (watchFolder) config.watchFolder = watchFolder;
  if (backend === 'ctranslate2' || backend === 'mlx') config.backend = backend;
  if (whisperExecutable) config.whisperExecutable = whisperExecutable;
  if (mlxExecutable) config.mlxExecutable = mlxExecutable;
  if (pythonExecutable !== undefined) config.pythonExecutable = pythonExecutable; // blank = auto-detect
  if (typeof req.body.autoStart === 'boolean') config.autoStart = req.body.autoStart;
  if (whisperArgs) config.whisperArgs = { ...config.whisperArgs, ...whisperArgs };
  if (mlxArgs) config.mlxArgs = { ...config.mlxArgs, ...mlxArgs };
  if (translation) {
    const { apiKeySet, defaultStylePrompt, ...t } = translation;
    if (!t.apiKey) delete t.apiKey; // blank = keep the saved key
    config.translation = { ...config.translation, ...t };
  }
  saveConfig(config);
  if (watchFolder) {
    // Clear all non-running queue items and synchronously scan the new folder
    // so the response already contains the full file list (don't wait on chokidar)
    queue = queue.filter(i => i.status === 'running');
    scanWatchFolder();
    broadcastState();
    startWatcher(config.watchFolder);
  }
  res.json({ ok: true, config: publicConfig(), queue: queue.map(itemSnapshot) });
});

// Synchronously adds the watch folder's .mp4s that aren't queued yet
// (so a response can carry the full list without waiting on chokidar)
function scanWatchFolder() {
  const scanDir = (dir, depth) => {
    if (depth > 10) return;
    try {
      fs.readdirSync(dir, { withFileTypes: true }).forEach(entry => {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          scanDir(full, depth + 1);
        } else if (entry.name.toLowerCase().endsWith('.mp4') && !queueHas(full)) {
          const item = makeItem(full);
          if (isDone(full)) { item.status = 'done'; item.progress = 100; }
          queue.push(item);
        }
      });
    } catch (e) { /* ignore permission errors on system dirs */ }
  };
  if (fs.existsSync(config.watchFolder)) scanDir(config.watchFolder, 0);
}

// Refresh button: rescan the current watch folder without losing the queue
// (unlike re-applying the folder, which clears every non-running item).
// Drops videos that are gone, marks queued ones done if their subtitles appeared.
app.post('/api/queue/rescan', (req, res) => {
  queue = queue.filter(i => i.status === 'running' || fs.existsSync(i.filePath));
  queue.forEach(i => {
    if (i.status === 'queued' && isDone(i.filePath)) { i.status = 'done'; i.progress = 100; }
  });
  scanWatchFolder();
  broadcastState();
  res.json({ ok: true, queue: queue.map(itemSnapshot) });
});

app.get('/api/browse', (req, res) => {
  // Returns folder listing for a given path (for the folder picker)
  const dir = req.query.path || (process.platform === 'win32' ? 'C:\\' : os.homedir());
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    res.json({
      path: dir,
      entries: entries
        .filter(e => e.isDirectory())
        .map(e => ({ name: e.name, path: path.join(dir, e.name) }))
    });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

// Subfolders (recursively, skipping hidden ones) + video files under one directory
const VIDEO_TREE_DEPTH = 6;
function listVideoTree(dir, depth = 0) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const byName = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
  const dirs = depth >= VIDEO_TREE_DEPTH ? [] : entries
    .filter(e => e.isDirectory() && !e.name.startsWith('.'))
    .sort(byName)
    .map(e => {
      const full = path.join(dir, e.name);
      let sub = { dirs: [], files: [] };
      try { sub = listVideoTree(full, depth + 1); } catch (err) { /* unreadable */ }
      return { name: e.name, path: full, dirs: sub.dirs, files: sub.files };
    });
  const files = entries
    .filter(e => e.isFile() && VIDEO_EXTS.includes(path.extname(e.name).toLowerCase()))
    .sort(byName)
    .map(e => {
      const full = path.join(dir, e.name);
      let size = 0;
      try { size = fs.statSync(full).size; } catch (err) { /* unreadable */ }
      return { name: e.name, path: full, size };
    });
  return { dirs, files };
}

app.get('/api/videos', (req, res) => {
  // Lists subfolders (with their contents) + video files in one directory (for the concat tab)
  const dir = req.query.path || config.watchFolder;
  try {
    const parent = path.dirname(dir);
    res.json({ path: dir, parent: parent !== dir ? parent : null, ...listVideoTree(dir) });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.post('/api/concat', (req, res) => {
  const { files, outputName, deleteOriginals } = req.body;
  if (concatJob && concatJob.status === 'running') return res.status(409).json({ error: 'A concat job is already running' });
  if (!Array.isArray(files) || files.length < 2) return res.status(400).json({ error: 'Select at least 2 files' });
  const missing = files.find(f => typeof f !== 'string' || !fs.existsSync(f));
  if (missing !== undefined) return res.status(400).json({ error: `File not found: ${missing}` });

  let name = String(outputName || '').trim();
  if (!name) return res.status(400).json({ error: 'Output file name is required' });
  if (/[\\/:*?"<>|]/.test(name)) return res.status(400).json({ error: 'Output name cannot contain \\ / : * ? " < > |' });
  if (!path.extname(name)) name += path.extname(files[0]);

  // Save next to the source videos (the first selected file's folder)
  const outputPath = path.join(path.dirname(files[0]), name);
  if (files.includes(outputPath)) return res.status(400).json({ error: 'Output cannot overwrite one of the inputs' });
  if (fs.existsSync(outputPath)) return res.status(409).json({ error: `File already exists: ${outputPath}` });

  startConcat(files, outputPath, deleteOriginals === true);
  res.json({ ok: true, outputPath });
});

app.post('/api/concat/cancel', (req, res) => {
  if (concatJob && concatJob.status === 'running') {
    concatJob.cancelRequested = true;
    if (concatProcess) concatProcess.kill('SIGTERM');
  }
  res.json({ ok: true });
});

// ─── Shutdown / restart ──────────────────────────────────────────────────────
// Used by the gear menu (POST /api/server/*) and by signals. Stops the running
// whisper/ffmpeg child so nothing keeps working (or holding the GPU) after we exit.
let shuttingDown = false;
let exitAfterJob = false; // Ctrl+C while a job runs: finish it, then exit

function shutdown({ restart = false } = {}) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(restart ? '\n  Restarting...' : '\n  Shutting down...');
  isRunning = false;
  stopCurrentJob();
  if (!restart) stopLlama(); // a restart keeps the model loaded for the new server
  if (concatJob && concatJob.status === 'running') {
    concatJob.cancelRequested = true;
    if (concatProcess) concatProcess.kill('SIGTERM');
    try { fs.unlinkSync(concatJob.outputPath); } catch (e) { /* not created yet */ }
  }
  if (watcher) watcher.close();
  wss.clients.forEach(c => c.terminate());

  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    if (restart) {
      // A fresh copy of this server; it retries the port until we've let go
      spawn(process.execPath, process.argv.slice(1), {
        cwd: process.cwd(), env: process.env, detached: true, stdio: 'inherit'
      }).unref();
    }
    process.exit(0);
  };
  server.close(finish);
  if (server.closeAllConnections) server.closeAllConnections(); // idle keep-alive sockets
  setTimeout(finish, 3000).unref();
}

// Only a same-origin JSON POST from the UI can trigger these: a cross-site
// JSON request needs a CORS preflight, which this server never grants.
function confirmedAction(req, res) {
  if (req.is('application/json') && req.body && req.body.confirm === true) return true;
  res.status(400).json({ error: 'send {"confirm": true} as JSON' });
  return false;
}

app.post('/api/server/shutdown', (req, res) => {
  if (!confirmedAction(req, res)) return;
  res.json({ ok: true });
  setImmediate(shutdown);
});

app.post('/api/server/restart', (req, res) => {
  if (!confirmedAction(req, res)) return;
  // Under `npm run dev`, nodemon owns the process: touching a watched file makes
  // it restart us, instead of a detached copy escaping nodemon
  if (process.env.npm_lifecycle_event === 'dev') {
    stopCurrentJob();
    isRunning = false;
    res.json({ ok: true });
    const now = new Date();
    setImmediate(() => fs.utimesSync(__filename, now, now));
    return;
  }
  res.json({ ok: true });
  setImmediate(() => shutdown({ restart: true }));
});

// ─── Start ───────────────────────────────────────────────────────────────────
// 3939 rather than 3000 so it doesn't collide with other Node apps.
// PORT env var > config.json "port" > 3939. A changed config port applies on reboot.
const DEFAULT_PORT = 3939;
const PORT = parseInt(process.env.PORT, 10) || parseInt(config.port, 10) || DEFAULT_PORT;

// The port a reboot will come back on (env PORT, if set, still wins)
function nextPort() {
  return parseInt(process.env.PORT, 10) || parseInt(config.port, 10) || DEFAULT_PORT;
}

// A restarted server can start before its predecessor has released the port,
// so retry briefly before giving up.
function listen(attempt = 0) {
  const onError = err => {
    if (err.code === 'EADDRINUSE' && attempt < 20) {
      setTimeout(() => listen(attempt + 1), 250);
      return;
    }
    console.error(err.code === 'EADDRINUSE'
      ? `\n  Port ${PORT} is already in use. Is Subtitle Studio already running? Use its gear menu → Shut down.\n`
      : `\n  Server error: ${err.message}\n`);
    process.exit(1);
  };
  server.once('error', onError);
  server.listen(PORT, '127.0.0.1', () => {
    server.removeListener('error', onError);
    console.log(`\n  Subtitle Studio running at http://localhost:${PORT}\n`);
  });
}
listen();

// After a restart the terminal may be gone; don't crash writing logs to it
process.stdout.on('error', () => {});
process.stderr.on('error', () => {});

// Closing the terminal window (SIGHUP) or `kill` (SIGTERM): stop everything now
process.on('SIGHUP', () => shutdown());
process.on('SIGTERM', () => shutdown());

process.on('SIGINT', () => {
  if (exitAfterJob || !(currentProcess || currentAbort)) return shutdown();
  // First Ctrl+C during a job: let it finish (no new jobs start), then exit
  exitAfterJob = true;
  isRunning = false;
  console.log('\n  Finishing the current job, then exiting (Ctrl+C again to stop it now)...');
});

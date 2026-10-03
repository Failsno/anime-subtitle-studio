// Shared helpers for the end-to-end tests: each test runs a real server
// (src/server.js) on a free port against its own temp config + watch folder.
const { spawn } = require('child_process');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const FIXTURES = path.join(__dirname, 'fixtures');
const HOME = os.homedir();
const isWin = process.platform === 'win32';

// Everything machine-specific is overridable with env vars (see README → Testing)
const env = {
  llmUrl: process.env.WT_TEST_LLM_URL || 'http://127.0.0.1:9931/v1',
  llmKey: process.env.WT_TEST_LLM_KEY || '',
  llmModel: process.env.WT_TEST_LLM_MODEL || '',
  mlxExe: process.env.WT_TEST_MLX_EXE || path.join(HOME, 'mlx-whisper-env', 'bin', 'mlx_whisper'),
  mlxModel: process.env.WT_TEST_MLX_MODEL || path.join(HOME, 'models', 'anime-whisper-mlx'),
  ct2Exe: process.env.WT_TEST_CT2_EXE || '',
  ct2Model: process.env.WT_TEST_CT2_MODEL || path.join(HOME, 'models', 'anime-whisper-ct2'),
  ct2Device: process.env.WT_TEST_CT2_DEVICE || (process.platform === 'darwin' ? 'cpu' : 'cuda'),
  ct2Compute: process.env.WT_TEST_CT2_COMPUTE || (process.platform === 'darwin' ? 'float32' : 'int8_float16'),
};

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
    srv.on('error', reject);
  });
}

// Temp watch folder with the fixture video (and optionally its .ja.srt)
function makeWatchFolder({ withJa = false } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wt-test-'));
  fs.copyFileSync(path.join(FIXTURES, 'speech.mp4'), path.join(dir, 'speech.mp4'));
  if (withJa) fs.copyFileSync(path.join(FIXTURES, 'speech.ja.srt'), path.join(dir, 'speech.ja.srt'));
  return dir;
}

function baseConfig(watchFolder, overrides = {}) {
  const cfg = {
    watchFolder,
    backend: 'mlx',
    whisperExecutable: 'whisper-ctranslate2',
    mlxExecutable: 'mlx_whisper',
    whisperArgs: {},
    mlxArgs: {},
    translation: { enabled: false, baseUrl: env.llmUrl, apiKey: env.llmKey, model: env.llmModel }
  };
  for (const [k, v] of Object.entries(overrides)) {
    cfg[k] = v && typeof v === 'object' && !Array.isArray(v) ? { ...cfg[k], ...v } : v;
  }
  return cfg;
}

// portVia: 'env' passes PORT (normal), 'config' writes it to config.json instead
async function startServer(config, { portVia = 'env' } = {}) {
  const port = await freePort();
  if (portVia === 'config') config = { ...config, port };
  const cfgPath = path.join(config.watchFolder, '..', path.basename(config.watchFolder) + '.config.json');
  fs.writeFileSync(cfgPath, JSON.stringify(config, null, 2));
  const proc = spawn(process.execPath, [path.join(ROOT, 'src', 'server.js')], {
    env: { ...process.env, WT_CONFIG: cfgPath, PORT: portVia === 'env' ? String(port) : '' },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let output = '';
  proc.stdout.on('data', d => { output += d; });
  proc.stderr.on('data', d => { output += d; });
  await waitFor(() => output.includes('running at'), 10000, 'server start');
  const url = p => `http://127.0.0.1:${port}${p}`;
  return {
    port,
    output: () => output,
    get: async p => (await fetch(url(p))).json(),
    post: async (p, json) => {
      const res = await fetch(url(p), json === undefined ? { method: 'POST' } : {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(json)
      });
      return { status: res.status, body: await res.json() };
    },
    proc,
    exited: () => proc.exitCode !== null || proc.signalCode !== null
      ? Promise.resolve()
      : new Promise(resolve => proc.once('exit', resolve)),
    // Wait until the queue's first item has settled into done/error
    waitForItem: async (pred, timeoutMs) => {
      let item;
      await waitFor(async () => {
        item = (await (await fetch(url('/api/state'))).json()).queue[0];
        return item && pred(item);
      }, timeoutMs, 'queue item');
      return item;
    },
    log: async id => (await (await fetch(url(`/api/log/${id}`))).json()).log,
    stop: () => new Promise(resolve => {
      const done = () => { fs.rmSync(cfgPath, { force: true }); resolve(); };
      if (proc.exitCode !== null || proc.signalCode !== null) return done();
      proc.once('exit', done);
      proc.kill();
    })
  };
}

async function waitFor(check, timeoutMs, what) {
  const end = Date.now() + timeoutMs;
  for (;;) {
    if (await check()) return;
    if (Date.now() > end) throw new Error(`Timed out after ${timeoutMs}ms waiting for ${what}`);
    await new Promise(r => setTimeout(r, 300));
  }
}

const settled = i => i.status === 'done' || i.status === 'error';

function parseSrt(text) {
  return text.replace(/^﻿/, '').replace(/\r\n?/g, '\n').trim().split(/\n\s*\n/).map(b => {
    const lines = b.split('\n');
    const t = lines.findIndex(l => l.includes('-->'));
    const [s, e] = lines[t].split('-->').map(x => {
      const [h, m, rest] = x.trim().split(':');
      return +h * 3600 + +m * 60 + parseFloat(rest.replace(',', '.'));
    });
    return { start: s, end: e, text: lines.slice(t + 1).join(' ').trim() };
  });
}

// Character error rate between two Japanese strings, ignoring punctuation
function cer(expected, actual) {
  const norm = s => s.replace(/[\s、。,.!?！？…「」『』ー~〜]/g, '');
  const a = [...norm(expected)];
  const b = [...norm(actual)];
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return a.length ? prev[b.length] / a.length : 0;
}

async function llmReachable() {
  try {
    const res = await fetch(env.llmUrl.replace(/\/+$/, '') + '/models', {
      headers: env.llmKey ? { Authorization: `Bearer ${env.llmKey}` } : {},
      signal: AbortSignal.timeout(3000)
    });
    return res.ok;
  } catch (e) {
    return false;
  }
}

function whisperJobDirs() {
  return fs.readdirSync(os.tmpdir()).filter(n => n.startsWith('whisper-job-')).sort();
}

function pidAlive(pid) {
  try { process.kill(pid, 0); return true; } catch (e) { return false; }
}

module.exports = {
  pidAlive, waitFor, freePort,
  FIXTURES, env, isWin, makeWatchFolder, baseConfig, startServer, settled,
  parseSrt, cer, llmReachable, whisperJobDirs
};

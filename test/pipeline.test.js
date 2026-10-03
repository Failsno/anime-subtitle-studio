// End-to-end tests for the transcribe → translate pipeline. Run: npm test
//
// Tests needing something machine-specific (LLM endpoint, mlx venv, CT2 setup)
// skip themselves with the reason when it isn't available. See README → Testing
// for the WT_TEST_* env vars.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const {
  FIXTURES, env, isWin, makeWatchFolder, baseConfig, startServer, settled,
  parseSrt, cer, llmReachable, whisperJobDirs, pidAlive, waitFor, freePort
} = require('./helpers');

const EXPECTED = parseSrt(fs.readFileSync(path.join(FIXTURES, 'speech.ja.srt'), 'utf8'));
const JAPANESE = /[぀-ヿ一-鿿]/;
const MIN = 60 * 1000;
// Test servers report what "open folder" would show instead of opening windows
process.env.WT_NO_REVEAL = '1';
const sleepMs = ms => new Promise(r => setTimeout(r, ms));

const mlxReady = process.platform === 'darwin' && fs.existsSync(env.mlxExe) && fs.existsSync(env.mlxModel);
const ct2Ready = !!env.ct2Exe && fs.existsSync(env.ct2Exe) && fs.existsSync(env.ct2Model);
// englishOnly off: these tests inspect (and re-translate from) the .ja.srt
const translation = { enabled: true, batchSize: '4', contextLines: '2', englishOnly: false };

const mlxBackend = () => ({
  backend: 'mlx',
  mlxExecutable: env.mlxExe,
  mlxArgs: { model: env.mlxModel, task: 'transcribe', vad_segmentation: 'True', initial_prompt: '' }
});
const ct2Backend = () => ({
  backend: 'ctranslate2',
  whisperExecutable: env.ct2Exe,
  whisperArgs: {
    model: env.ct2Model, task: 'transcribe', vad_segmentation: 'True', initial_prompt: '',
    device: env.ct2Device, compute_type: env.ct2Compute, beam_size: '5'
  }
});

// The transcript must match the fixture: same cues, timings within 0.3s,
// and close text (anime-whisper scores 0% CER on this clip).
function assertMatchesFixture(srtPath) {
  const cues = parseSrt(fs.readFileSync(srtPath, 'utf8'));
  assert.equal(cues.length, EXPECTED.length, 'cue count');
  cues.forEach((c, i) => assert.ok(Math.abs(c.start - EXPECTED[i].start) < 0.3, `cue ${i + 1} start ${c.start} vs ${EXPECTED[i].start}`));
  const meanCer = cues.reduce((sum, c, i) => sum + cer(EXPECTED[i].text, c.text), 0) / cues.length;
  assert.ok(meanCer < 0.15, `mean CER ${meanCer.toFixed(3)} too high`);
}

// Every transcribed line must stream to the job log intact (Windows children
// end lines with \r\n, which once blanked them all)
function assertTranscriptLogged(log) {
  assert.ok(log.some(l => /^VAD: \d+ speech segments$/.test(l)), 'VAD summary in log');
  const logged = log.filter(l => /^\[\d\d:\d\d:\d\d\.\d+ --> /.test(l));
  assert.equal(logged.length, EXPECTED.length, 'transcript lines in log');
  assert.ok(logged.every(l => !l.includes('\uFFFD')), 'no broken UTF-8 in log');
}

function assertEnglishTranslation(enPath, jaPath) {
  const en = parseSrt(fs.readFileSync(enPath, 'utf8'));
  const ja = parseSrt(fs.readFileSync(jaPath, 'utf8'));
  assert.equal(en.length, ja.length, 'English cue count matches Japanese');
  en.forEach((c, i) => assert.equal(c.start, ja[i].start, `cue ${i + 1} keeps its timing`));
  const untranslated = en.filter(c => JAPANESE.test(c.text)).length;
  assert.ok(untranslated <= 1, `${untranslated} cues still contain Japanese`);
}

async function withServer(config, fn) {
  const server = await startServer(config);
  try {
    await fn(server);
  } finally {
    await server.stop();
    fs.rmSync(config.watchFolder, { recursive: true, force: true });
  }
}

// ─── Always runnable ─────────────────────────────────────────────────────────

test('missing whisper executable → error, no temp dirs left behind', { timeout: MIN }, async () => {
  const before = whisperJobDirs();
  const dir = makeWatchFolder();
  await withServer(baseConfig(dir, {
    backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper', translation
  }), async s => {
    await s.post('/api/start');
    const item = await s.waitForItem(settled, 30000);
    assert.equal(item.status, 'error');
    assert.ok((await s.log(item.id)).some(l => /ENOENT|not found|cannot find/i.test(l)), 'log shows the spawn failure');
    assert.ok(!fs.existsSync(path.join(dir, 'speech.ja.srt')));
  });
  assert.deepEqual(whisperJobDirs(), before, 'temp whisper-job dirs cleaned up');
});

test('whisper exits 0 without writing an SRT → error, not done', { timeout: MIN, skip: isWin && 'needs a POSIX shell script' }, async () => {
  const dir = makeWatchFolder();
  const fake = path.join(dir, '..', path.basename(dir) + '-fake-whisper.sh');
  fs.writeFileSync(fake, '#!/bin/sh\necho "pretending to transcribe"\nexit 0\n', { mode: 0o755 });
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: fake, whisperArgs: { vad_segmentation: 'False' }
    }), async s => {
      await s.post('/api/start');
      const item = await s.waitForItem(settled, 30000);
      assert.equal(item.status, 'error');
    });
  } finally {
    fs.rmSync(fake, { force: true });
  }
});

test('retranslate is refused while translation is turned off', { timeout: MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  await withServer(baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper' }), async s => {
    await s.post('/api/start');
    await s.waitForItem(settled, 30000);
    const res = await s.post('/api/queue/1/retranslate');
    assert.equal(res.status, 400);
    assert.match(res.body.error, /translation/i);
  });
});

// ─── Gear menu: shut down / reboot the server ────────────────────────────────

// A whisper stand-in that records its PID and runs until killed
function longRunningWhisper(dir) {
  const fake = path.join(dir, '..', path.basename(dir) + '-slow-whisper.sh');
  const pidFile = fake + '.pid';
  fs.writeFileSync(fake, `#!/bin/sh\necho $$ > '${pidFile}'\necho "VAD: 9 speech segments"\nexec sleep 600\n`, { mode: 0o755 });
  return { fake, pidFile, cleanup: () => { fs.rmSync(fake, { force: true }); fs.rmSync(pidFile, { force: true }); } };
}

test('shutdown needs a JSON confirm, then stops the job, exits and frees the port', {
  timeout: MIN, skip: isWin && 'needs a POSIX shell script'
}, async () => {
  const dir = makeWatchFolder();
  const w = longRunningWhisper(dir);
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: w.fake, whisperArgs: { vad_segmentation: 'False' }
    }), async s => {
      assert.equal((await s.post('/api/server/shutdown')).status, 400, 'plain POST refused');
      assert.equal((await s.post('/api/server/shutdown', { confirm: false })).status, 400);

      await s.post('/api/start');
      await waitFor(() => fs.existsSync(w.pidFile), 15000, 'whisper to start');
      const pid = parseInt(fs.readFileSync(w.pidFile, 'utf8'), 10);
      assert.ok(pidAlive(pid));

      assert.equal((await s.post('/api/server/shutdown', { confirm: true })).status, 200);
      await s.exited();
      assert.equal(s.proc.exitCode, 0, 'clean exit');
      await waitFor(() => !pidAlive(pid), 5000, 'whisper child to be killed');
      await assert.rejects(fetch(`http://127.0.0.1:${s.port}/api/state`), 'port is free');
    });
  } finally {
    w.cleanup();
  }
});

test('reboot starts a fresh server on the same port', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  await withServer(baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper' }), async s => {
    assert.equal((await s.post('/api/server/restart')).status, 400, 'plain POST refused');
    assert.equal((await s.post('/api/server/restart', { confirm: true })).status, 200);
    await s.exited();
    // The replacement is a detached process; wait for it to answer
    await waitFor(async () => {
      try { return (await fetch(`http://127.0.0.1:${s.port}/api/state`)).ok; } catch (e) { return false; }
    }, 15000, 'restarted server');
    const state = await s.get('/api/state');
    assert.equal(state.queue.length, 1, 'watch folder rescanned');
    // Clean up the replacement through the same endpoint
    await s.post('/api/server/shutdown', { confirm: true });
    await waitFor(async () => {
      try { await fetch(`http://127.0.0.1:${s.port}/api/state`); return false; } catch (e) { return true; }
    }, 10000, 'replacement to shut down');
  });
});

test('port set in Settings applies on reboot; invalid ports are refused', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  const cfg = baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper' });
  const server = await startServer(cfg, { portVia: 'config' });
  const answers = async port => {
    try { return (await fetch(`http://127.0.0.1:${port}/api/state`)).ok; } catch (e) { return false; }
  };
  const newPort = await freePort();
  try {
    const bad = await fetch(`http://127.0.0.1:${server.port}/api/config`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ port: '80' })
    });
    assert.equal(bad.status, 400, 'privileged port refused');

    const ok = await fetch(`http://127.0.0.1:${server.port}/api/config`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ port: String(newPort) })
    });
    const { config } = await ok.json();
    assert.equal(config.runningPort, server.port, 'still running on the old port');
    assert.equal(config.nextPort, newPort, 'reboot target is the new port');

    await server.post('/api/server/restart', { confirm: true });
    await server.exited();
    await waitFor(() => answers(newPort), 15000, 'server on the new port');
    assert.equal(await answers(server.port), false, 'old port released');
  } finally {
    await fetch(`http://127.0.0.1:${newPort}/api/server/shutdown`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirm: true })
    }).catch(() => {});
    await waitFor(async () => !(await answers(newPort)), 10000, 'replacement to shut down').catch(() => {});
    await server.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('SIGHUP (terminal window closed) stops the job and exits', {
  timeout: MIN, skip: isWin && 'needs POSIX signals'
}, async () => {
  const dir = makeWatchFolder();
  const w = longRunningWhisper(dir);
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: w.fake, whisperArgs: { vad_segmentation: 'False' }
    }), async s => {
      await s.post('/api/start');
      await waitFor(() => fs.existsSync(w.pidFile), 15000, 'whisper to start');
      const pid = parseInt(fs.readFileSync(w.pidFile, 'utf8'), 10);
      s.proc.kill('SIGHUP');
      await s.exited();
      await waitFor(() => !pidAlive(pid), 5000, 'whisper child to be killed');
    });
  } finally {
    w.cleanup();
  }
});

// ─── UI support endpoints ────────────────────────────────────────────────────

test('errored items carry their error and timings; Retry errors requeues them all', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  fs.copyFileSync(path.join(dir, 'speech.mp4'), path.join(dir, 'speech2.mp4'));
  await withServer(baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper' }), async s => {
    await s.post('/api/start');
    await waitFor(async () => (await s.get('/api/state')).queue.every(settled), 30000, 'both items to fail');
    const { queue } = await s.get('/api/state');
    for (const item of queue) {
      assert.equal(item.status, 'error');
      assert.match(item.lastError, /ENOENT|not found|cannot find/i);
      assert.ok(item.startedAt > 0 && item.finishedAt >= item.startedAt, 'timings recorded');
    }
    const res = await s.post('/api/queue/retry-errors');
    assert.deepEqual(res.body, { ok: true, count: 2 });
    const after = (await s.get('/api/state')).queue;
    assert.ok(after.every(i => i.status === 'queued' && !i.lastError && !i.finishedAt), 'requeued and reset');
  });
});

test('auto-start respects Pause: a new video waits until Resume', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  // A whisper stand-in that takes a moment, then fails (no .srt written)
  const slow = path.join(dir, '..', path.basename(dir) + '-slow-whisper.js');
  fs.writeFileSync(slow, 'setTimeout(() => process.exit(1), 2000);\n');
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: slow, whisperArgs: { vad_segmentation: 'False' }, autoStart: true
    }), async s => {
      await sleepMs(3500); // past chokidar's initial scan
      await s.post('/api/start');
      await waitFor(async () => (await s.get('/api/state')).queue[0].status === 'running', 10000, 'job to start');
      await s.post('/api/pause');
      await waitFor(async () => (await s.get('/api/state')).queue[0].status === 'error', 15000, 'job to finish');

      fs.copyFileSync(path.join(dir, 'speech.mp4'), path.join(dir, 'new.mp4'));
      await waitFor(async () => (await s.get('/api/state')).queue.some(i => i.name === 'new.mp4'), 15000, 'new video to be found');
      await sleepMs(500);
      let state = await s.get('/api/state');
      assert.equal(state.isPaused, true, 'still paused');
      assert.equal(state.queue.find(i => i.name === 'new.mp4').status, 'queued', 'waits while paused');

      await s.post('/api/resume');
      await waitFor(async () => {
        state = await s.get('/api/state');
        return state.queue.find(i => i.name === 'new.mp4').status === 'error'; // it ran
      }, 15000, 'new video to run after Resume');
    });
  } finally {
    fs.rmSync(slow, { force: true });
  }
});

test('open folder selects the best subtitle file, by queue item id only', { timeout: MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  await withServer(baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper' }), async s => {
    const plain = await fetch(`http://127.0.0.1:${s.port}/api/queue/1/reveal`, { method: 'POST' });
    assert.equal(plain.status, 400, 'a non-JSON POST (e.g. a cross-site form) is refused');
    assert.equal((await s.post('/api/queue/999/reveal', {})).status, 404);
    const shown = async () => (await s.post('/api/queue/1/reveal', {})).body.path;
    assert.equal(await shown(), path.join(dir, 'speech.ja.srt'));
    fs.writeFileSync(path.join(dir, 'speech.srt'), '');
    assert.equal(await shown(), path.join(dir, 'speech.srt'));
    fs.writeFileSync(path.join(dir, 'speech.en.srt'), '');
    assert.equal(await shown(), path.join(dir, 'speech.en.srt'));
  });
});

test('Concat: show the joined file once it exists', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  fs.copyFileSync(path.join(dir, 'speech.mp4'), path.join(dir, 'part2.mp4'));
  await withServer(baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper' }), async s => {
    assert.equal((await s.post('/api/concat/reveal', {})).status, 400, 'nothing joined yet');
    const files = [path.join(dir, 'speech.mp4'), path.join(dir, 'part2.mp4')];
    assert.equal((await s.post('/api/concat', { files, outputName: 'joined.mp4' })).status, 200);
    await waitFor(async () => { const j = (await s.get('/api/state')).concat; return j && j.status !== 'running'; }, 30000, 'concat');
    assert.equal((await s.get('/api/state')).concat.status, 'done');
    assert.equal((await s.post('/api/concat/reveal', {})).body.path, path.join(dir, 'joined.mp4'));
  });
});

test('VAD segmentation passes speech padding and the anti-loop settings to the script', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  // A "python" that records its arguments and writes an empty transcript
  const fake = path.join(dir, '..', path.basename(dir) + '-fake-python.js');
  const argsFile = fake + '.args.json';
  fs.writeFileSync(fake, `const fs = require('fs'), path = require('path');
fs.writeFileSync(${JSON.stringify(argsFile)}, JSON.stringify(process.argv.slice(2)));
const out = process.argv[process.argv.indexOf('--output-dir') + 1];
fs.writeFileSync(path.join(out, 'speech.srt'), '');
`);
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'whisper-ctranslate2', pythonExecutable: fake,
      whisperArgs: { model: 'large-v2', device: 'cpu', compute_type: 'int8', vad_segmentation: 'True',
        vad_speech_pad_ms: '400', repetition_penalty: '1.5', no_repeat_ngram_size: '3' }
    }), async s => {
      await s.post('/api/start');
      await waitFor(() => fs.existsSync(argsFile), 30000, 'fake python to run');
      const args = JSON.parse(fs.readFileSync(argsFile, 'utf8'));
      const flag = f => args[args.indexOf(f) + 1];
      assert.equal(flag('--vad-speech-pad-ms'), '400');
      assert.equal(flag('--repetition-penalty'), '1.5');
      assert.equal(flag('--no-repeat-ngram-size'), '3');
    });
  } finally {
    fs.rmSync(fake, { force: true });
    fs.rmSync(argsFile, { force: true });
  }
});

test('check-paths reports which files, folders and PATH executables exist', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  await withServer(baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper' }), async s => {
    const res = await s.post('/api/check-paths', { paths: {
      folder: dir, file: path.join(dir, 'speech.mp4'), missing: path.join(dir, 'nope.gguf'),
      onPath: 'node', notOnPath: 'definitely-not-a-real-exe', home: '~'
    } });
    assert.deepEqual(res.body, { folder: true, file: true, missing: false, onPath: true, notOnPath: false, home: true });
  });
});

test('auto-start picks up new videos, but not the ones found at startup', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  await withServer(baseConfig(dir, {
    backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper', autoStart: true
  }), async s => {
    await sleepMs(4000); // past chokidar's initial scan
    let state = await s.get('/api/state');
    assert.equal(state.isRunning, false, 'startup scan does not start the queue');
    assert.equal(state.queue[0].status, 'queued');
    fs.copyFileSync(path.join(dir, 'speech.mp4'), path.join(dir, 'new.mp4'));
    await waitFor(async () => {
      state = await s.get('/api/state');
      const added = state.queue.find(i => i.name === 'new.mp4');
      return added && added.status === 'error'; // it ran (and failed on the fake whisper)
    }, 30000, 'new video to be processed');
  });
});

// ─── llama-server auto-start (with a fake llama-server) ──────────────────────

// Translation settings that make the app launch test/fake-llama-server.js
async function fakeLlama(dir, { loadMs = 1000, loadWait = '15' } = {}) {
  const port = await freePort();
  const pidFile = path.join(dir, '..', path.basename(dir) + '-llama.pid');
  const script = path.join(__dirname, 'fake-llama-server.js');
  return {
    port, pidFile,
    pid: () => parseInt(fs.readFileSync(pidFile, 'utf8'), 10),
    translation: {
      ...translation, baseUrl: `http://127.0.0.1:${port}/v1`, apiKey: '', model: '',
      llamaExecutable: process.execPath, llamaModel: '',
      llamaArgs: `"${script}" --load-ms ${loadMs} --pid-file "${pidFile}"`, llamaLoadWait: loadWait
    },
    cleanup() {
      try { process.kill(this.pid()); } catch (e) { /* not running */ }
      fs.rmSync(pidFile, { force: true });
    }
  };
}

const answersOn = async port => {
  try { await fetch(`http://127.0.0.1:${port}/health`); return true; } catch (e) { return false; }
};

test('llama-server auto-starts for translation, .ja.srt is deleted by default, Restart keeps it, Shutdown stops it', { timeout: 2 * MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  const llama = await fakeLlama(dir);
  const { englishOnly, ...defaults } = llama.translation; // englishOnly left at its default
  try {
    await withServer(baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper', translation: defaults }), async s => {
      assert.equal(await answersOn(llama.port), false, 'nothing on the llama port yet');
      assert.equal((await s.get('/api/llama/status')).state, 'stopped');
      await s.post('/api/start');
      const item = await s.waitForItem(settled, MIN);
      const log = await s.log(item.id);
      assert.equal(item.status, 'done', log.slice(-5).join('\n'));
      assert.ok(log.some(l => l.startsWith('Starting llama-server')), 'started it');
      assert.deepEqual(await s.get('/api/llama/status'), {
        local: true, url: `http://127.0.0.1:${llama.port}`, autoStart: true, startedByApp: true,
        state: 'ready', model: 'fake-model-Q4_K_M.gguf', ctx: 4096
      });
      // English only: .ja.srt deleted, .en.srt renamed to plain .srt
      assert.equal(parseSrt(fs.readFileSync(path.join(dir, 'speech.srt'), 'utf8')).length, EXPECTED.length);
      assert.ok(!fs.existsSync(path.join(dir, 'speech.ja.srt')), '.ja.srt deleted');
      assert.ok(!fs.existsSync(path.join(dir, 'speech.en.srt')), '.en.srt renamed');
      assert.ok(log.some(l => l.startsWith('Deleted')), 'deletion logged');
      assert.ok(log.some(l => l.startsWith('Renamed speech.en.srt → speech.srt')), 'rename logged');
      assert.equal((await s.get('/api/state')).queue[0].status, 'done', 'still counts as done');

      // Restart leaves the model loaded; the replacement server shuts it down
      const pid = llama.pid();
      await s.post('/api/server/restart', { confirm: true });
      await s.exited();
      await waitFor(async () => {
        try { return (await fetch(`http://127.0.0.1:${s.port}/api/state`)).ok; } catch (e) { return false; }
      }, 15000, 'restarted server');
      assert.ok(pidAlive(pid), 'llama-server survived Restart');
      await s.post('/api/server/shutdown', { confirm: true });
      await waitFor(() => !pidAlive(pid), 10000, 'llama-server to be stopped by Shutdown');
      assert.equal(await answersOn(llama.port), false, 'llama port freed');
    });
  } finally {
    llama.cleanup();
  }
});

test('an already-running llama-server is reused, keep-.ja.srt works, and Shutdown still stops it', { timeout: 2 * MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  const llama = await fakeLlama(dir, { loadMs: 0 });
  const manual = require('child_process').spawn(process.execPath,
    [path.join(__dirname, 'fake-llama-server.js'), '--port', String(llama.port), '--pid-file', llama.pidFile], { stdio: 'ignore' });
  try {
    await waitFor(() => answersOn(llama.port), 10000, 'manually started llama-server');
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper',
      translation: { ...llama.translation, englishOnly: false }
    }), async s => {
      await s.post('/api/start');
      const item = await s.waitForItem(settled, MIN);
      const log = await s.log(item.id);
      assert.equal(item.status, 'done', log.slice(-5).join('\n'));
      assert.ok(!log.some(l => l.startsWith('Starting llama-server')), 'reused the running one');
      assert.ok(fs.existsSync(path.join(dir, 'speech.ja.srt')), '.ja.srt kept');
      assert.ok(fs.existsSync(path.join(dir, 'speech.en.srt')));

      // Subtitle preview: Japanese and English cues side by side
      const subs = await s.get(`/api/subs/${item.id}`);
      assert.equal(subs.ja.length, EXPECTED.length);
      assert.equal(subs.en.length, EXPECTED.length);
      assert.equal(subs.srt, null);
      assert.equal(subs.en[0].text, 'English line 1');
      assert.equal(subs.en[0].time, subs.ja[0].time);

      // Translation test from Settings: one sample line through the endpoint
      const t = await s.post('/api/translation/test');
      assert.equal(t.body.ok, true, JSON.stringify(t.body));
      assert.equal(t.body.english, 'English line 1');

      await s.post('/api/server/shutdown', { confirm: true });
      await s.exited();
      await waitFor(() => manual.exitCode !== null || manual.signalCode !== null, 10000, 'manual llama-server to be stopped');
    });
  } finally {
    manual.kill();
    llama.cleanup();
  }
});

test('changing the model in Settings restarts llama-server with it before the next translation', { timeout: 2 * MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  const llama = await fakeLlama(dir, { loadMs: 0 });
  const modelA = path.join(dir, 'model-A.gguf');
  const modelB = path.join(dir, 'model-B.gguf');
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper',
      translation: { ...llama.translation, llamaModel: modelA }
    }), async s => {
      await s.post('/api/start');
      assert.equal((await s.waitForItem(settled, MIN)).status, 'done');
      assert.equal((await s.get('/api/llama/status')).model, 'model-A.gguf');
      const pidA = llama.pid();

      await s.post('/api/config', { translation: { llamaModel: modelB } });
      const pending = await s.get('/api/llama/status');
      assert.equal(pending.model, 'model-A.gguf', 'still the old model until a translation needs it');
      assert.equal(pending.nextModel, 'model-B.gguf');

      await s.post('/api/queue/1/retranslate');
      await s.post('/api/start');
      const item = await s.waitForItem(i => i.status === 'done' || i.status === 'error', MIN);
      const log = await s.log(item.id);
      assert.equal(item.status, 'done', log.slice(-5).join('\n'));
      assert.ok(log.some(l => l.startsWith('Switching llama-server model')), 'switch logged');
      assert.ok(!pidAlive(pidA), 'old llama-server stopped');
      const after = await s.get('/api/llama/status');
      assert.equal(after.model, 'model-B.gguf');
      assert.equal(after.nextModel, undefined);
      await s.post('/api/server/shutdown', { confirm: true });
      await s.exited();
    });
  } finally {
    llama.cleanup();
  }
});

test('Settings lists installed .gguf models, without vision projectors, embeddings or extra split parts', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  const models = path.join(dir, 'models');
  fs.mkdirSync(path.join(models, 'sub'), { recursive: true });
  for (const f of ['Qwen-27B-Q6_K.gguf', 'mmproj-model-bf16.gguf', 'nomic-embed-text-v2-q8_0.gguf',
    'big-Q4-00001-of-00002.gguf', 'big-Q4-00002-of-00002.gguf', 'sub/gemma-3-27b-Q6_K.gguf', 'notes.txt']) {
    fs.writeFileSync(path.join(models, f), 'x'.repeat(10));
  }
  await withServer(baseConfig(dir, {
    backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper',
    translation: { llamaModel: path.join(models, 'Qwen-27B-Q6_K.gguf') }
  }), async s => {
    const { models: list } = await s.get('/api/llama/models');
    const mine = list.filter(m => m.path.startsWith(models)).map(m => [m.name, m.size]);
    assert.deepEqual(mine, [['big-Q4-00001-of-00002', 20], ['gemma-3-27b-Q6_K', 10], ['Qwen-27B-Q6_K', 10]]);
  });
});

test('English only: the translation replaces an existing video.srt', { timeout: MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  fs.writeFileSync(path.join(dir, 'speech.srt'), '1\n00:00:00,000 --> 00:00:01,000\nold subtitles\n');
  const llama = await fakeLlama(dir, { loadMs: 0 });
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper',
      translation: { ...llama.translation, englishOnly: true }
    }), async s => {
      // An existing video.srt makes the item done; EN↻ translates it again
      await waitFor(async () => (await s.get('/api/state')).queue.length === 1, 10000, 'scan');
      assert.equal((await s.post('/api/queue/1/retranslate')).status, 200);
      await s.post('/api/start');
      const item = await s.waitForItem(settled, 30000);
      const log = await s.log(item.id);
      assert.equal(item.status, 'done', log.slice(-5).join('\n'));
      const cues = parseSrt(fs.readFileSync(path.join(dir, 'speech.srt'), 'utf8'));
      assert.equal(cues.length, EXPECTED.length, 'new translation, not the old file');
      assert.equal(cues[0].text, 'English line 1');
      assert.ok(log.some(l => /^Renamed speech\.en\.srt → speech\.srt \(replaced/.test(l)), 'replacement logged');
      await s.post('/api/server/shutdown', { confirm: true });
      await s.exited();
    });
  } finally {
    llama.cleanup();
  }
});

test('English only: if renaming to video.srt fails, the .ja.srt is kept', { timeout: MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  // A folder named speech.srt makes the rename fail on every OS (like a locked file)
  fs.mkdirSync(path.join(dir, 'speech.srt'));
  const llama = await fakeLlama(dir, { loadMs: 0 });
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper',
      translation: { ...llama.translation, englishOnly: true }
    }), async s => {
      await waitFor(async () => (await s.get('/api/state')).queue.length === 1, 10000, 'scan');
      assert.equal((await s.post('/api/queue/1/retranslate')).status, 200);
      await s.post('/api/start');
      const item = await s.waitForItem(settled, 30000);
      assert.equal(item.status, 'error');
      assert.ok(fs.existsSync(path.join(dir, 'speech.ja.srt')), '.ja.srt kept, so Retry only translates');
      await s.post('/api/server/shutdown', { confirm: true });
      await s.exited();
    });
  } finally {
    llama.cleanup();
  }
});

test('Pause with nothing running does not leave the queue paused', { timeout: MIN }, async () => {
  const dir = makeWatchFolder();
  await withServer(baseConfig(dir, { backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper' }), async s => {
    await s.post('/api/pause');
    assert.equal((await s.get('/api/state')).isPaused, false);
  });
});

test('a bad llama-server path can be fixed in Settings without restarting the app', { timeout: MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  const llama = await fakeLlama(dir, { loadMs: 0 });
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper',
      translation: { ...llama.translation, llamaExecutable: path.join(dir, 'no-such-llama-server.exe') }
    }), async s => {
      await s.post('/api/start');
      let item = await s.waitForItem(settled, 30000);
      assert.equal(item.status, 'error');
      assert.match(item.lastError, /could not start llama-server/);

      await s.post('/api/config', { translation: { llamaExecutable: process.execPath } });
      await s.post(`/api/queue/${item.id}/retranslate`);
      await s.post('/api/start');
      item = await s.waitForItem(settled, 30000);
      assert.equal(item.status, 'done', (await s.log(item.id)).slice(-5).join('\n'));
      await s.post('/api/server/shutdown', { confirm: true });
      await s.exited();
    });
  } finally {
    llama.cleanup();
  }
});

test('changing the Base URL port starts llama-server on the new port and stops the old one', { timeout: 2 * MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  const llama = await fakeLlama(dir, { loadMs: 0, loadWait: '5' });
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper', translation: llama.translation
    }), async s => {
      await s.post('/api/start');
      let item = await s.waitForItem(settled, 30000);
      assert.equal(item.status, 'done');
      const pidA = llama.pid();

      const portB = await freePort();
      await s.post('/api/config', { translation: { baseUrl: `http://127.0.0.1:${portB}/v1` } });
      await s.post(`/api/queue/${item.id}/retranslate`);
      await s.post('/api/start');
      item = await s.waitForItem(settled, 30000);
      assert.equal(item.status, 'done', (await s.log(item.id)).slice(-5).join('\n'));
      assert.ok(await answersOn(portB), 'llama-server on the new port');
      await waitFor(() => !pidAlive(pidA), 5000, 'old llama-server to stop');
      await s.post('/api/server/shutdown', { confirm: true });
      await s.exited();
    });
  } finally {
    llama.cleanup();
  }
});

test('a llama-server that is still loading is waited for, not started a second time', { timeout: MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  const llama = await fakeLlama(dir);
  const manual = require('child_process').spawn(process.execPath,
    [path.join(__dirname, 'fake-llama-server.js'), '--port', String(llama.port), '--load-ms', '4000'], { stdio: 'ignore' });
  try {
    await waitFor(() => answersOn(llama.port), 10000, 'manual llama-server to answer');
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper', translation: llama.translation
    }), async s => {
      await s.post('/api/start');
      const item = await s.waitForItem(settled, 30000);
      const log = await s.log(item.id);
      assert.equal(item.status, 'done', log.slice(-5).join('\n'));
      assert.ok(!log.some(l => l.startsWith('Starting llama-server')), 'no second llama-server');
    });
  } finally {
    manual.kill();
    llama.cleanup();
  }
});

test('Shut down stops the llama-server it started even after translation settings change', { timeout: MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  const llama = await fakeLlama(dir, { loadMs: 0 });
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper', translation: llama.translation
    }), async s => {
      await s.post('/api/start');
      assert.equal((await s.waitForItem(settled, 30000)).status, 'done');
      const pid = llama.pid();
      await s.post('/api/config', { translation: { enabled: false } });
      await s.post('/api/server/shutdown', { confirm: true });
      await s.exited();
      await waitFor(() => !pidAlive(pid), 5000, 'llama-server to be stopped');
    });
  } finally {
    llama.cleanup();
  }
});

test('llama-server slower than the load wait → error with a hint, .ja.srt kept', { timeout: MIN }, async () => {
  const dir = makeWatchFolder({ withJa: true });
  const llama = await fakeLlama(dir, { loadMs: 20000, loadWait: '1' });
  try {
    await withServer(baseConfig(dir, {
      backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper', translation: llama.translation
    }), async s => {
      await s.post('/api/start');
      const item = await s.waitForItem(settled, 30000);
      assert.equal(item.status, 'error');
      assert.ok((await s.log(item.id)).some(l => /Model load wait/.test(l)), 'log says which setting to raise');
      assert.equal((await s.get('/api/llama/status')).state, 'loading', 'still loading in the background');
      assert.ok(fs.existsSync(path.join(dir, 'speech.ja.srt')), '.ja.srt kept');
      await s.post('/api/server/shutdown', { confirm: true });
      await s.exited();
    });
  } finally {
    llama.cleanup();
  }
});

// ─── Needs the LLM endpoint (llama-server or OpenRouter) ─────────────────────

test('translation stage: existing .ja.srt → .en.srt, then EN↻ re-translate', { timeout: 10 * MIN }, async t => {
  if (!(await llmReachable())) return t.skip(`LLM endpoint not reachable at ${env.llmUrl}`);
  const dir = makeWatchFolder({ withJa: true });
  await withServer(baseConfig(dir, {
    backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper', translation
  }), async s => {
    await s.post('/api/start');
    const item = await s.waitForItem(settled, 8 * MIN);
    assert.equal(item.status, 'done', (await s.log(item.id)).slice(-5).join('\n'));
    assert.ok((await s.log(item.id)).some(l => l.startsWith('Using existing transcript')), 'skipped transcription');
    const enPath = path.join(dir, 'speech.en.srt');
    assertEnglishTranslation(enPath, path.join(dir, 'speech.ja.srt'));

    // Re-translate overwrites the English file from the same transcript
    fs.writeFileSync(enPath, 'stale');
    assert.equal((await s.post('/api/queue/1/retranslate')).status, 200);
    await s.post('/api/start');
    const again = await s.waitForItem(i => i.status === 'done' && fs.readFileSync(enPath, 'utf8') !== 'stale', 8 * MIN);
    assert.equal(again.status, 'done');
    assertEnglishTranslation(enPath, path.join(dir, 'speech.ja.srt'));
  });
});

test('Stop during translation cancels cleanly and writes no .en.srt', { timeout: 5 * MIN }, async t => {
  if (!(await llmReachable())) return t.skip(`LLM endpoint not reachable at ${env.llmUrl}`);
  const dir = makeWatchFolder({ withJa: true });
  await withServer(baseConfig(dir, {
    backend: 'ctranslate2', whisperExecutable: 'definitely-not-a-real-whisper',
    translation: { ...translation, batchSize: '1' }
  }), async s => {
    await s.post('/api/start');
    await s.waitForItem(i => i.stage === 'translate' && i.segment >= 1, 3 * MIN);
    await s.post('/api/stop');
    const item = await s.waitForItem(settled, 30000);
    assert.equal(item.status, 'error');
    assert.ok((await s.log(item.id)).some(l => /cancelled/.test(l)), 'log says cancelled');
    assert.ok(!fs.existsSync(path.join(dir, 'speech.en.srt')), 'no partial English file');
  });
});

// ─── Needs a transcription backend ───────────────────────────────────────────

test('mlx backend: VAD-segmented transcription matches the fixture', {
  timeout: 10 * MIN, skip: !mlxReady && `mlx venv/model not found (${env.mlxExe}, ${env.mlxModel})`
}, async () => {
  const dir = makeWatchFolder();
  await withServer(baseConfig(dir, mlxBackend()), async s => {
    await s.post('/api/start');
    const item = await s.waitForItem(settled, 8 * MIN);
    assert.equal(item.status, 'done', (await s.log(item.id)).slice(-5).join('\n'));
    assertMatchesFixture(path.join(dir, 'speech.srt'));
    assertTranscriptLogged(await s.log(item.id));
  });
});

test('ctranslate2 backend: VAD-segmented transcription matches the fixture', {
  timeout: 15 * MIN, skip: !ct2Ready && `set WT_TEST_CT2_EXE (whisper-ctranslate2 path) and a CT2 model at ${env.ct2Model}`
}, async () => {
  const dir = makeWatchFolder();
  await withServer(baseConfig(dir, ct2Backend()), async s => {
    await s.post('/api/start');
    const item = await s.waitForItem(settled, 12 * MIN);
    assert.equal(item.status, 'done', (await s.log(item.id)).slice(-5).join('\n'));
    assertMatchesFixture(path.join(dir, 'speech.srt'));
    assertTranscriptLogged(await s.log(item.id));
  });
});

test('a video ffmpeg cannot read fails with ffmpeg\'s reason as the item error', {
  timeout: 5 * MIN, skip: !ct2Ready && 'set WT_TEST_CT2_EXE (whisper-ctranslate2 path) and a CT2 model'
}, async () => {
  const dir = makeWatchFolder();
  fs.writeFileSync(path.join(dir, 'speech.mp4'), 'not a video');
  await withServer(baseConfig(dir, ct2Backend()), async s => {
    await s.post('/api/start');
    const item = await s.waitForItem(settled, 4 * MIN);
    assert.equal(item.status, 'error');
    assert.match(item.lastError, /^ERROR: ffmpeg could not read .*speech\.mp4: .+/);
  });
});

test('full two-stage pipeline: video → .ja.srt → .en.srt', { timeout: 20 * MIN }, async t => {
  if (!mlxReady && !ct2Ready) return t.skip('no transcription backend available');
  if (!(await llmReachable())) return t.skip(`LLM endpoint not reachable at ${env.llmUrl}`);
  const dir = makeWatchFolder();
  await withServer(baseConfig(dir, { ...(mlxReady ? mlxBackend() : ct2Backend()), translation }), async s => {
    await s.post('/api/start');
    const item = await s.waitForItem(settled, 18 * MIN);
    assert.equal(item.status, 'done', (await s.log(item.id)).slice(-5).join('\n'));
    assert.ok(!fs.existsSync(path.join(dir, 'speech.srt')), 'no single-stage video.srt next to the video');
    assertMatchesFixture(path.join(dir, 'speech.ja.srt'));
    assertEnglishTranslation(path.join(dir, 'speech.en.srt'), path.join(dir, 'speech.ja.srt'));
  });
});

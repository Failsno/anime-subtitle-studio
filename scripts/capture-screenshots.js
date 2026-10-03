// Regenerates the README screenshots in docs/screenshots/. Run: npm run screenshots
//
// Runs demo servers (temp configs, a temp watch folder with neutral file names,
// and a fake whisper that streams the test fixture's lines) and captures them
// with headless Chrome via puppeteer-core. Your config.json is never touched.
// Works on macOS, Windows and Linux; finds Chrome (or Edge) in the usual
// places, or set CHROME_PATH.
const puppeteer = require('puppeteer-core');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const OUT = path.join(ROOT, 'docs', 'screenshots');
const FIXTURES = path.join(ROOT, 'test', 'fixtures');
function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const win = [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA].filter(Boolean);
  const candidates = {
    darwin: ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'],
    win32: [...win.map(d => path.join(d, 'Google', 'Chrome', 'Application', 'chrome.exe')),
      ...win.map(d => path.join(d, 'Microsoft', 'Edge', 'Application', 'msedge.exe'))],
    linux: ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser']
  }[process.platform] || [];
  const found = candidates.find(p => fs.existsSync(p));
  if (!found) throw new Error('Chrome/Edge not found; set CHROME_PATH to its executable');
  return found;
}
const CHROME = findChrome();

const sleep = ms => new Promise(r => setTimeout(r, ms));

// Temp workspace: demo videos + subtitles, and a fake whisper
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'wt-shots-'));
const DEMO = path.join(TMP, 'Videos', 'Japanese');

function makeDemo() {
  const files = ['Series A/episode_01', 'Series A/episode_02', 'Series A/episode_03',
    'Series B/clip_01', 'Series B/clip_02', 'Series B/clip_03', 'Series B/clip_04'];
  for (const f of files) {
    fs.mkdirSync(path.dirname(path.join(DEMO, f)), { recursive: true });
    fs.copyFileSync(path.join(FIXTURES, 'speech.mp4'), path.join(DEMO, f + '.mp4'));
  }
  // Two finished items (both subtitle files present)
  for (const f of ['Series A/episode_01', 'Series A/episode_02']) {
    fs.copyFileSync(path.join(FIXTURES, 'speech.ja.srt'), path.join(DEMO, f + '.ja.srt'));
    fs.copyFileSync(path.join(FIXTURES, 'speech.ja.srt'), path.join(DEMO, f + '.en.srt'));
  }
  // Stand-in for whisper: prints the fixture's segments slowly and never
  // finishes. A .js file, which the server runs with Node on any OS.
  const fake = path.join(TMP, 'fake-whisper.js');
  fs.writeFileSync(fake, `
const fs = require('fs');
const blocks = fs.readFileSync(${JSON.stringify(path.join(FIXTURES, 'speech.ja.srt'))}, 'utf8')
  .replace(/\\r/g, '').trim().split(/\\n\\s*\\n/);
const lines = blocks.map(b => {
  const [, time, ...text] = b.split('\\n');
  const [start, end] = time.split(' --> ').map(t => t.replace(',', '.'));
  return '[' + start + ' --> ' + end + '] ' + text.join(' ');
});
console.log('VAD: ' + lines.length + ' speech segments');
lines.forEach((l, i) => setTimeout(() => console.log(l), (i + 1) * 1000));
setTimeout(() => {}, 600000);
`);
  return fake;
}

async function startServer(name, port, config) {
  const cfgPath = path.join(TMP, `${name}.config.json`);
  fs.writeFileSync(cfgPath, JSON.stringify(config, null, 2));
  const proc = spawn(process.execPath, [path.join(ROOT, 'src', 'server.js')], {
    env: { ...process.env, WT_CONFIG: cfgPath, PORT: String(port) }, stdio: ['ignore', 'pipe', 'pipe']
  });
  let out = '';
  proc.stdout.on('data', d => { out += d; });
  for (let i = 0; i < 50 && !out.includes('running at'); i++) await sleep(200);
  return proc;
}

// Show the temp demo folder as it would look in a real home directory
async function prettifyPaths(page) {
  await page.evaluate(root => {
    // Forward slashes on Windows too, so the images look the same from any machine
    const swap = s => s.split(root).join('~/Videos/Japanese').replace(/\\/g, '/');
    for (const el of document.querySelectorAll('input')) if (el.value.includes(root)) el.value = swap(el.value);
    const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walk.nextNode()) if (walk.currentNode.nodeValue.includes(root)) walk.currentNode.nodeValue = swap(walk.currentNode.nodeValue);
    // The Settings demo uses the Mac's paths, which don't exist on every machine
    document.querySelectorAll('.path-mark').forEach(m => m.remove());
  }, DEMO);
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const fake = makeDemo();
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 2 });
  const shot = async (name, target = page) => {
    await prettifyPaths(page);
    await target.screenshot({ path: path.join(OUT, name) });
    console.log('  docs/screenshots/' + name);
  };

  // ── Translate + Concat: a two-stage job in progress ──
  const portA = 3301;
  const a = await startServer('a', portA, {
    watchFolder: DEMO,
    backend: 'ctranslate2',
    whisperExecutable: fake,
    whisperArgs: { model: '~/models/anime-whisper-ct2', vad_segmentation: 'False' },
    mlxArgs: {},
    translation: { enabled: true, baseUrl: 'https://openrouter.ai/api/v1' }
  });
  try {
    await sleep(3500); // let the watcher find the files
    await page.goto(`http://127.0.0.1:${portA}/`, { waitUntil: 'networkidle0' });
    await page.evaluate(() => localStorage.setItem('logPanelHeight', '230'));
    await page.click('#log-header');
    await fetch(`http://127.0.0.1:${portA}/api/start`, { method: 'POST' });
    await sleep(8000); // segments stream into the log
    await shot('translate-queue.png');

    await page.click('#log-header');
    await page.click('#help-header');
    await page.evaluate(() => { document.getElementById('help-body').style.maxHeight = '60vh'; });
    await sleep(400);
    await shot('help-panel.png');
    await page.click('#help-header');

    await page.evaluate(() => switchTab('concat'));
    await page.evaluate(dir => loadConcatFolder(dir), path.join(DEMO, 'Series B'));
    await sleep(800);
    await page.evaluate(() => { toggleConcatFile(0); toggleConcatFile(1); toggleConcatFile(2); });
    await sleep(300);
    await shot('concat.png');
  } finally {
    await fetch(`http://127.0.0.1:${portA}/api/stop`, { method: 'POST' }).catch(() => {});
    a.kill();
  }

  // ── Settings, as configured on the Mac ──
  const portB = 3302;
  const b = await startServer('b', portB, {
    watchFolder: '~/Videos/Japanese',
    backend: 'mlx',
    whisperExecutable: 'whisper-ctranslate2',
    mlxExecutable: '~/mlx-whisper-env/bin/mlx_whisper',
    whisperArgs: {},
    mlxArgs: { model: '~/models/anime-whisper-mlx', task: 'transcribe', initial_prompt: '' },
    translation: { enabled: true, baseUrl: 'http://127.0.0.1:9931/v1' }
  });
  try {
    await page.setViewport({ width: 1280, height: 1000, deviceScaleFactor: 2 });
    await page.goto(`http://127.0.0.1:${portB}/`, { waitUntil: 'networkidle0' });
    await page.evaluate(() => {
      openSettings();
      document.getElementById('settings-modal').style.maxHeight = '96vh';
    });
    const modal = await page.$('#settings-modal');

    // Model picker with the install tooltip open
    await page.evaluate(() => {
      const body = document.getElementById('settings-body');
      body.scrollTop = document.getElementById('section-mlx').offsetTop - body.offsetTop - 120;
      document.querySelector('.mlx-params .info').focus();
    });
    await sleep(400);
    await shot('settings-model.png', modal);

    // Translation section
    await page.evaluate(() => {
      document.activeElement.blur();
      const body = document.getElementById('settings-body');
      const sec = [...document.querySelectorAll('.setting-section')].find(s => /Translation/.test(s.textContent));
      body.scrollTop = sec.offsetTop - body.offsetTop - 12;
    });
    await sleep(300);
    await shot('settings-translation.png', modal);
  } finally {
    b.kill();
    await browser.close();
  }
}

main()
  .catch(e => { console.error(e); process.exitCode = 1; })
  .finally(() => fs.rmSync(TMP, { recursive: true, force: true }));

// Stand-in for llama-server in the auto-start tests: takes the same --host /
// --port flags, answers /health with 503 while "loading" (--load-ms), then
// translates every "[n] text" line of a request into "[n] English line n".
const http = require('http');
const fs = require('fs');

const argv = process.argv.slice(2);
const flag = (name, def) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : def; };
const port = parseInt(flag('--port'), 10);
const loadMs = parseInt(flag('--load-ms', '0'), 10);
const pidFile = flag('--pid-file');
const readyAt = Date.now() + loadMs;

if (pidFile) fs.writeFileSync(pidFile, String(process.pid));
console.log(`fake llama-server: loading model (${loadMs} ms)`);

http.createServer((req, res) => {
  const reply = (status, obj) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(obj)); };
  if (Date.now() < readyAt) return reply(503, { error: { message: 'Loading model' } });
  if (req.url === '/health') return reply(200, { status: 'ok' });
  if (req.url === '/props') return reply(200, { model_path: flag('-m', '/models/fake-model-Q4_K_M.gguf'), default_generation_settings: { n_ctx: 4096 } });
  if (req.url === '/v1/chat/completions') {
    let body = '';
    req.on('data', d => { body += d; });
    req.on('end', () => {
      const user = JSON.parse(body).messages.find(m => m.role === 'user').content;
      const block = user.split('Translate these lines:\n')[1].split('\n\n')[0];
      const lines = block.split('\n').map(l => l.match(/^\[(\d+)\]/)).filter(Boolean).map(m => `[${m[1]}] English line ${m[1]}`);
      reply(200, { choices: [{ message: { content: lines.join('\n') } }] });
    });
    return;
  }
  reply(404, { error: 'not found' });
}).listen(port, flag('--host', '127.0.0.1'));

const { app, BrowserWindow, dialog, shell, session } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const net = require('node:net');
let backend, origin, window;
const root = app.isPackaged
  ? path.join(process.resourcesPath, 'backend')
  : path.join(__dirname, '..');
const dataDirectory =
  process.env.MUA_DATA_DIR ||
  (app.isPackaged ? path.join(app.getPath('userData'), 'workspace') : root);

async function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
  });
}
async function startBackend() {
  const port = await freePort();
  origin = 'http://127.0.0.1:' + port;
  fs.mkdirSync(dataDirectory, { recursive: true });
  const log = fs.createWriteStream(path.join(dataDirectory, 'desktop-backend.log'), { flags: 'a' });
  const python =
    process.env.MUA_PYTHON ||
    (fs.existsSync(path.join(root, '.venv', 'Scripts', 'python.exe'))
      ? path.join(root, '.venv', 'Scripts', 'python.exe')
      : fs.existsSync(path.join(root, '.venv', 'bin', 'python'))
        ? path.join(root, '.venv', 'bin', 'python')
        : process.platform === 'win32'
          ? 'python'
          : 'python3');
  let startupError;
  backend = spawn(
    python,
    ['-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', String(port)],
    {
      cwd: root,
      windowsHide: true,
      env: { ...process.env, MUA_DATA_DIR: dataDirectory, PYTHONUNBUFFERED: '1' },
    },
  );
  backend.stdout.pipe(log);
  backend.stderr.pipe(log);
  backend.on('error', (error) => (startupError = error));
  backend.on('exit', (code) => {
    if (code !== 0) startupError = new Error('Python backend stopped with code ' + code);
  });
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    if (startupError) throw startupError;
    try {
      const response = await fetch(origin + '/api/health', { signal: AbortSignal.timeout(1500) });
      const data = await response.json();
      if (data.service === 'mua') return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  throw new Error('The Python backend did not become ready in 90 seconds.');
}
function openWindow() {
  window = new BrowserWindow({
    width: 1440,
    height: 1000,
    minWidth: 760,
    minHeight: 640,
    backgroundColor: '#f8f9f7',
    title: 'MUA',
    icon: path.join(__dirname, 'icon.png'),
    show: false,
    autoHideMenuBar: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true },
  });
  window.once('ready-to-show', () => {
    if (process.env.MUA_SMOKE_TEST !== '1') window.show();
  });
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://')) shell.openExternal(url);
    return { action: 'deny' };
  });
  window.webContents.on('will-navigate', (event, url) => {
    if (new URL(url).origin !== origin) {
      event.preventDefault();
      if (url.startsWith('https://')) shell.openExternal(url);
    }
  });
  window.loadURL(origin);
}
app.whenReady().then(async () => {
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) =>
    callback(false),
  );
  try {
    await startBackend();
    openWindow();
  } catch (error) {
    const detail =
      'Install Python 3.11+ and the project requirements, or set MUA_PYTHON to your Python executable.\n\n' +
      error.message +
      '\n\nLog: ' +
      path.join(dataDirectory, 'desktop-backend.log');
    if (process.env.MUA_SMOKE_TEST === '1') console.error(detail);
    else dialog.showErrorBox('MUA could not start', detail);
    app.quit();
  }
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => {
  if (backend && !backend.killed) backend.kill();
});
app.on('activate', () => {
  if (!BrowserWindow.getAllWindows().length && origin) openWindow();
});

const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
function browserPath() {
  if (process.env.MUA_CHROMIUM) return process.env.MUA_CHROMIUM;
  if (fs.existsSync(chromium.executablePath())) return chromium.executablePath();
  const base = path.join(process.env.LOCALAPPDATA || '', 'ms-playwright');
  if (fs.existsSync(base))
    for (const name of fs
      .readdirSync(base)
      .filter((n) => /^chromium-\d+$/.test(n))
      .sort()
      .reverse()) {
      const exe = path.join(base, name, 'chrome-win64', 'chrome.exe');
      if (fs.existsSync(exe)) return exe;
    }
  throw new Error(
    'Install the test browser with npx playwright install chromium, or set MUA_CHROMIUM.',
  );
}
module.exports = { chromium, browserPath };

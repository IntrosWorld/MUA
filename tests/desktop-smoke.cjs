const { _electron } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
(async () => {
  const root = path.join(__dirname, '..');
  const env = { ...process.env, MUA_SMOKE_TEST: '1' };
  delete env.ELECTRON_RUN_AS_NODE;
  const application = await _electron.launch({
    executablePath: process.env.MUA_DESKTOP_EXE || require('electron'),
    args: process.env.MUA_DESKTOP_EXE ? [] : [root],
    cwd: root,
    env,
    timeout: 120000,
  });
  try {
    const page = await application.firstWindow({ timeout: 120000 });
    await page.waitForSelector('[data-command]');
    assert.equal(await page.evaluate(() => typeof require), 'undefined');
    assert.equal(await page.evaluate(() => typeof process), 'undefined');
    const prefs = await application.evaluate(({ BrowserWindow }) =>
      BrowserWindow.getAllWindows()[0].webContents.getLastWebPreferences(),
    );
    assert.equal(prefs.contextIsolation, true);
    assert.equal(prefs.sandbox, true);
    assert.equal(prefs.nodeIntegration, false);
    await page.click('[data-nav="workspaces"]');
    await page.waitForSelector('[data-tab="6"]');
    assert.equal(await page.locator('.hero-name').count(), 1);
    console.log('Electron startup, owned backend, navigation, and renderer isolation passed.');
  } finally {
    await application.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

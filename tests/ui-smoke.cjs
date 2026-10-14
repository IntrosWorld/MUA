const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, browserPath } = require('./browser.cjs');
const base = process.env.MUA_TEST_URL || 'http://127.0.0.1:8765';
const results = path.join(__dirname, '..', 'test-results');
(async () => {
  fs.mkdirSync(results, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: browserPath() });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const fulfill = (route, data) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify(data) });
  try {
    await page.goto(base);
    await page.waitForSelector('[data-command]');
    await page.screenshot({ path: path.join(results, 'studio-light.png'), fullPage: true });
    await page.click('#g-speakers [data-v="3"]');
    assert.equal(await page.locator('[data-command]').count(), 3);
    await page.click('[data-source="upload"]');
    assert.equal(await page.locator('#runBtn').isDisabled(), true);
    await page.setInputFiles(
      '#fileInput',
      path.join(__dirname, '..', 'data', 'mixed', 'demo_mixed.wav'),
    );
    assert.equal(await page.locator('#runBtn').isDisabled(), false);
    await page.click('[data-source="demo"]');
    await page.route('**/process_demo', (route) =>
      fulfill(route, {
        results: [
          {
            identified_speaker: 'Mohit',
            speaker_confidence: 0.73,
            transcript: '<img src=x onerror="window.injected=true">',
            intent: 'get_schedule',
            is_command: true,
            response: 'Your math class is at 10.',
            response_source: 'llm',
            llm_model: 'test-model',
          },
        ],
        metrics: { audio_duration: 4.2, processing_time: 1.1, xRT: 0.26, wer: 0.12 },
      }),
    );
    await page.click('#runBtn');
    await page.waitForSelector('.result-card');
    assert.equal(await page.evaluate(() => window.injected), undefined);
    assert.equal(await page.locator('.result-transcript img').count(), 0);
    const downloadPromise = page.waitForEvent('download');
    await page.click('#exportBtn');
    assert.equal((await downloadPromise).suggestedFilename(), 'mua-session.json');
    await page.click('[data-nav="people"]');
    await page.waitForSelector('#peopleView:not([hidden])');
    await page.route('**/identify_voice', (route) =>
      fulfill(route, { identified_speaker: 'Mohit', confidence: 0.71 }),
    );
    await page.setInputFiles(
      '#testVoiceFile',
      path.join(__dirname, '..', 'data', 'mixed', 'demo_mixed.wav'),
    );
    await page.locator('#testForm button[type="submit"]').click();
    await page.waitForFunction(() =>
      document.getElementById('voiceTestResult').textContent.includes('Mohit'),
    );
    await page.screenshot({ path: path.join(results, 'people-light.png'), fullPage: true });
    await page.goto(base + '/profiles-page');
    await page.waitForSelector('[data-tab="6"]');
    const fixture = await (await page.request.get(base + '/profiles/mohit')).json();
    await page.route('**/profiles/mohit', (route) => fulfill(route, fixture));
    await page.route('**/profiles/mohit/workspace/**', (route) => {
      const section = new URL(route.request().url()).pathname.split('/').pop();
      fixture.workspace[section] = route.request().postDataJSON().data;
      return fulfill(route, { status: 'ok' });
    });
    await page.route('**/profiles/mohit/notes', (route) => {
      fixture.workspace.notes.push({
        text: route.request().postDataJSON().text,
        date: 'Test date',
      });
      return fulfill(route, { status: 'ok' });
    });
    await page.locator('[data-user="mohit"]').click();
    for (let tab = 0; tab < 7; tab++) {
      await page.click('[data-tab="' + tab + '"]');
      await page.waitForSelector('#panel-' + tab);
    }
    await page.click('[data-tab="4"]');
    await page.fill('[data-form="note"] input', '<img src=x onerror="window.injected=true">');
    await page.locator('[data-form="note"] button').click();
    await page.waitForFunction(() => document.querySelector('.note-text') !== null);
    assert.equal(await page.locator('.note-text img').count(), 0);
    assert.equal(await page.locator('#panel-4').count(), 1);
    let turns = [];
    await page.route('**/llm/chat', (route) => {
      turns.push(route.request().postDataJSON());
      return fulfill(route, {
        response: 'Hello <script>unsafe</script>',
        model: 'mock-local',
        error: null,
      });
    });
    await page.click('[data-tab="6"]');
    await page.fill('#chatInput', 'My schedule?');
    await page.click('#chatSend');
    await page.waitForFunction(() => document.querySelectorAll('.chat-meta').length === 1);
    await page.fill('#chatInput', 'And my reminders?');
    await page.click('#chatSend');
    await page.waitForFunction(() => document.querySelectorAll('.chat-meta').length === 2);
    assert.equal(turns[1].history.length, 2);
    assert.equal(await page.locator('#chatBox script').count(), 0);
    await page.click('#addUserBtn');
    assert.equal(await page.locator('#addUserModal').evaluate((el) => el.open), true);
    await page.keyboard.press('Escape');
    await page.click('[data-tab="0"]');
    await page.screenshot({ path: path.join(results, 'workspace-light.png'), fullPage: true });
    await page.goto(base + '/models-page');
    await page.waitForFunction(() =>
      document.getElementById('runtimeStatus').textContent.startsWith('Connected'),
    );
    assert((await page.locator('#modelSelect option').count()) > 1);
    let saved = false;
    await page.route('**/llm/config', (route) => {
      saved = true;
      return fulfill(route, { status: 'ok' });
    });
    await page.route('**/llm/mode', (route) => fulfill(route, { status: 'ok' }));
    await page.click('#saveModels');
    await page.waitForFunction(
      () => document.getElementById('saveHint').textContent === 'Configuration saved.',
    );
    assert(saved);
    await page.screenshot({ path: path.join(results, 'models-light.png'), fullPage: true });
    for (const url of ['/', '/profiles-page', '/models-page'])
      for (const width of [390, 800, 1440]) {
        await page.setViewportSize({ width, height: 1000 });
        await page.goto(base + url);
        await page.waitForTimeout(350);
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth + 1,
        );
        assert.equal(overflow, false, url + ' overflows at ' + width);
        for (const theme of ['light', 'dark']) {
          await page.evaluate((theme) => {
            MUA.theme(theme);
            localStorage.setItem('mua-theme', theme);
          }, theme);
          await page.screenshot({
            path: path.join(
              results,
              url === '/'
                ? 'studio-' + theme + '-' + width + '.png'
                : url.slice(1) + '-' + theme + '-' + width + '.png',
            ),
            fullPage: true,
          });
        }
      }
    assert.deepEqual(errors, []);
    console.log(
      'UI flows, upload, report export, identification, workspace edits, chat history, model settings, escaping, both themes, and 3 viewport sizes passed.',
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

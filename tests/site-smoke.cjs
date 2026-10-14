const assert = require('node:assert/strict');
const path = require('node:path');
const { chromium, browserPath } = require('./browser.cjs');
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: browserPath() });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  try {
    for (const width of [390, 800, 1440]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto('http://127.0.0.1:8767');
      await page.waitForFunction(() =>
        [...document.images].every((img) => img.complete && img.naturalWidth > 0),
      );
      assert.equal(
        await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1),
        false,
      );
      for (const theme of ['light', 'dark']) {
        await page.evaluate((theme) => setTheme(theme), theme);
        await page.screenshot({
          path: path.join(__dirname, '../test-results', 'website-' + theme + '-' + width + '.png'),
          fullPage: true,
          animations: 'disabled',
        });
      }
    }
    await page.click('[data-speaker="shakti"]');
    assert((await page.locator('#exampleContext').textContent()).includes('Shakti'));
    await page.locator('.faq summary').first().click();
    assert.equal(
      await page
        .locator('.faq details')
        .first()
        .evaluate((el) => el.open),
      true,
    );
    const brokenAnchors = await page.evaluate(() =>
      [...document.querySelectorAll('a[href^="#"]')]
        .map((a) => a.getAttribute('href'))
        .filter((href) => href !== '#' && !document.getElementById(href.slice(1))),
    );
    assert.deepEqual(brokenAnchors, []);
    assert.deepEqual(errors, []);
    console.log(
      'Website examples, FAQ, anchors, images, both themes, and three viewport sizes passed.',
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

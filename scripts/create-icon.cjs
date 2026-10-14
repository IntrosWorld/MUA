const { chromium, browserPath } = require('../tests/browser.cjs');
(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: browserPath() });
  const page = await browser.newPage({ viewport: { width: 256, height: 256 } });
  await page.setContent(
    '<style>body{margin:0}.logo{width:256px;height:256px;background:#236349;display:flex;align-items:center;justify-content:center;gap:12px}.logo i{width:16px;background:#f8f9f7;border-radius:8px}</style><div class="logo">' +
      [64, 100, 136, 100, 64].map((height) => '<i style="height:' + height + 'px"></i>').join('') +
      '</div>',
  );
  await page.screenshot({ path: 'desktop/icon.png' });
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

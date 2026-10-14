const fs = require('node:fs');
const path = require('node:path');
const root = path.join(__dirname, '..');
const assets = path.join(root, 'frontend', 'assets');
fs.mkdirSync(path.join(assets, 'icons'), { recursive: true });
fs.mkdirSync(path.join(assets, 'fonts'), { recursive: true });
const icons = [
  'waveform',
  'users',
  'squares-four',
  'sliders-horizontal',
  'arrow-up-right',
  'arrow-right',
  'plus',
  'upload-simple',
  'play',
  'microphone',
  'moon',
  'sun',
  'check',
  'x',
  'trash',
  'cpu',
  'arrow-clockwise',
  'download-simple',
  'lightbulb',
  'fan',
  'thermometer',
  'television',
  'chat-circle',
  'calendar-blank',
  'note',
  'music-notes',
  'film-slate',
  'heartbeat',
  'shield-check',
];
for (const name of icons)
  fs.copyFileSync(
    path.join(root, 'node_modules', '@phosphor-icons', 'core', 'assets', 'regular', name + '.svg'),
    path.join(assets, 'icons', name + '.svg'),
  );
for (const weight of [400, 500, 600, 700])
  fs.copyFileSync(
    path.join(
      root,
      'node_modules',
      '@fontsource',
      'geist',
      'files',
      `geist-latin-${weight}-normal.woff2`,
    ),
    path.join(assets, 'fonts', `geist-${weight}.woff2`),
  );
fs.copyFileSync(
  path.join(root, 'node_modules', '@phosphor-icons', 'core', 'LICENSE'),
  path.join(assets, 'icons', 'LICENSE'),
);
fs.copyFileSync(
  path.join(root, 'node_modules', '@fontsource', 'geist', 'LICENSE'),
  path.join(assets, 'fonts', 'LICENSE'),
);
console.log('Copied self-hosted Geist fonts and Phosphor icons.');

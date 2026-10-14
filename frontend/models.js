let config = {},
  provider = 'ollama',
  discovery = 0;
const $ = (id) => document.getElementById(id);
function providerFields() {
  provider = $('provider').value;
  const local = provider === 'ollama' || provider === 'local-api',
    prefix = provider === 'ollama' ? 'ollama' : 'local_api';
  $('localFields').hidden = !local;
  $('legacyFields').hidden = provider !== 'offline';
  $('cloudFields').hidden = provider !== 'gemini';
  $('runtimeUrl').required = local;
  $('modelSelect').required = local;
  if (local) {
    $('runtimeUrl').value = config[prefix + '_url'];
    $('modelSelect').innerHTML =
      '<option value="' +
      MUA.escape(config[prefix + '_model']) +
      '">' +
      MUA.escape(config[prefix + '_model'] || 'Choose a model') +
      '</option>';
    refreshModels();
  }
}
async function refreshModels() {
  const version = ++discovery,
    requestedProvider = provider;
  if (!['ollama', 'local-api'].includes(provider)) return;
  $('runtimeStatus').textContent = 'Checking the local runtime…';
  $('refreshModels').disabled = true;
  try {
    const d = await MUA.api(
      '/llm/models?provider=' +
        encodeURIComponent(provider) +
        '&base_url=' +
        encodeURIComponent($('runtimeUrl').value),
    );
    if (version !== discovery || provider !== requestedProvider) return;
    const selected = $('modelSelect').value;
    $('modelSelect').innerHTML =
      '<option value="">Choose a model</option>' +
      d.models
        .map(
          (m) => '<option value="' + MUA.escape(m.name) + '">' + MUA.escape(m.name) + '</option>',
        )
        .join('');
    if (d.models.some((m) => m.name === selected)) $('modelSelect').value = selected;
    else if (d.models.length) $('modelSelect').value = d.models[0].name;
    $('runtimeStatus').textContent =
      'Connected / ' + d.models.length + ' installed model' + (d.models.length === 1 ? '' : 's');
    $('availableModels').innerHTML = d.models
      .map(
        (m) =>
          '<div class="model-row"><span>' +
          MUA.escape(m.name) +
          '</span><small>' +
          MUA.escape(m.parameters || '') +
          (m.size ? ' / ' + (m.size / 1e9).toFixed(1) + ' GB' : '') +
          '</small></div>',
      )
      .join('');
    if (!d.models.length)
      $('runtimeStatus').textContent =
        'Connected, but no models are installed. Pull a model in Ollama or load one in your local server.';
  } catch (error) {
    if (version === discovery) {
      $('runtimeStatus').textContent = error.message;
      $('availableModels').innerHTML = '';
    }
  } finally {
    if (version === discovery) $('refreshModels').disabled = false;
  }
}
document.addEventListener('DOMContentLoaded', async () => {
  $('provider').addEventListener('change', providerFields);
  $('refreshModels').addEventListener('click', refreshModels);
  $('modelForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    $('saveModels').disabled = true;
    try {
      if (provider === 'ollama' || provider === 'local-api') {
        const prefix = provider === 'ollama' ? 'ollama' : 'local_api';
        if (!$('modelSelect').value) throw new Error('Select an installed model.');
        const update = {
          [prefix + '_url']: $('runtimeUrl').value,
          [prefix + '_model']: $('modelSelect').value,
        };
        await MUA.api('/llm/config', MUA.json('POST', update));
        Object.assign(config, update);
      }
      if (provider === 'gemini' && $('geminiKey').value.trim()) {
        await MUA.api('/llm/gemini_key', MUA.json('POST', { key: $('geminiKey').value.trim() }));
        $('geminiKey').value = '';
      }
      await MUA.api('/llm/mode', MUA.json('POST', { mode: provider }));
      $('saveHint').textContent = 'Configuration saved.';
      MUA.notify('Response engine updated.');
      document.querySelector('[data-model-label]').textContent =
        provider === 'ollama'
          ? config.ollama_model
          : provider === 'local-api'
            ? config.local_api_model
            : provider === 'offline'
              ? 'Local weights'
              : 'Gemini';
    } catch (error) {
      MUA.notify(error.message, true);
    } finally {
      $('saveModels').disabled = false;
    }
  });
  try {
    const d = await MUA.api('/llm/status');
    config = d.local_config;
    $('provider').value = d.mode;
    providerFields();
  } catch (error) {
    MUA.notify(error.message, true);
  }
});

let commands = {};
let source = 'demo',
  uploadedFile = null,
  audioObjectUrl = null,
  report = null,
  busy = false;
const $ = (id) => document.getElementById(id);
const e = (value) => MUA.escape(value);
const selection = (id) => document.querySelector('#' + id + ' .active')?.dataset.v;
function buildCommands() {
  const prior = Object.fromEntries(
    [...document.querySelectorAll('[data-command]')].map((el) => [el.dataset.command, el.value]),
  );
  $('cmdGrid').innerHTML = Object.entries(commands)
    .slice(0, Number(selection('g-speakers')))
    .map(
      ([name, items]) =>
        '<div class="command-row"><div class="avatar" aria-hidden="true">' +
        e(name[0].toUpperCase()) +
        '</div><div><label for="cmd-' +
        e(name) +
        '">' +
        e(name[0].toUpperCase() + name.slice(1)) +
        '</label><select id="cmd-' +
        e(name) +
        '" data-command="' +
        e(name) +
        '">' +
        items
          .map(
            (text, i) =>
              '<option value="' +
              i +
              '"' +
              (String(i) === (prior[name] || '0') ? ' selected' : '') +
              '>' +
              e(text) +
              '</option>',
          )
          .join('') +
        '</select></div></div>',
    )
    .join('');
}
function formData() {
  const fd = new FormData();
  for (const [key, value] of Object.entries({
    num_speakers: selection('g-speakers'),
    overlap_type: selection('g-overlap'),
    noise_level: selection('g-noise'),
    noise_type: selection('g-ntype'),
    demo_oracle: source === 'upload' ? 'false' : selection('g-oracle'),
    sep_model: $('sepModel').value,
    cmd_indices: [...document.querySelectorAll('[data-command]')].map((el) => el.value).join(','),
  }))
    fd.append(key, value);
  if (source === 'upload') fd.append('file', uploadedFile);
  return fd;
}
function updateSource(next) {
  source = next;
  document.querySelectorAll('[data-source]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.source === source);
    btn.setAttribute('aria-selected', btn.dataset.source === source);
  });
  $('demoSource').hidden = source !== 'demo';
  $('uploadSource').hidden = source !== 'upload';
  $('demoControls').hidden = source !== 'demo';
  $('g-oracle').querySelector('[data-v="true"]').disabled = source === 'upload';
  if (source === 'upload') $('g-oracle').querySelector('[data-v="false"]').click();
  $('runBtn').innerHTML =
    MUA.icon('waveform') + (source === 'demo' ? 'Run conversation' : 'Process recording');
  $('runBtn').disabled = source === 'upload' && !uploadedFile;
  $('mixedSection').hidden = true;
  if (source === 'upload' && uploadedFile) showUploadPreview();
}
function showUploadPreview() {
  if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  audioObjectUrl = URL.createObjectURL(uploadedFile);
  $('mixedPlayer').src = audioObjectUrl;
  $('mixedSection').hidden = false;
  $('previewLabel').textContent = 'Your recording';
}
function chooseFile(file) {
  if (!file) return;
  if (!(file.type.startsWith('audio/') || /\.(wav|mp3|flac|ogg|m4a|webm)$/i.test(file.name)))
    return MUA.notify('Choose an audio recording.', true);
  uploadedFile = file;
  $('fileSummary').textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
  $('fileSummary').hidden = false;
  $('runBtn').disabled = false;
  showUploadPreview();
}
async function session(action) {
  if (busy) return;
  if (action === 'run' && source === 'upload' && !uploadedFile)
    return MUA.notify('Choose an audio file first.', true);
  busy = true;
  $('statusBar').classList.remove('hidden');
  $('statusText').textContent =
    action === 'preview'
      ? 'Preparing the conversation audio…'
      : 'Separating voices and preparing personal responses. This can take a few minutes.';
  const controls = [...$('studioView').querySelectorAll('button,input,select')];
  const states = controls.map((el) => el.disabled);
  controls.forEach((el) => (el.disabled = true));
  try {
    const data = await MUA.api(
      action === 'preview'
        ? '/generate_demo'
        : source === 'demo'
          ? '/process_demo'
          : '/process_audio',
      { method: 'POST', body: formData() },
    );
    if (action === 'preview') {
      $('mixedPlayer').src = data.audio_url + '?t=' + Date.now();
      $('mixedSection').hidden = false;
      $('previewLabel').textContent = 'Conversation preview';
    } else renderResults(data);
    MUA.notify(action === 'preview' ? 'Conversation ready to preview.' : 'Session complete.');
  } catch (error) {
    MUA.notify(error.message, true);
  } finally {
    busy = false;
    controls.forEach((el, i) => (el.disabled = states[i]));
    $('statusBar').classList.add('hidden');
  }
}
function renderResults(data) {
  report = data;
  $('resultsEmpty').hidden = true;
  $('exportBtn').hidden = false;
  $('resultsGrid').innerHTML = (data.results || [])
    .map((r) => {
      const name = r.identified_speaker || 'Unknown';
      const confidence = Number(r.speaker_confidence);
      const warnings = [
        r.speaker_fallback ? 'Reference identity was used instead of a voice match.' : '',
        r.asr_fallback ? r.asr_warning || 'Reference transcript used.' : '',
      ].filter(Boolean);
      const audio =
        r.audio_url && r.audio_url.startsWith('/audio/')
          ? '<audio controls src="' + e(r.audio_url) + '"></audio>'
          : '';
      return (
        '<article class="panel result-card"><div class="result-person"><div class="avatar">' +
        e(name[0].toUpperCase()) +
        '</div><div><h3>' +
        e(name) +
        '</h3><p>' +
        (Number.isFinite(confidence)
          ? (confidence * 100).toFixed(1) + '% voice match'
          : 'Voice match unavailable') +
        '</p></div></div><p class="result-transcript">' +
        e(r.transcript || 'No speech detected.') +
        '</p><span class="tag accent">' +
        e((r.intent || 'unknown').replaceAll('_', ' ')) +
        '</span> <span class="tag">' +
        (r.is_command ? 'Command detected' : 'Conversation') +
        '</span><div class="response">' +
        e(r.response) +
        '</div><p class="small muted" style="margin-bottom:12px">' +
        e(r.response_source === 'llm' ? 'Response from ' + r.llm_model : 'Rule-based response') +
        '</p>' +
        audio +
        warnings.map((w) => '<p class="warning">' + e(w) + '</p>').join('') +
        '</article>'
      );
    })
    .join('');
  if (!data.results?.length) {
    $('resultsEmpty').hidden = false;
    $('resultsEmpty').querySelector('p').textContent =
      'No speech streams were found. Try a clearer recording.';
  }
  const m = data.metrics || {},
    fmt = (value, unit = '') => (value == null ? 'Unavailable' : Number(value).toFixed(2) + unit);
  $('metricsStrip').hidden = false;
  $('metricsStrip').innerHTML = [
    ['Audio duration', fmt(m.audio_duration, ' s')],
    ['Processing time', fmt(m.processing_time, ' s')],
    ['Real-time factor', fmt(m.xRT, '×')],
    ['Word error rate', m.wer == null ? 'Unavailable' : (m.wer * 100).toFixed(1) + '%'],
    ['Signal quality / SI-SNR', fmt(m.si_snr, ' dB')],
    ['Separation model', data.separation_model || data.separation_mode || 'Unavailable'],
    ['Model parameters', m.model_params || 'Unavailable'],
  ]
    .map(
      ([label, value]) =>
        '<div class="metric' +
        (label === 'Separation model' || label === 'Model parameters' ? ' metric-text' : '') +
        '"><strong>' +
        e(value) +
        '</strong><span>' +
        label +
        '</span></div>',
    )
    .join('');
  if (data.separation_warning) MUA.notify(data.separation_warning, true);
  $('separationNote').hidden = !data.separation_warning;
  $('separationNote').textContent = data.separation_warning || '';
  $('resultsGrid').scrollIntoView({
    behavior: matchMedia('(prefers-reduced-motion:reduce)').matches ? 'instant' : 'smooth',
    block: 'start',
  });
}
async function loadSpeakers() {
  try {
    const data = await MUA.api('/speakers');
    $('speakerList').innerHTML =
      (data.speakers || [])
        .map(
          (p) =>
            '<div class="speaker-row"><div class="avatar">' +
            e(p.name[0].toUpperCase()) +
            '</div><div><h3>' +
            e(p.name) +
            '</h3><p>' +
            (p.type === 'real' ? 'Enrolled voice' : 'Demo voice') +
            (p.fallback ? ' / fallback embedding' : '') +
            '</p></div><button class="icon-btn" data-delete="' +
            e(p.name) +
            '" aria-label="Remove ' +
            e(p.name) +
            '">' +
            MUA.icon('trash') +
            '</button></div>',
        )
        .join('') ||
      '<p class="small muted">No voices enrolled yet. Add your first voice here.</p>';
  } catch (error) {
    $('speakerList').textContent = error.message;
  }
}
function route() {
  const people = location.hash === '#people';
  $('studioView').hidden = people;
  $('peopleView').hidden = !people;
  $('breadcrumb').textContent = people ? 'People' : 'Voice studio';
  document
    .querySelectorAll('[data-nav]')
    .forEach((el) =>
      el.classList.toggle('active', el.dataset.nav === (people ? 'people' : 'studio')),
    );
  document.title = (people ? 'People' : 'Voice studio') + ' | MUA';
  if (people) loadSpeakers();
}
document.addEventListener('DOMContentLoaded', async () => {
  document.querySelectorAll('.segmented').forEach((group) => {
    group.setAttribute(
      'aria-label',
      group.closest('.control')?.querySelector('.control-label')?.textContent || 'Session option',
    );
    group.addEventListener('click', (event) => {
      const btn = event.target.closest('.pill');
      if (!btn || btn.disabled) return;
      group.querySelectorAll('.pill').forEach((el) => {
        el.classList.toggle('active', el === btn);
        el.setAttribute('aria-pressed', el === btn);
      });
      if (group.id === 'g-speakers') buildCommands();
      if (group.id === 'g-oracle') {
        $('row-sep-model').hidden = btn.dataset.v === 'true';
        $('oracleHelp').hidden = btn.dataset.v !== 'true';
      }
    });
  });
  document
    .querySelectorAll('[data-source]')
    .forEach((btn) => btn.addEventListener('click', () => updateSource(btn.dataset.source)));
  $('generateBtn').addEventListener('click', () => session('preview'));
  $('runBtn').addEventListener('click', () => session('run'));
  $('fileInput').addEventListener('change', (event) => chooseFile(event.target.files[0]));
  $('dropzone').addEventListener('click', () => $('fileInput').click());
  $('dropzone').addEventListener('dragover', (event) => {
    event.preventDefault();
    $('dropzone').classList.add('dragging');
  });
  $('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('dragging'));
  $('dropzone').addEventListener('drop', (event) => {
    event.preventDefault();
    $('dropzone').classList.remove('dragging');
    chooseFile(event.dataTransfer.files[0]);
  });
  $('exportBtn').addEventListener('click', () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mua-session.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $('enrollForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const files = $('enrollFiles').files;
    if (files.length > 5) return MUA.notify('Choose up to 5 recordings.', true);
    const btn = event.submitter;
    btn.disabled = true;
    btn.textContent = 'Learning voice…';
    try {
      const fd = new FormData();
      fd.append('name', $('enrollName').value.trim());
      for (const file of files) fd.append('files', file);
      await MUA.api('/enroll_speaker', { method: 'POST', body: fd });
      MUA.notify('Voice enrolled.');
      event.target.reset();
      await loadSpeakers();
    } catch (error) {
      MUA.notify(error.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Enroll voice';
    }
  });
  $('testForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const btn = event.submitter;
    btn.disabled = true;
    try {
      const fd = new FormData();
      fd.append('file', $('testVoiceFile').files[0]);
      const d = await MUA.api('/identify_voice', { method: 'POST', body: fd });
      $('voiceTestResult').hidden = false;
      $('voiceTestResult').textContent =
        (d.identified_speaker || 'Unknown') +
        ' / ' +
        (Number(d.confidence || 0) * 100).toFixed(1) +
        '% match';
    } catch (error) {
      MUA.notify(error.message, true);
    } finally {
      btn.disabled = false;
    }
  });
  let deleteName = '';
  $('speakerList').addEventListener('click', (event) => {
    const btn = event.target.closest('[data-delete]');
    if (!btn) return;
    deleteName = btn.dataset.delete;
    $('deleteDescription').textContent = 'MUA will stop matching recordings to ' + deleteName + '.';
    $('deleteDialog').showModal();
  });
  $('cancelDelete').addEventListener('click', () => $('deleteDialog').close());
  $('confirmDelete').addEventListener('click', async () => {
    try {
      await MUA.api('/speakers/' + encodeURIComponent(deleteName), { method: 'DELETE' });
      $('deleteDialog').close();
      await loadSpeakers();
      MUA.notify('Voice removed.');
    } catch (error) {
      MUA.notify(error.message, true);
    }
  });
  $('refreshPeople').addEventListener('click', loadSpeakers);
  window.addEventListener('hashchange', route);
  route();
  try {
    commands = await MUA.api('/commands');
    buildCommands();
  } catch (error) {
    MUA.notify(error.message, true);
    $('runBtn').disabled = true;
  }
});

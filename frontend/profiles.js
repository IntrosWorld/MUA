let currentUser = null,
  currentProfile = null,
  activeTab = 0,
  llmMode = 'ollama';
const chats = new Map(),
  pendingChats = new Set();
const $ = (id) => document.getElementById(id),
  e = (value) => MUA.escape(value);
const tabs = [
  ['Calendar', 'calendar-blank'],
  ['Fitness', 'heartbeat'],
  ['Movies', 'film-slate'],
  ['Music', 'music-notes'],
  ['Notes', 'note'],
  ['Smart home', 'lightbulb'],
  ['Assistant', 'chat-circle'],
];
const endpoint = (suffix = '') => '/profiles/' + encodeURIComponent(currentUser) + suffix;
const button = (text, action, extra = '') =>
  '<button class="btn" data-action="' + action + '" ' + extra + '>' + text + '</button>';
async function loadProfiles() {
  const d = await MUA.api('/profiles');
  $('profileList').innerHTML = (d.profiles || [])
    .map(
      (p) =>
        '<button class="profile-chip' +
        (p.name.toLowerCase() === currentUser ? ' active' : '') +
        '" data-user="' +
        e(p.name.toLowerCase()) +
        '"><span class="avatar">' +
        e(p.display_name[0]) +
        '</span><span><span class="chip-name">' +
        e(p.display_name) +
        '</span><br><span class="chip-type">' +
        (p.type === 'tts' ? 'Demo workspace' : 'Personal workspace') +
        '</span></span></button>',
    )
    .join('');
  if (!currentUser && d.profiles?.length) await openProfile(d.profiles[0].name.toLowerCase());
}
async function openProfile(name) {
  currentUser = name;
  const profile = await MUA.api('/profiles/' + encodeURIComponent(name));
  if (currentUser !== name) return;
  currentProfile = profile;
  renderProfile();
  document.querySelectorAll('[data-user]').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.user === currentUser);
    btn.setAttribute('aria-pressed', btn.dataset.user === currentUser);
  });
}
function renderProfile() {
  const p = currentProfile,
    ws = p.workspace || {};
  $('profileMain').innerHTML =
    '<div class="profile-hero"><div class="avatar">' +
    e(p.display_name[0]) +
    '</div><div><h2 class="hero-name">' +
    e(p.display_name) +
    '</h2><p class="hero-sub">' +
    e(p.room) +
    ' / ' +
    (p.type === 'tts' ? 'Demo profile' : 'Personal profile') +
    '</p></div></div><div class="hero-stats">' +
    [
      [(ws.calendar || []).filter((x) => !x.done).length, 'Upcoming events'],
      [(ws.reminders || []).filter((x) => !x.done).length, 'Open reminders'],
      [Number(ws.fitness?.steps_today || 0).toLocaleString(), 'Steps logged'],
    ]
      .map(
        ([v, l]) =>
          '<div class="stat-item"><div class="stat-val">' +
          e(v) +
          '</div><div class="stat-label">' +
          l +
          '</div></div>',
      )
      .join('') +
    '</div><div class="ws-tabs" role="tablist" aria-label="Workspace sections">' +
    tabs
      .map(
        ([t, i], n) =>
          '<button class="ws-tab' +
          (activeTab === n ? ' active' : '') +
          '" role="tab" aria-selected="' +
          (activeTab === n) +
          '" id="tab-' +
          n +
          '" aria-controls="panel-' +
          n +
          '" data-tab="' +
          n +
          '">' +
          MUA.icon(i) +
          t +
          '</button>',
      )
      .join('') +
    '</div><div class="panel ws-panel" role="tabpanel" id="panel-' +
    activeTab +
    '" aria-labelledby="tab-' +
    activeTab +
    '">' +
    renderPanel(ws) +
    '</div>';
  if (activeTab === 6) restoreChat();
}
function renderPanel(ws) {
  switch (activeTab) {
    case 0:
      return (
        '<h2>Schedule</h2><div class="event-list">' +
        (ws.calendar || [])
          .map(
            (ev) =>
              '<div class="event-item' +
              (ev.done ? ' event-done' : '') +
              '"><button class="check-btn' +
              (ev.done ? ' done' : '') +
              '" data-action="event" data-id="' +
              e(ev.id) +
              '" aria-label="' +
              e((ev.done ? 'Reopen ' : 'Complete ') + ev.event) +
              '" aria-pressed="' +
              !!ev.done +
              '">' +
              MUA.icon('check') +
              '</button><span class="event-time">' +
              e(ev.time) +
              '</span><span class="event-name">' +
              e(ev.event) +
              '</span><span class="tag">' +
              e(ev.category || 'personal') +
              '</span></div>',
          )
          .join('') +
        (!(ws.calendar || []).length ? '<p class="small muted">Nothing scheduled yet.</p>' : '') +
        '<form class="add-form" data-form="calendar"><input type="time" name="time" aria-label="Event time" value="09:00" required style="max-width:120px"><input name="event" aria-label="Event name" placeholder="Add an event" required maxlength="200"><select name="category" aria-label="Event category" style="max-width:130px"><option>personal</option><option>academic</option><option>work</option><option>fitness</option><option>club</option></select><button class="btn" type="submit">Add event</button></form><h3 style="margin-top:28px">Reminders</h3>' +
        (ws.reminders || [])
          .map(
            (r) =>
              '<div class="reminder"><button class="check-btn' +
              (r.done ? ' done' : '') +
              '" data-action="reminder" data-id="' +
              e(r.id) +
              '" aria-pressed="' +
              !!r.done +
              '" aria-label="' +
              e((r.done ? 'Reopen ' : 'Complete ') + r.text) +
              '">' +
              MUA.icon('check') +
              '</button><span>' +
              e(r.text) +
              '</span></div>',
          )
          .join('') +
        '<form class="add-form" data-form="reminder"><input name="text" aria-label="Reminder" placeholder="Remember to…" required maxlength="500"><button class="btn">Add reminder</button></form>'
      );
    case 1: {
      const f = ws.fitness || {};
      return (
        '<h2>Fitness</h2><p class="small muted" style="margin-bottom:20px">Goal: ' +
        e(f.goal || 'No goal set') +
        '</p><div class="fitness-grid">' +
        [
          [Number(f.steps_today || 0).toLocaleString(), 'Steps'],
          [f.calories_burned || 0, 'Calories'],
          [(f.water_ml || 0) + ' ml', 'Water'],
          [(f.weight_kg || '–') + ' kg', 'Weight'],
        ]
          .map(
            ([v, l]) =>
              '<div class="fitness-stat"><div class="fs-val">' +
              e(v) +
              '</div><div class="fs-label">' +
              l +
              '</div></div>',
          )
          .join('') +
        '</div><h3>Logged workouts</h3>' +
        (f.workouts || [])
          .map(
            (w) =>
              '<div class="workout-item"><strong>' +
              e(w.type) +
              '</strong><span>' +
              e(w.duration_min) +
              ' min</span><span>' +
              e(w.calories) +
              ' cal</span></div>',
          )
          .join('') +
        (!(f.workouts || []).length
          ? '<p class="small muted" style="margin-top:12px">No workouts logged.</p>'
          : '')
      );
    }
    case 2: {
      const m = ws.movies || {};
      return (
        '<h2>Watchlist</h2><div class="movie-grid">' +
        (m.watchlist || []).map((v) => '<div class="movie-chip">' + e(v) + '</div>').join('') +
        '</div>' +
        empty(m.watchlist, 'Add a film you want to watch.') +
        '<form class="add-form" data-form="movie"><input name="title" aria-label="Movie title" placeholder="Movie title" required maxlength="200"><button class="btn">Add film</button></form><h3 style="margin-top:28px">Watched</h3><div class="movie-grid">' +
        (m.watched || [])
          .map(
            (v) =>
              '<div class="movie-chip">' +
              e(v.title) +
              (v.rating ? ' / ' + e(v.rating) + '/10' : '') +
              '</div>',
          )
          .join('') +
        '</div>'
      );
    }
    case 3: {
      const m = ws.music || {};
      return (
        '<h2>Your music</h2>' +
        (m.currently_playing
          ? '<p class="small muted">Selected: ' + e(m.currently_playing) + '</p>'
          : '') +
        (m.playlists || [])
          .map(
            (p) =>
              '<div class="playlist-card"><h3 class="playlist-name">' +
              e(p.name) +
              '</h3><p class="track-list">' +
              (p.tracks || []).map(e).join(' / ') +
              '</p></div>',
          )
          .join('') +
        empty(m.playlists, 'Create a playlist for your day.') +
        '<form class="add-form" data-form="playlist"><input name="name" aria-label="Playlist name" placeholder="New playlist name" required maxlength="200"><button class="btn">Add playlist</button></form>'
      );
    }
    case 4:
      return (
        '<h2>Your notes</h2>' +
        (ws.notes || [])
          .map(
            (n) =>
              '<div class="note-card"><p class="note-text">' +
              e(n.text) +
              '</p><p class="note-date">' +
              e(n.date) +
              (n.pinned ? ' / Pinned' : '') +
              '</p></div>',
          )
          .join('') +
        empty(ws.notes, 'A place for things you want to remember.') +
        '<form class="add-form" data-form="note"><input name="text" aria-label="New note" placeholder="Write a note…" required maxlength="4000"><button class="btn">Save note</button></form>'
      );
    case 5: {
      const sh = ws.smart_home || {};
      return (
        '<h2>Room controls</h2><p class="small muted" style="margin-bottom:20px">Simulated devices in ' +
        e(currentProfile.room) +
        '.</p><div class="sh-grid">' +
        [
          ['bedroom_light', 'Light', 'lightbulb'],
          ['fan', 'Fan', 'fan'],
          ['ac', 'Air conditioning', 'thermometer'],
          ['tv', 'Television', 'television'],
        ]
          .map(
            ([key, label, icon]) =>
              '<button class="sh-card' +
              (sh[key] ? ' on' : '') +
              '" data-action="device" data-device="' +
              key +
              '" aria-pressed="' +
              !!sh[key] +
              '">' +
              MUA.icon(icon) +
              '<span><strong>' +
              label +
              '</strong><span class="sh-state">' +
              (sh[key] ? 'On' : 'Off') +
              '</span></span><span class="sh-switch" aria-hidden="true"></span></button>',
          )
          .join('') +
        '</div><label class="temperature" for="temperature">Temperature<input id="temperature" type="range" min="16" max="30" value="' +
        Number(sh.ac_temp || 24) +
        '"><output id="temperatureValue">' +
        Number(sh.ac_temp || 24) +
        '°C</output></label>'
      );
    }
    case 6:
      return (
        '<div class="chat-context"><span>Assistant for ' +
        e(currentProfile.display_name) +
        '<br>Uses this person’s schedule and preferences.</span><a class="btn btn-quiet" href="/models-page">Model settings ' +
        MUA.icon('arrow-up-right') +
        '</a></div><div class="chat-box" id="chatBox" role="log" aria-label="Conversation"></div><div class="chat-prompts">' +
        ['What is on my schedule?', 'Suggest a playlist', 'Summarize my reminders']
          .map((t) => button(e(t), 'prompt', 'data-prompt="' + e(t) + '"'))
          .join('') +
        '</div><form class="chat-input-row" data-form="chat"><input id="chatInput" name="message" aria-label="Message your assistant" placeholder="Ask about your day…" required maxlength="8000"><button class="btn btn-primary" id="chatSend">Send ' +
        MUA.icon('arrow-right') +
        '</button></form><p class="small muted" style="margin-top:10px">Answers can be inaccurate. Device controls are available in Smart home.</p>'
      );
  }
}
function empty(items, text) {
  return items?.length ? '' : '<p class="small muted">' + text + '</p>';
}
function restoreChat() {
  const history = chats.get(currentUser) || [];
  $('chatBox').innerHTML = history.length
    ? history
        .map(
          (m) =>
            '<div class="chat-msg ' +
            (m.role === 'user' ? 'chat-user' : 'chat-ai') +
            '">' +
            e(m.content) +
            (m.model ? '<span class="chat-meta">' + e(m.model) + '</span>' : '') +
            '</div>',
        )
        .join('')
    : '<div class="chat-msg chat-ai">Hi ' +
      e(currentProfile.display_name) +
      '. Ask me about your schedule, reminders, or playlists.</div>';
  if (pendingChats.has(currentUser))
    $('chatBox').innerHTML += '<div class="chat-msg chat-ai">Thinking…</div>';
  $('chatSend').disabled = pendingChats.has(currentUser);
  $('chatBox').scrollTop = $('chatBox').scrollHeight;
}
async function sendChat(message) {
  const name = currentUser;
  if (pendingChats.has(name)) return;
  const history = chats.get(name) || [],
    prior = history.slice(-20).map(({ role, content }) => ({ role, content }));
  history.push({ role: 'user', content: message });
  chats.set(name, history);
  pendingChats.add(name);
  $('chatInput').value = '';
  restoreChat();
  try {
    const status = await MUA.api('/llm/status');
    llmMode = status.mode;
    const d = await MUA.api(
      '/llm/chat',
      MUA.json('POST', { name, message, mode: llmMode, history: prior }),
    );
    if (d.error) throw new Error(d.error);
    history.push({ role: 'assistant', content: d.response, model: d.model });
  } catch (error) {
    history.pop();
    MUA.notify(error.message, true);
    if (currentUser === name && activeTab === 6) $('chatInput').value = message;
  } finally {
    pendingChats.delete(name);
    if (currentUser === name && activeTab === 6) restoreChat();
  }
}
async function saveSection(section, data) {
  await MUA.api(endpoint('/workspace/' + section), MUA.json('PUT', { data }));
  await openProfile(currentUser);
}
document.addEventListener('DOMContentLoaded', async () => {
  $('profileList').addEventListener('click', async (event) => {
    const btn = event.target.closest('[data-user]');
    if (btn)
      try {
        await openProfile(btn.dataset.user);
      } catch (error) {
        MUA.notify(error.message, true);
      }
  });
  $('profileMain').addEventListener('click', async (event) => {
    const tab = event.target.closest('[data-tab]');
    if (tab) {
      activeTab = Number(tab.dataset.tab);
      renderProfile();
      document.getElementById('tab-' + activeTab)?.focus();
      return;
    }
    const btn = event.target.closest('[data-action]');
    if (!btn) return;
    try {
      const action = btn.dataset.action,
        ws = currentProfile.workspace;
      btn.disabled = true;
      if (action === 'event' || action === 'reminder') {
        const section = action === 'event' ? 'calendar' : 'reminders';
        const items = structuredClone(ws[section] || []);
        const item = items.find((x) => x.id === btn.dataset.id);
        if (item) item.done = !item.done;
        await saveSection(section, items);
      }
      if (action === 'device') {
        const key = btn.dataset.device;
        await MUA.api(
          endpoint('/smart_home/' + key),
          MUA.json('POST', { state: !ws.smart_home[key] }),
        );
        await openProfile(currentUser);
      }
      if (action === 'prompt') {
        await sendChat(btn.dataset.prompt);
      }
    } catch (error) {
      MUA.notify(error.message, true);
    } finally {
      btn.disabled = false;
    }
  });
  $('profileMain').addEventListener('submit', async (event) => {
    event.preventDefault();
    const type = event.target.dataset.form;
    if (!type) return;
    const data = Object.fromEntries(new FormData(event.target)),
      btn = event.submitter;
    if (type === 'chat') {
      await sendChat(data.message.trim());
      return;
    }
    btn.disabled = true;
    try {
      const ws = structuredClone(currentProfile.workspace);
      if (type === 'calendar') {
        const now = new Date(),
          date =
            now.getFullYear() +
            '-' +
            String(now.getMonth() + 1).padStart(2, '0') +
            '-' +
            String(now.getDate()).padStart(2, '0');
        await MUA.api(endpoint('/calendar'), MUA.json('POST', { ...data, date, done: false }));
      }
      if (type === 'reminder')
        await MUA.api(endpoint('/reminders'), MUA.json('POST', { text: data.text }));
      if (type === 'note') await MUA.api(endpoint('/notes'), MUA.json('POST', { text: data.text }));
      if (type === 'movie') {
        ws.movies = ws.movies || { watchlist: [], watched: [] };
        ws.movies.watchlist = [...(ws.movies.watchlist || []), data.title];
        await saveSection('movies', ws.movies);
      }
      if (type === 'playlist') {
        ws.music = ws.music || { playlists: [] };
        ws.music.playlists = [...(ws.music.playlists || []), { name: data.name, tracks: [] }];
        await saveSection('music', ws.music);
      }
      await openProfile(currentUser);
      MUA.notify('Workspace updated.');
    } catch (error) {
      MUA.notify(error.message, true);
    } finally {
      btn.disabled = false;
    }
  });
  $('profileMain').addEventListener('input', (event) => {
    if (event.target.id === 'temperature')
      $('temperatureValue').textContent = event.target.value + '°C';
  });
  $('profileMain').addEventListener('change', async (event) => {
    if (event.target.id === 'temperature')
      try {
        await MUA.api(
          endpoint('/smart_home/ac_temp'),
          MUA.json('POST', { state: Number(event.target.value) }),
        );
        currentProfile.workspace.smart_home.ac_temp = Number(event.target.value);
      } catch (error) {
        MUA.notify(error.message, true);
      }
  });
  $('addUserBtn').addEventListener('click', () => $('addUserModal').showModal());
  $('cancelAddUser').addEventListener('click', () => $('addUserModal').close());
  $('addUserForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const name = $('newUserName').value.trim(),
      key = name.toLowerCase().replace(/\s+/g, '-');
    if (!/^[a-z0-9][a-z0-9-]{0,49}$/.test(key))
      return MUA.notify('Use letters, numbers, spaces, or hyphens for the name.', true);
    const btn = event.submitter;
    btn.disabled = true;
    try {
      await MUA.api(
        '/profiles/' + encodeURIComponent(key) + '/create',
        MUA.json('POST', {
          display_name: name,
          room: $('newUserRoom').value.trim() || name + '’s room',
          avatar_color: '#236349',
        }),
      );
      $('addUserModal').close();
      event.target.reset();
      await loadProfiles();
      await openProfile(key);
      MUA.notify('Workspace created.');
    } catch (error) {
      MUA.notify(error.message, true);
    } finally {
      btn.disabled = false;
    }
  });
  try {
    await loadProfiles();
  } catch (error) {
    $('profileList').textContent = error.message;
  }
});

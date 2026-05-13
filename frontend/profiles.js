let currentUser = null;
let currentProfile = null;
let llmMode = "offline";

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  await loadProfiles();
  await loadLLMStatus();
});

async function loadLLMStatus() {
  try {
    const r = await fetch("/llm/status");
    const d = await r.json();
    llmMode = d.mode || "offline";
    updateLLMButtons();
  } catch(e) {}
}

function updateLLMButtons() {
  document.getElementById("btnOffline").classList.toggle("active", llmMode === "offline");
  document.getElementById("btnGemini").classList.toggle("active", llmMode === "gemini");
}

async function setLLMMode(mode) {
  llmMode = mode;
  updateLLMButtons();
  await fetch("/llm/mode", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({mode}) });
}

// ── Profile list ────────────────────────────────────────────────────────────
async function loadProfiles() {
  const r  = await fetch("/profiles");
  const d  = await r.json();
  const el = document.getElementById("profileList");
  el.innerHTML = "";
  (d.profiles || []).forEach(p => {
    const chip = document.createElement("div");
    chip.className = "profile-chip";
    chip.id = `chip-${p.name.toLowerCase()}`;
    chip.onclick = () => openProfile(p.name.toLowerCase());
    chip.innerHTML = `
      <div class="chip-avatar" style="background:${p.avatar_color}22;font-size:1.1rem;">${p.avatar_emoji}</div>
      <div>
        <div class="chip-name">${p.display_name}</div>
        <div class="chip-type">${p.type === "tts" ? "TTS voice" : "Real user"}</div>
      </div>`;
    el.appendChild(chip);
  });
}

// ── Open profile ────────────────────────────────────────────────────────────
async function openProfile(name) {
  currentUser = name;
  document.querySelectorAll(".profile-chip").forEach(c => c.classList.remove("active"));
  const chip = document.getElementById(`chip-${name}`);
  if (chip) chip.classList.add("active");

  const r = await fetch(`/profiles/${name}`);
  currentProfile = await r.json();
  renderProfile(currentProfile);
}

function renderProfile(p) {
  const ws = p.workspace || {};
  const sh = ws.smart_home || {};

  const eventsToday = (ws.calendar || []).filter(e => !e.done).length;
  const pendingRem  = (ws.reminders || []).filter(r => !r.done).length;

  document.getElementById("profileMain").innerHTML = `
    <!-- Hero -->
    <div class="profile-hero">
      <div class="hero-avatar" style="background:${p.avatar_color}33;">${p.avatar_emoji}</div>
      <div>
        <div class="hero-name" style="background:linear-gradient(90deg,${p.avatar_color},#f9a8d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">${p.display_name}</div>
        <div class="hero-sub">${p.room || ""} &nbsp;·&nbsp; ${p.type === "tts" ? "TTS Voice" : "Real User"}</div>
      </div>
      <div class="hero-stats">
        <div class="stat-item"><div class="stat-val">${eventsToday}</div><div class="stat-label">Events today</div></div>
        <div class="stat-item"><div class="stat-val">${pendingRem}</div><div class="stat-label">Reminders</div></div>
        <div class="stat-item"><div class="stat-val">${ws.fitness?.steps_today?.toLocaleString() || 0}</div><div class="stat-label">Steps</div></div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="ws-tabs">
      ${["Calendar","Fitness","Movies","Music","Notes","Smart Home","AI Chat"].map((t,i) =>
        `<button class="ws-tab${i===0?' active':''}" onclick="switchTab(${i},this)">${t}</button>`
      ).join("")}
    </div>

    <!-- Panels -->
    ${renderCalendar(ws.calendar||[], p.name)}
    ${renderFitness(ws.fitness||{}, p.name)}
    ${renderMovies(ws.movies||{}, p.name)}
    ${renderMusic(ws.music||{}, p.name)}
    ${renderNotes(ws.notes||[], p.name)}
    ${renderSmartHome(sh, p.name)}
    ${renderAIChat(p.name, p.display_name)}
  `;
}

function switchTab(idx, btn) {
  document.querySelectorAll(".ws-tab").forEach((t,i) => t.classList.toggle("active", i===idx));
  document.querySelectorAll(".ws-panel").forEach((p,i) => p.classList.toggle("active", i===idx));
}

// ── Calendar ────────────────────────────────────────────────────────────────
function renderCalendar(events, uname) {
  const rows = events.map(e => `
    <div class="event-item${e.done?' event-done':''}" id="evt-${e.id}">
      <button class="check-btn${e.done?' done':''}" onclick="toggleEvent('${uname}','${e.id}',${!e.done})">&#10003;</button>
      <span class="event-time">${e.time}</span>
      <span class="event-name">${e.event}</span>
      <span class="event-cat cat-${e.category||'personal'}">${e.category||'personal'}</span>
    </div>`).join("") || '<div style="color:var(--text-3);font-size:0.85rem;">No events today.</div>';

  return `<div class="ws-panel active" id="panel-0">
    <div class="event-list">${rows}</div>
    <div class="add-form">
      <input class="add-input" id="cal-time" type="time" value="09:00" style="max-width:100px;"/>
      <input class="add-input" id="cal-name" placeholder="Event name" style="flex:2;"/>
      <select class="add-input" id="cal-cat" style="max-width:120px;">
        <option value="personal">Personal</option>
        <option value="academic">Academic</option>
        <option value="work">Work</option>
        <option value="fitness">Fitness</option>
        <option value="club">Club</option>
      </select>
      <button class="add-btn" onclick="addEvent('${uname}')">+ Add</button>
    </div>
  </div>`;
}

async function toggleEvent(uname, id, done) {
  const p = await (await fetch(`/profiles/${uname}`)).json();
  const ev = p.workspace.calendar.find(e => e.id === id);
  if (ev) ev.done = done;
  await fetch(`/profiles/${uname}/calendar`, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({...p.workspace.calendar.find(e=>e.id===id)})});
  // Full re-save via workspace section
  await fetch(`/profiles/${uname}/workspace/calendar`, {method:"PUT", headers:{"Content-Type":"application/json"}, body: JSON.stringify({data: p.workspace.calendar})});
  document.getElementById(`evt-${id}`)?.classList.toggle("event-done", done);
  const btn = document.querySelector(`#evt-${id} .check-btn`);
  if (btn) btn.classList.toggle("done", done);
}

async function addEvent(uname) {
  const time = document.getElementById("cal-time")?.value;
  const name = document.getElementById("cal-name")?.value?.trim();
  const cat  = document.getElementById("cal-cat")?.value;
  if (!name) return;
  const today = new Date().toISOString().split("T")[0];
  await fetch(`/profiles/${uname}/calendar`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({time, event: name, category: cat, date: today, done: false})
  });
  await openProfile(uname);
}

// ── Fitness ─────────────────────────────────────────────────────────────────
function renderFitness(fit, uname) {
  const stats = [
    {label:"Steps",    val: fit.steps_today?.toLocaleString()||0},
    {label:"Calories", val: fit.calories_burned||0},
    {label:"Water",    val: `${fit.water_ml||0}ml`},
    {label:"Weight",   val: `${fit.weight_kg||"-"}kg`},
  ];
  const statsHtml = stats.map(s => `
    <div class="fitness-stat">
      <div class="fs-val">${s.val}</div>
      <div class="fs-label">${s.label}</div>
    </div>`).join("");

  const workouts = (fit.workouts||[]).map(w => `
    <div class="workout-item">
      <span class="w-type">${w.type}</span>
      <span class="w-dur">${w.duration_min} min</span>
      <span class="w-cal">${w.calories} cal</span>
      <span style="font-size:0.75rem;color:var(--text-3);margin-left:0.5rem;">${w.notes||""}</span>
    </div>`).join("");

  return `<div class="ws-panel" id="panel-1">
    <div style="font-size:0.78rem;color:var(--violet);font-weight:600;margin-bottom:0.75rem;">Goal: ${fit.goal||"Not set"}</div>
    <div class="fitness-grid">${statsHtml}</div>
    <div class="card-title" style="margin-bottom:0.5rem;">Today's Workouts</div>
    ${workouts || '<div style="color:var(--text-3);font-size:0.85rem;">No workouts logged today.</div>'}
  </div>`;
}

// ── Movies ──────────────────────────────────────────────────────────────────
function renderMovies(mov, uname) {
  const watchlist = (mov.watchlist||[]).map(m => `<span class="movie-chip watchlist">${m}</span>`).join("");
  const watched   = (mov.watched||[]).map(m => `<span class="movie-chip watched">${m.title} <span class="movie-rating">${m.rating ? m.rating + "/10" : ""}</span></span>`).join("");

  return `<div class="ws-panel" id="panel-2">
    <div class="movie-section-title">Watchlist (${(mov.watchlist||[]).length})</div>
    <div class="movie-grid">${watchlist || '<span style="color:var(--text-3);font-size:0.82rem;">Empty</span>'}</div>
    <div class="add-form">
      <input class="add-input" id="mov-name" placeholder="Add to watchlist…"/>
      <button class="add-btn" onclick="addMovie('${uname}')">+ Add</button>
    </div>
    <div class="movie-section-title" style="margin-top:1rem;">Watched (${(mov.watched||[]).length})</div>
    <div class="movie-grid">${watched || '<span style="color:var(--text-3);font-size:0.82rem;">None yet</span>'}</div>
  </div>`;
}

async function addMovie(uname) {
  const title = document.getElementById("mov-name")?.value?.trim();
  if (!title) return;
  const p = await (await fetch(`/profiles/${uname}`)).json();
  p.workspace.movies.watchlist = [...(p.workspace.movies.watchlist||[]), title];
  await fetch(`/profiles/${uname}/workspace/movies`, {method:"PUT", headers:{"Content-Type":"application/json"}, body: JSON.stringify({data: p.workspace.movies})});
  await openProfile(uname);
}

// ── Music ───────────────────────────────────────────────────────────────────
function renderMusic(mus, uname) {
  const np = mus.currently_playing ? `<div class="now-playing">▶ ${mus.currently_playing}</div>` : "";
  const playlists = (mus.playlists||[]).map(pl => `
    <div class="playlist-card">
      <div class="playlist-name">${pl.name}</div>
      <div class="track-list">${pl.tracks.map(t=>`<span class="track-chip">${t}</span>`).join("")}</div>
    </div>`).join("");

  return `<div class="ws-panel" id="panel-3">
    ${np}
    <div class="card-title" style="margin-bottom:0.75rem;">Playlists</div>
    ${playlists || '<div style="color:var(--text-3);font-size:0.85rem;">No playlists.</div>'}
    <div class="add-form">
      <input class="add-input" id="mus-name" placeholder="New playlist name…"/>
      <button class="add-btn" onclick="addPlaylist('${uname}')">+ Playlist</button>
    </div>
  </div>`;
}

async function addPlaylist(uname) {
  const name = document.getElementById("mus-name")?.value?.trim();
  if (!name) return;
  const p = await (await fetch(`/profiles/${uname}`)).json();
  p.workspace.music.playlists.push({name, tracks:[]});
  await fetch(`/profiles/${uname}/workspace/music`, {method:"PUT", headers:{"Content-Type":"application/json"}, body: JSON.stringify({data: p.workspace.music})});
  await openProfile(uname);
}

// ── Notes ───────────────────────────────────────────────────────────────────
function renderNotes(notes, uname) {
  const rows = notes.map(n => `
    <div class="note-card">
      <span class="note-pin${n.pinned?' pinned':''}"></span>
      <div style="flex:1;">
        <div class="note-text">${n.text}</div>
        <div class="note-date">${n.date}</div>
      </div>
    </div>`).join("");

  return `<div class="ws-panel" id="panel-4">
    ${rows || '<div style="color:var(--text-3);font-size:0.85rem;">No notes.</div>'}
    <div class="add-form">
      <input class="add-input" id="note-text" placeholder="Write a note…"/>
      <button class="add-btn" onclick="addNote('${uname}')">+ Note</button>
    </div>
  </div>`;
}

async function addNote(uname) {
  const text = document.getElementById("note-text")?.value?.trim();
  if (!text) return;
  await fetch(`/profiles/${uname}/notes`, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({text})});
  await openProfile(uname);
}

// ── Smart Home ──────────────────────────────────────────────────────────────
const SH_DEVICES = [
  {key:"bedroom_light", name:"Light"},
  {key:"fan",           name:"Fan"},
  {key:"ac",            name:"AC"},
  {key:"tv",            name:"TV"},
];

function renderSmartHome(sh, uname) {
  const cards = SH_DEVICES.map(d => `
    <div class="sh-card ${sh[d.key]?'on':'off'}" id="sh-${d.key}" onclick="toggleDevice('${uname}','${d.key}')">
      <div class="sh-name">${d.name}</div>
      <div class="sh-state">${sh[d.key]?"ON":"OFF"}</div>
    </div>`).join("");

  return `<div class="ws-panel" id="panel-5">
    <div class="sh-grid">${cards}</div>
    ${sh.ac_temp !== undefined ? `
    <div style="margin-top:1rem;display:flex;align-items:center;gap:0.85rem;">
      <span style="font-size:0.85rem;color:var(--text-2);">AC Temperature</span>
      <input type="range" min="16" max="30" value="${sh.ac_temp}" oninput="updateACTemp('${uname}',this.value)"
        style="flex:1;accent-color:var(--cyan);" />
      <span id="ac-temp-val" style="font-family:monospace;color:var(--cyan);font-weight:700;">${sh.ac_temp}°C</span>
    </div>` : ""}
  </div>`;
}

async function toggleDevice(uname, device) {
  const p  = await (await fetch(`/profiles/${uname}`)).json();
  const sh = p.workspace.smart_home;
  sh[device] = !sh[device];
  await fetch(`/profiles/${uname}/smart_home/${device}`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({state: sh[device]})
  });
  const card = document.getElementById(`sh-${device}`);
  if (card) {
    card.classList.toggle("on", sh[device]);
    card.classList.toggle("off", !sh[device]);
    card.querySelector(".sh-state").textContent = sh[device] ? "ON" : "OFF";
  }
}

async function updateACTemp(uname, val) {
  document.getElementById("ac-temp-val").textContent = `${val}°C`;
  const p = await (await fetch(`/profiles/${uname}`)).json();
  p.workspace.smart_home.ac_temp = parseInt(val);
  await fetch(`/profiles/${uname}/smart_home/ac_temp`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({state: parseInt(val)})
  });
}

// ── AI Chat ──────────────────────────────────────────────────────────────────
function renderAIChat(uname, displayName) {
  return `<div class="ws-panel" id="panel-6">
    <div style="font-size:0.78rem;color:var(--text-3);margin-bottom:0.75rem;">
      Chat with the AI as <strong style="color:var(--violet);">${displayName}</strong> —
      it knows your full profile (schedule, fitness, movies, music, smart home).
      Mode: <span id="chat-mode-label" style="color:var(--cyan);">${llmMode}</span>
    </div>
    <div class="chat-box" id="chatBox-${uname}">
      <div class="chat-msg chat-ai">Hi ${displayName}! I know your schedule, fitness data, playlists and more. Ask me anything!</div>
    </div>
    <div class="chat-input-row">
      <input class="chat-input" id="chatInput-${uname}" placeholder="Ask anything about your day, schedule, playlists…"
        onkeydown="if(event.key==='Enter')sendChat('${uname}')"/>
      <button class="chat-send" id="chatSend-${uname}" onclick="sendChat('${uname}')">Send</button>
    </div>
  </div>`;
}

async function sendChat(uname) {
  const input  = document.getElementById(`chatInput-${uname}`);
  const box    = document.getElementById(`chatBox-${uname}`);
  const sendBtn= document.getElementById(`chatSend-${uname}`);
  const msg    = input?.value?.trim();
  if (!msg) return;

  // Add user bubble
  box.innerHTML += `<div class="chat-msg chat-user">${msg}</div>`;
  input.value = "";
  sendBtn.disabled = true;
  box.innerHTML += `<div class="chat-msg chat-ai" id="chat-thinking">⏳ Thinking…</div>`;
  box.scrollTop = box.scrollHeight;

  try {
    const r = await fetch("/llm/chat", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({name: uname, message: msg, mode: llmMode})
    });
    const d = await r.json();
    document.getElementById("chat-thinking")?.remove();
    const resp = d.response || d.error || "No response.";
    const meta = `${d.mode||llmMode} · ${d.model||""}`;
    box.innerHTML += `
      <div>
        <div class="chat-msg chat-ai">${resp}</div>
        <div class="chat-meta">${meta}</div>
      </div>`;
  } catch(e) {
    document.getElementById("chat-thinking")?.remove();
    box.innerHTML += `<div class="chat-msg chat-ai" style="color:#f87171;">Error: ${e.message}</div>`;
  }
  box.scrollTop = box.scrollHeight;
  sendBtn.disabled = false;
}

// ── Gemini key modal ────────────────────────────────────────────────────────
function showGeminiKeyModal() { document.getElementById("geminiModal").classList.remove("hidden"); }
function closeModal()         { document.getElementById("geminiModal").classList.add("hidden"); }
async function saveGeminiKey() {
  const key = document.getElementById("geminiKeyInput").value.trim();
  if (!key) return;
  await fetch("/llm/gemini_key", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({key})});
  closeModal();
  alert("Gemini API key saved!");
}

// ── Add user modal ──────────────────────────────────────────────────────────
function showAddUser()  { document.getElementById("addUserModal").classList.remove("hidden"); }
function closeAddUser() { document.getElementById("addUserModal").classList.add("hidden"); }
async function createUser() {
  const name  = document.getElementById("newUserName").value.trim();
  const room  = document.getElementById("newUserRoom").value.trim();
  const color = document.getElementById("newUserColor").value;
  if (!name) { alert("Enter a name."); return; }
  await fetch(`/profiles/${name.toLowerCase()}/create`, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({display_name: name, room: room || `${name}'s room`, avatar_color: color})
  });
  closeAddUser();
  await loadProfiles();
  await openProfile(name.toLowerCase());
}

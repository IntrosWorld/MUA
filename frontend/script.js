// ── Data ──
const COMMANDS = {
  mohit: [
    "Hey assistant, turn on my bedroom light.",
    "Hey assistant, what is my schedule today?",
    "Hey assistant, turn off the fan.",
  ],
  rahul: [
    "Hey assistant, play my study playlist.",
    "Hey assistant, set an alarm for seven AM.",
    "Hey assistant, turn on the AC.",
  ],
  priya: [
    "Hey assistant, remind me to call home.",
    "Hey assistant, play relaxing music.",
    "Hey assistant, turn off the room light.",
  ],
};

const SPEAKER_EMOJI = { mohit: "M", rahul: "R", priya: "P" };
const AV_CLASS = ["av-0", "av-1", "av-2"];

let uploadedFile = null;

// ── Pill selector ──
function pick(btn, groupId) {
  document.querySelectorAll(`#${groupId} .pill`).forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  if (groupId === "g-speakers") buildCmdGrid();
}

function val(groupId) {
  const el = document.querySelector(`#${groupId} .pill.active`);
  return el ? el.dataset.v : null;
}

// ── Command grid ──
function buildCmdGrid() {
  const n = parseInt(val("g-speakers")) || 2;
  const speakers = ["mohit", "rahul", "priya"].slice(0, n);
  const grid = document.getElementById("cmdGrid");
  grid.innerHTML = "";
  speakers.forEach(spk => {
    const item = document.createElement("div");
    item.className = "cmd-item";
    item.innerHTML = `
      <div class="cmd-item-label">
        ${spk.charAt(0).toUpperCase() + spk.slice(1)}
      </div>
      <select id="cmd_${spk}">
        ${COMMANDS[spk].map((c, i) => `<option value="${i}">${c}</option>`).join("")}
      </select>`;
    grid.appendChild(item);
  });
}

// ── Status helpers ──
function showStatus(msg) {
  document.getElementById("statusText").textContent = msg;
  document.getElementById("statusBar").classList.remove("hidden");
}
function hideStatus() {
  document.getElementById("statusBar").classList.add("hidden");
}

// ── Toggle sep-model row visibility ──
function toggleSepModel() {
  const isOracle = val("g-oracle") === "true";
  const row = document.getElementById("row-sep-model");
  if (row) row.style.opacity = isOracle ? "0.35" : "1";
}

// ── Build FormData from current UI state ──
function buildFD(extra = {}) {
  const fd = new FormData();
  fd.append("num_speakers", val("g-speakers") || "2");
  fd.append("overlap_type", val("g-overlap") || "partial");
  fd.append("noise_level",  val("g-noise")   || "none");
  fd.append("noise_type",   val("g-ntype")   || "white");
  fd.append("demo_oracle",  val("g-oracle") === "true" ? "true" : "false");
  fd.append("sep_model",    val("g-sep-model") || "sepformer_whamr16k");
  const n = parseInt(val("g-speakers")) || 2;
  const idxs = ["mohit","rahul","priya"].slice(0,n).map(spk => {
    const el = document.getElementById(`cmd_${spk}`);
    return el ? el.value : "0";
  });
  fd.append("cmd_indices", idxs.join(","));
  for (const [k, v2] of Object.entries(extra)) fd.append(k, v2);
  return fd;
}

// ── Generate Demo ──
async function generateDemo() {
  showStatus("Generating TTS voices and mixing audio…");
  try {
    const res  = await fetch("/generate_demo", { method: "POST", body: buildFD() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");

    // Show mixed audio section
    const sec = document.getElementById("mixedSection");
    sec.classList.remove("hidden");
    document.getElementById("mixedPlayer").src = data.audio_url + "?t=" + Date.now();

    // Ground truth chips
    const chips = document.getElementById("gtChips");
    chips.innerHTML = "";
    (data.ground_truth?.sources || []).forEach(s => {
      const c = document.createElement("div");
      c.className = "gt-chip";
      c.innerHTML = `<span class="gt-chip-name">${s.speaker}</span><span class="gt-chip-text">${s.text}</span>`;
      chips.appendChild(c);
    });

    hideStatus();
  } catch (e) {
    hideStatus();
    alert("Generate error: " + e.message);
  }
}

// ── Process Demo ──
async function processDemo() {
  showStatus("Running full pipeline on demo audio…");
  try {
    const res  = await fetch("/process_demo", { method: "POST", body: buildFD() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");
    renderResults(data);
    hideStatus();
  } catch (e) {
    hideStatus();
    alert("Process error: " + e.message);
  }
}

// ── Upload ──
function onFileChosen(input) {
  uploadedFile = input.files[0];
  if (!uploadedFile) return;
  const nameEl = document.getElementById("uploadedName");
  nameEl.textContent = uploadedFile.name;
  nameEl.classList.remove("hidden");
  document.getElementById("btnProcess").disabled = false;
}

async function processUpload() {
  if (!uploadedFile) return;
  showStatus("Processing uploaded audio…");
  try {
    const fd = new FormData();
    fd.append("file", uploadedFile);
    fd.append("num_speakers", val("g-speakers") || "2");
    fd.append("demo_oracle",  val("g-oracle") === "true" ? "true" : "false");
    fd.append("sep_model",    val("g-sep-model") || "sepformer_whamr16k");
    const res  = await fetch("/process_audio", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");
    renderResults(data);
    hideStatus();
  } catch (e) {
    hideStatus();
    alert("Upload process error: " + e.message);
  }
}

// ── Render results ──
function renderResults(data) {
  // Speaker cards
  const sec  = document.getElementById("resultsSection");
  const grid = document.getElementById("resultsGrid");
  sec.classList.remove("hidden");
  grid.innerHTML = "";

  (data.results || []).forEach((r, i) => {
    const conf    = parseFloat(r.speaker_confidence);
    const confPct = isNaN(conf) ? "N/A" : (conf * 100).toFixed(1) + "%";
    const confCls = conf >= 0.65 ? "conf-hi" : conf >= 0.4 ? "conf-md" : "conf-lo";
    const avCls   = AV_CLASS[i % AV_CLASS.length];
    const initial = r.identified_speaker ? r.identified_speaker.charAt(0).toUpperCase() : "?";

    const warns = [];
    if (r.speaker_fallback) warns.push("Speaker ID used oracle fallback");
    if (r.asr_fallback && r.asr_warning) warns.push(r.asr_warning);

    const card = document.createElement("div");
    card.className = "spk-card";
    card.innerHTML = `
      <div class="spk-card-top">
        <div class="spk-avatar ${avCls}">${initial}</div>
        <div class="spk-meta">
          <div class="spk-name">${r.identified_speaker || "Unknown"}</div>
          <div class="spk-conf ${confCls}">Match: ${confPct}</div>
        </div>
      </div>

      ${r.audio_url ? `
      <div class="spk-card-audio">
        <audio controls src="${r.audio_url}?t=${Date.now()}"></audio>
      </div>` : ""}

      <div class="spk-card-body">
        <div class="info-row">
          <span class="info-key">Transcript</span>
          <span class="info-val transcript">"${r.transcript || "(empty)"}"</span>
        </div>
        <div class="info-row">
          <span class="info-key">Intent</span>
          <span class="info-val"><span class="badge badge-intent">${r.intent}</span></span>
        </div>
        <div class="info-row">
          <span class="info-key">Command</span>
          <span class="info-val">
            ${r.is_command
              ? '<span class="badge badge-cmd">Command detected</span>'
              : '<span class="badge badge-nocmd">Not a command</span>'}
          </span>
        </div>

        <div class="response-block">${r.response}</div>

        ${warns.map(w => `<div class="warn-tag">${w}</div>`).join("")}
      </div>`;
    grid.appendChild(card);
  });

  renderMetrics(data.metrics, data.separation_mode, data.separation_warning,
                data.separation_model);
}

// ── Render metrics ──
function renderMetrics(m, sepMode, sepWarn, sepModelLabel) {
  const sec   = document.getElementById("metricsSection");
  const strip = document.getElementById("metricsStrip");
  sec.classList.remove("hidden");

  const fmt = (v, unit = "", dec = 2) =>
    v === null || v === undefined ? null : Number(v).toFixed(dec) + unit;

  // Display friendly model label if available, else fall back to mode string
  const sepDisplay = sepModelLabel || sepMode || "—";

  const tiles = [
    { label: "Audio Duration",  val: fmt(m.audio_duration, "s"),              sub: "" },
    { label: "Process Time",    val: fmt(m.processing_time, "s"),             sub: "" },
    { label: "xRT",             val: fmt(m.xRT, "", 3),                       sub: "lower = faster than real-time" },
    { label: "WER",             val: m.wer != null ? fmt(m.wer * 100, "%", 1) : null, sub: "word error rate" },
    { label: "SI-SNR",          val: fmt(m.si_snr, " dB"),                    sub: "signal quality" },
    { label: "Sep Model",       val: sepDisplay,                              sub: sepWarn || "" },
    { label: "Model Params",    val: typeof m.model_params === "string" && m.model_params.length < 20 ? m.model_params : null,
                                sub: typeof m.model_params === "string" && m.model_params.length >= 20 ? m.model_params : "" },
  ];

  strip.innerHTML = tiles.map(t => `
    <div class="metric-tile">
      <div class="m-val${t.val === null ? " na" : ""}">${t.val ?? "N/A"}</div>
      <div class="m-label">${t.label}</div>
      ${t.sub ? `<div class="m-sub">${t.sub}</div>` : ""}
    </div>`).join("");
}

// ── Enroll real speaker ──
async function enrollSpeaker() {
  const name  = document.getElementById("enrollName").value.trim();
  const files = document.getElementById("enrollFiles").files;
  const statusEl = document.getElementById("enrollStatus");

  if (!name) { flashEnroll("Please enter your name.", "warn"); return; }
  if (!files.length) { flashEnroll("Please select at least one audio file.", "warn"); return; }

  flashEnroll("Enrolling speaker — computing voice embeddings…", "info");

  const fd = new FormData();
  fd.append("name", name);
  for (const f of files) fd.append("files", f);

  try {
    const res  = await fetch("/enroll_speaker", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");
    flashEnroll(`Enrolled "${data.enrolled.name}" with ${data.enrolled.samples} sample(s). Embedding dim: ${data.enrolled.embedding_dim}.`, "ok");
    loadSpeakerList();
  } catch (e) {
    flashEnroll("Enrollment failed: " + e.message, "err");
  }
}

function flashEnroll(msg, type) {
  const el = document.getElementById("enrollStatus");
  el.classList.remove("hidden");
  const colors = {
    ok:   "background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.25);color:#6ee7b7;",
    warn: "background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.25);color:#fcd34d;",
    err:  "background:rgba(248,113,113,0.10);border:1px solid rgba(248,113,113,0.2);color:#fca5a5;",
    info: "background:rgba(139,92,246,0.10);border:1px solid rgba(139,92,246,0.25);color:#c4b5fd;",
  };
  el.style.cssText = colors[type] || colors.info;
  el.textContent = msg;
}

async function loadSpeakerList() {
  try {
    const res  = await fetch("/speakers");
    const data = await res.json();
    const container = document.getElementById("speakerList");
    container.innerHTML = "";
    (data.speakers || []).forEach(spk => {
      const chip = document.createElement("div");
      chip.style.cssText = `display:flex;align-items:center;gap:0.5rem;padding:0.35rem 0.75rem;
        border-radius:999px;border:1px solid var(--border);background:var(--surface);font-size:0.78rem;`;
      const typeColor = spk.type === "real" ? "#06b6d4" : "#8b5cf6";
      chip.innerHTML = `
        <span style="width:8px;height:8px;border-radius:50%;background:${typeColor};flex-shrink:0"></span>
        <span style="color:var(--text);font-weight:600;">${spk.name}</span>
        <span style="color:var(--text-3);">${spk.type === "real" ? "Real" : "TTS"}</span>
        <button onclick="deleteSpeaker('${spk.name}', this.parentElement)"
          style="background:none;border:none;color:var(--text-3);cursor:pointer;font-size:0.85rem;padding:0 0.1rem;"
          title="Remove">x</button>`;
      container.appendChild(chip);
    });
    if (!data.speakers?.length) {
      container.innerHTML = '<span style="color:var(--text-3);font-size:0.8rem;">No speakers enrolled yet.</span>';
    }
  } catch(e) { console.error(e); }
}

async function deleteSpeaker(name, chipEl) {
  if (!confirm(`Remove speaker "${name}"?`)) return;
  try {
    const res = await fetch(`/speakers/${encodeURIComponent(name)}`, { method: "DELETE" });
    if (res.ok) chipEl.remove();
    else alert("Could not delete speaker.");
  } catch(e) { alert(e.message); }
}

// ── Test voice identity ──
async function testVoice() {
  const input = document.getElementById("testVoiceFile");
  if (!input.files.length) { alert("Please select an audio file to test."); return; }
  const resultEl = document.getElementById("voiceTestResult");
  resultEl.classList.remove("hidden");
  resultEl.innerHTML = '<span style="color:var(--text-2);font-size:0.82rem;">Identifying…</span>';

  const fd = new FormData();
  fd.append("file", input.files[0]);

  try {
    const res  = await fetch("/identify_voice", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");

    const conf   = (parseFloat(data.confidence) * 100).toFixed(1);
    const isUnk  = data.identified_speaker === "Unknown";
    const color  = isUnk ? "#f87171" : "#6ee7b7";

    // Build similarity bars
    const simRows = Object.entries(data.all_similarities || {})
      .sort((a, b) => b[1] - a[1])
      .map(([spk, sim]) => {
        const pct = Math.max(0, Math.min(100, sim * 100 / 0.8 * 100));
        return `<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.35rem;">
          <span style="min-width:60px;font-size:0.78rem;color:var(--text-2);">${spk}</span>
          <div style="flex:1;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">
            <div style="height:100%;width:${Math.abs(sim)*150}%;background:linear-gradient(90deg,var(--violet),var(--cyan));border-radius:3px;transition:width 0.4s;"></div>
          </div>
          <span style="font-size:0.72rem;font-family:monospace;color:var(--text-3);min-width:42px;">${sim.toFixed(3)}</span>
        </div>`;
      }).join("");

    resultEl.innerHTML = `
      <div style="padding:0.85rem 1rem;background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:12px;">
        <div style="font-size:1.1rem;font-weight:700;color:${color};margin-bottom:0.3rem;">
          ${isUnk ? "Unknown Speaker" : data.identified_speaker}
        </div>
        <div style="font-size:0.78rem;color:var(--text-2);margin-bottom:0.85rem;">
          Confidence: ${conf}% ${data.fallback ? "(fallback mode)" : ""}
        </div>
        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;color:var(--text-3);margin-bottom:0.5rem;">
          Similarity vs all enrolled speakers
        </div>
        ${simRows}
      </div>`;
  } catch(e) {
    resultEl.innerHTML = `<span style="color:#f87171;font-size:0.82rem;">Error: ${e.message}</span>`;
  }
}

// ── Custom file label updater ──
function updateFileLabel(input, labelId) {
  const el = document.getElementById(labelId);
  if (!el) return;
  if (!input.files.length) {
    el.textContent = 'No files chosen';
    el.classList.remove('has-file');
  } else if (input.files.length === 1) {
    el.textContent = input.files[0].name;
    el.classList.add('has-file');
  } else {
    el.textContent = `${input.files.length} files selected`;
    el.classList.add('has-file');
  }
}

// ── Init ──
// ── Load separation model download status on page init ──
async function loadSepModelStatus() {
  try {
    const data = await (await fetch("/api/separation_models")).json();
    data.forEach(m => {
      const btn = document.querySelector(`#g-sep-model .pill[data-v="${m.id}"]`);
      if (!btn) return;
      if (!m.downloaded) {
        btn.title += " (not downloaded — will download on first use)";
        btn.style.opacity = "0.6";
      } else {
        btn.title += " (Downloaded)";
      }
    });
  } catch (_) { /* ignore — server may still be starting */ }
}

document.addEventListener("DOMContentLoaded", () => {
  buildCmdGrid();
  loadSpeakerList();
  loadSepModelStatus();
  toggleSepModel();   // set initial opacity based on default oracle=false
});

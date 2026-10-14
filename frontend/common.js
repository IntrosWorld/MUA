const MUA = {
  escape(value) {
    return String(value ?? '').replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
    );
  },
  icon(name) {
    return (
      '<span class="icon" aria-hidden="true" style="--icon:url(\'/assets/icons/' +
      name +
      '.svg\')"></span>'
    );
  },
  async api(path, options = {}) {
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok)
      throw new Error(
        typeof data.detail === 'string'
          ? data.detail
          : data.error || 'The request could not be completed.',
      );
    return data;
  },
  json(method, body) {
    return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  },
  notify(message, error = false) {
    let toast = document.getElementById('toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'toast';
      toast.setAttribute('role', 'status');
      document.body.append(toast);
    }
    toast.className = 'toast' + (error ? ' error' : '');
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => (toast.hidden = true), error ? 10000 : 5000);
  },
  init() {
    if (this.initialized) return;
    this.initialized = true;
    const dark = matchMedia('(prefers-color-scheme:dark)');
    this.theme(localStorage.getItem('mua-theme') || (dark.matches ? 'dark' : 'light'));
    document
      .querySelectorAll('[data-icon]')
      .forEach((el) =>
        el.style.setProperty('--icon', "url('/assets/icons/" + el.dataset.icon + ".svg')"),
      );
    document.querySelectorAll('[data-theme-toggle]').forEach((btn) =>
      btn.addEventListener('click', () => {
        const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
        this.theme(next);
        localStorage.setItem('mua-theme', next);
      }),
    );
    dark.addEventListener('change', (e) => {
      if (!localStorage.getItem('mua-theme')) this.theme(e.matches ? 'dark' : 'light');
    });
    this.api('/llm/status')
      .then((d) =>
        document
          .querySelectorAll('[data-model-label]')
          .forEach(
            (el) =>
              (el.textContent =
                d.mode === 'ollama'
                  ? d.local_config.ollama_model
                  : d.mode === 'local-api'
                    ? d.local_config.local_api_model || 'Select a model'
                    : d.mode === 'offline'
                      ? 'Local weights'
                      : 'Gemini'),
          ),
      )
      .catch(() =>
        document
          .querySelectorAll('[data-model-label]')
          .forEach((el) => (el.textContent = 'Backend unavailable')),
      );
  },
  theme(value) {
    document.documentElement.dataset.theme = value;
    document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
      btn.innerHTML = this.icon(value === 'dark' ? 'sun' : 'moon');
      btn.setAttribute(
        'aria-label',
        'Switch to ' + (value === 'dark' ? 'light' : 'dark') + ' theme',
      );
    });
  },
};
document.addEventListener('DOMContentLoaded', () => MUA.init());
document.addEventListener('keydown', (event) => {
  const tab = event.target.closest('[role=tab]');
  if (!tab || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  const tabs = [...tab.closest('[role=tablist]').querySelectorAll('[role=tab]')];
  const at = tabs.indexOf(tab);
  const next =
    event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? tabs.length - 1
        : (at + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  event.preventDefault();
  tabs[next].focus();
  tabs[next].click();
});

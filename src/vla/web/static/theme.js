/* Apply the saved theme before the first paint; storage may be unavailable. */
(() => {
  const key = 'vla.theme';
  const root = document.documentElement;
  function apply(value) {
    root.dataset.theme = value === 'light' ? 'light' : 'dark';
    const button = document.getElementById('themeToggle');
    if (button) {
      const light = root.dataset.theme === 'light';
      button.textContent = light ? '切换深色' : '切换浅色';
      button.title = light ? '切换到深色模式' : '切换到浅色模式';
      button.setAttribute('aria-pressed', String(light));
    }
    window.dispatchEvent(new Event('vla-theme-change'));
  }
  let saved;
  try { saved = localStorage.getItem(key); } catch {}
  apply(saved);
  document.addEventListener('DOMContentLoaded', () => {
    apply(root.dataset.theme);
    document.getElementById('themeToggle').addEventListener('click', () => {
      const value = root.dataset.theme === 'light' ? 'dark' : 'light';
      try { localStorage.setItem(key, value); } catch {}
      apply(value);
    });
  });
  window.addEventListener('storage', event => {
    if (event.key === key || event.key === null) apply(event.newValue);
  });
})();

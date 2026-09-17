/* XORTRON browser workspace. Original integration; no upstream code copied. */
(() => {
  'use strict';
  const MAX_FILE = 10 * 1024 * 1024, MAX_TOTAL = 25 * 1024 * 1024, MAX_FILES = 32;
  const encoder = new TextEncoder(), decoder = new TextDecoder();
  let database;
  const memory = new Map();
  function filename(value) {
    const name = String(value || '').replace(/^\/workspace\//, '');
    if (!name || name.length > 160 || /[\\/\x00-\x1f]/.test(name) || name === '.' || name === '..') {
      throw new Error('Use a simple filename without folders.');
    }
    return name;
  }
  async function db() {
    if (!database) database = new Promise((resolve, reject) => {
      const r = indexedDB.open('xortron-workspace-v1', 1);
      r.onupgradeneeded = () => r.result.createObjectStore('chats');
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
    return database;
  }
  async function state(chatId) {
    if (memory.has(chatId)) return memory.get(chatId);
    let value;
    try {
      const d = await db();
      value = await new Promise((resolve, reject) => {
        const r = d.transaction('chats').objectStore('chats').get(chatId);
        r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error);
      });
    } catch (_) { /* private mode: retain a working in-memory workspace */ }
    value ||= { files: [], downloads: [] };
    memory.set(chatId, value);
    return value;
  }
  async function save(chatId, value) {
    memory.set(chatId, value);
    try {
      const d = await db();
      await new Promise((resolve, reject) => {
        const t = d.transaction('chats', 'readwrite');
        t.objectStore('chats').put(value, chatId);
        t.oncomplete = resolve; t.onerror = () => reject(t.error); t.onabort = () => reject(t.error);
      });
      return true;
    } catch (_) { return false; }
  }
  function validate(files) {
    if (files.length > MAX_FILES) throw new Error('Workspace limit is 32 files.');
    let total = 0;
    const seen = new Set();
    for (const f of files) {
      f.name = filename(f.name);
      if (seen.has(f.name)) throw new Error('Duplicate filename.');
      seen.add(f.name);
      if (!(f.bytes instanceof Uint8Array)) f.bytes = new Uint8Array(f.bytes);
      if (f.bytes.byteLength > MAX_FILE) throw new Error(`${f.name}: maximum file size is 10 MB.`);
      total += f.bytes.byteLength;
    }
    if (total > MAX_TOTAL) throw new Error('Workspace limit is 25 MB.');
  }
  function equal(a, b) {
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }
  function upsert(files, item) {
    const next = files.filter(f => f.name !== item.name).concat(item);
    validate(next); return next;
  }

  // Each execution has a fresh opaque-origin iframe and a dedicated worker.
  // The iframe cannot see cookies, DOM, IndexedDB, credentials or other chats.
  // CSP only permits the pinned runtime's CDN, not app/API/network access.
  function sandboxHost() {
    let worker;
    addEventListener('message', event => {
      if (!event.ports[0] || event.source !== parent) return;
      const port = event.ports[0];
      port.onmessage = event => {
        if (event.data.stop) { worker?.terminate(); return; }
        const url = URL.createObjectURL(new Blob([event.data.source], {type: 'text/javascript'}));
        worker = new Worker(url, {type: 'module'});
        URL.revokeObjectURL(url);
        worker.onmessage = e => port.postMessage(e.data);
        worker.onerror = e => port.postMessage({error: e.message || 'Python worker failed.'});
        worker.postMessage(event.data.task);
      };
      port.start(); port.postMessage({ready: true});
    });
  }
  async function pythonWorker() {
    self.onmessage = async ({data}) => {
      let output = '';
      const append = s => { output = (output + s + '\n').slice(-14000); };
      try {
        const {loadPyodide} = await import('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs');
        const py = await loadPyodide({
          indexURL: 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/',
          stdout: append, stderr: append,
        });
        py.FS.mkdirTree('/workspace'); py.FS.chdir('/workspace');
        for (const f of data.files) py.FS.writeFile('/workspace/' + f.name, f.bytes);
        await py.runPythonAsync("import os\nos.environ['MPLBACKEND'] = 'Agg'");
        await py.loadPackagesFromImports(data.code);
        let failure = '';
        try {
          const answer = await py.runPythonAsync(data.code);
          if (answer !== undefined) { append(String(answer)); answer?.destroy?.(); }
        } catch (error) { failure = String(error); }
        const files = [];
        let total = 0;
        for (const name of py.FS.readdir('/workspace')) {
          if (name === '.' || name === '..') continue;
          const path = '/workspace/' + name;
          const stat = py.FS.lstat(path);
          if (!py.FS.isFile(stat.mode)) continue; // Never follow symlinks.
          if (stat.size > 10 * 1024 * 1024 || total + stat.size > 25 * 1024 * 1024 || files.length >= 32) {
            throw new Error('Generated files exceed workspace limits.');
          }
          total += stat.size;
          files.push({name, bytes: py.FS.readFile(path)});
        }
        self.postMessage({output, files, failure});
      } catch (error) { self.postMessage({error: String(error)}); }
    };
  }
  async function executePython(code, files, signal) {
    if (typeof code !== 'string' || code.length > 60000) throw new Error('Invalid Python program.');
    if (signal?.aborted) throw new DOMException('Stopped', 'AbortError');
    return new Promise((resolve, reject) => {
      const frame = document.createElement('iframe');
      frame.hidden = true; frame.setAttribute('sandbox', 'allow-scripts');
      frame.referrerPolicy = 'no-referrer';
      const policy = "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' blob: https://cdn.jsdelivr.net; worker-src blob:; connect-src https://cdn.jsdelivr.net; child-src blob:;";
      frame.srcdoc = `<meta http-equiv="Content-Security-Policy" content="${policy}"><script>(${sandboxHost.toString()})()<\/script>`;
      const channel = new MessageChannel();
      let settled = false;
      function finish(error, value) {
        if (settled) return; settled = true;
        clearTimeout(timer); signal?.removeEventListener('abort', abort);
        channel.port1.postMessage({stop: true}); channel.port1.close(); frame.remove();
        if (error) reject(error); else resolve(value);
      }
      const abort = () => finish(new DOMException('Stopped', 'AbortError'));
      const timer = setTimeout(() => finish(new Error('Python timed out after 90 seconds; try a smaller task.')), 90000);
      signal?.addEventListener('abort', abort, {once: true});
      channel.port1.onmessage = ({data}) => {
        if (data.ready) channel.port1.postMessage({source: `(${pythonWorker.toString()})()`, task: {code, files}});
        else if (data.error) finish(new Error(data.error));
        else finish(null, data);
      };
      frame.onload = () => frame.contentWindow.postMessage({init: true}, '*', [channel.port2]);
      document.body.appendChild(frame);
    });
  }
  async function call(call, chatId, message, signal) {
    let result;
    try {
      if (signal?.aborted) throw new DOMException('Stopped', 'AbortError');
      const s = await state(chatId), args = call.arguments || {};
      let next = s.files, output = '';
      switch (call.name) {
        case 'browser_python': {
          const response = await executePython(args.code, s.files, signal);
          validate(response.files);
          next = response.files;
          output = response.output + (response.failure ? '\nPython error (correct and retry): ' + response.failure : '\nExecution succeeded.');
          break;
        }
        case 'browser_write_file': {
          if (typeof args.text !== 'string') throw new Error('File text must be a string.');
          next = upsert(s.files, {name: filename(args.name), bytes: encoder.encode(args.text)});
          output = 'File created.'; break;
        }
        case 'browser_read_file': {
          const f = s.files.find(f => f.name === filename(args.name));
          if (!f) throw new Error('File not found. Use browser_list_files.');
          output = decoder.decode(f.bytes.slice(0, 14000)) + (f.bytes.length > 14000 ? '\n[Truncated; use Python for the full file.]' : '');
          break;
        }
        case 'browser_list_files': output = JSON.stringify(s.files.map(f => ({name:f.name, bytes:f.bytes.length}))); break;
        default: throw new Error('Unknown browser tool.');
      }
      if (signal?.aborted) throw new DOMException('Stopped', 'AbortError');
      const changed = next.filter(f => !s.files.some(old => old.name === f.name && equal(old.bytes, f.bytes)));
      s.files = next;
      for (const f of changed) {
        const id = crypto.randomUUID();
        s.downloads.push({id, name: f.name, bytes: f.bytes});
        message.downloads ||= [];
        message.downloads.push({id, name:f.name});
      }
      // Keep immutable download snapshots within a separate 25 MB budget.
      let retained = s.downloads.reduce((sum, f) => sum + f.bytes.length, 0);
      while (retained > MAX_TOTAL || s.downloads.length > 64) retained -= s.downloads.shift().bytes.length;
      const durable = await save(chatId, s);
      result = output + (changed.length ? '\nDownload buttons created: ' + changed.map(f => f.name).join(', ') : '');
      if (!durable) result += '\nBrowser storage unavailable: files last only for this page session.';
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      result = 'Tool error: ' + error.message;
    }
    const response = await fetch('/browser-tool-result/' + encodeURIComponent(call.token), {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({result: result.slice(0, 18000)}), signal,
    });
    if (!response.ok) throw new Error('Browser tool result could not be delivered (' + response.status + ').');
  }
  async function upload(chatId, file) {
    if (file.size > MAX_FILE) throw new Error('Maximum attachment size is 10 MB.');
    const s = await state(chatId);
    s.files = upsert(s.files, {name: filename(file.name), bytes: new Uint8Array(await file.arrayBuffer())});
    await save(chatId, s);
  }
  async function manifest(chatId) {
    const s = await state(chatId);
    return s.files.map(f => ({name:f.name, bytes:f.bytes.length}));
  }
  function buttons(container, chatId, downloads) {
    if (!downloads?.length) return;
    const group = document.createElement('div'); group.className = 'browser-downloads';
    for (const item of downloads) {
      const button = document.createElement('button'); button.className = 'file-chip';
      button.textContent = '↓ ' + item.name;
      button.onclick = async () => {
        const s = await state(chatId), f = s.downloads.find(f => f.id === item.id);
        if (!f) { button.textContent = item.name + ' — no longer stored'; return; }
        // Download only: generated HTML/SVG never runs inside the chat origin.
        const url = URL.createObjectURL(new Blob([f.bytes], {type:'application/octet-stream'}));
        const link = document.createElement('a'); link.href = url; link.download = f.name;
        document.body.appendChild(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      };
      group.appendChild(button);
    }
    container.appendChild(group);
  }
  function activities(container, items) {
    const opened = new Set([...container.querySelectorAll('details[open]')].map(d => d.dataset.id));
    container.replaceChildren();
    for (const item of items || []) {
      const details = document.createElement('details');
      details.className = 'tool-step'; details.dataset.id = item.id;
      details.open = opened.has(item.id);
      const summary = document.createElement('summary');
      const icon = document.createElement('span'); icon.className = 'tool-step-icon';
      icon.textContent = item.status === 'running' ? '◌' : item.status === 'error' ? '!' : item.status === 'stopped' ? '−' : '✓';
      const title = document.createElement('span');
      title.textContent = item.label + (item.query ? ': ' + item.query : '');
      const timing = document.createElement('span'); timing.className = 'tool-step-time';
      timing.textContent = item.status === 'running' ? 'Working…' : item.status === 'stopped' ? 'Stopped' : `${item.seconds || 0}s`;
      summary.append(icon, title, timing); details.appendChild(summary);
      const sources = document.createElement('div'); sources.className = 'tool-source-chips';
      for (const source of item.sources || []) {
        let url; try { url = new URL(source.url); } catch (_) { continue; }
        if (!['https:', 'http:'].includes(url.protocol)) continue;
        const a = document.createElement('a');
        a.href = url.href; a.target = '_blank'; a.rel = 'noopener noreferrer';
        a.textContent = '◎ ' + (source.title || url.hostname); a.title = url.href;
        sources.appendChild(a);
      }
      if (item.code) {
        const pre = document.createElement('pre'); pre.textContent = item.code;
        pre.className = 'tool-step-code'; details.appendChild(pre);
      }
      if (item.result) {
        const pre = document.createElement('pre'); pre.textContent = item.result;
        pre.className = 'tool-step-result'; details.appendChild(pre);
      }
      container.appendChild(details);
      if (sources.childNodes.length) container.appendChild(sources);
    }
  }
  async function clear(chatId) {
    if (chatId) memory.delete(chatId); else memory.clear();
    try {
      const d = await db();
      await new Promise((resolve, reject) => {
        const t = d.transaction('chats', 'readwrite'), store = t.objectStore('chats');
        if (chatId) store.delete(chatId); else store.clear();
        t.oncomplete = resolve; t.onerror = () => reject(t.error);
      });
    } catch (_) {}
  }
  window.XortronBrowser = {call, upload, manifest, buttons, activities, clear};
})();

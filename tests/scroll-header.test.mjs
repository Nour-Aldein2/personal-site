import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
const code = await readFile(new URL('../src/scripts/scroll-header.js', import.meta.url), 'utf8');

// Run the actual shipped controller, not a reimplementation of its state logic.
function harness({ scroll = 0, loading = false, resizeObserver = true, present = true } = {}) {
  const makeEvents = () => {
    const events = new Map();
    return {
      addEventListener(type, fn, options) {
        const listeners = events.get(type) || new Map();
        listeners.set(fn, options); events.set(type, listeners);
      },
      removeEventListener(type, fn) { events.get(type)?.delete(fn); },
      emit(type) {
        for (const [fn, options] of [...(events.get(type) || [])]) {
          fn(); if (options?.once) events.get(type)?.delete(fn);
        }
      },
      count(type) { return events.get(type)?.size || 0; },
    };
  };
  let height = 56;
  let header = present ? { dataset: {}, getBoundingClientRect: () => ({ height }) } : null;
  const props = new Map(), frames = new Map(), observers = [];
  let counter = 0;
  const window = {
    ...makeEvents(), scrollY: scroll,
    requestAnimationFrame(fn) { frames.set(++counter, fn); return counter; },
    cancelAnimationFrame(id) { frames.delete(id); },
  };
  const document = {
    ...makeEvents(), readyState: loading ? 'loading' : 'complete',
    documentElement: { style: { setProperty: (key, value) => props.set(key, value) } },
    querySelector: () => header,
  };
  class ResizeObserver {
    constructor(fn) { this.fn = fn; observers.push(this); }
    observe(node) { this.node = node; this.active = true; }
    disconnect() { this.active = false; }
  }
  const context = vm.createContext({ window, document, ...(resizeObserver ? { ResizeObserver } : {}) });
  if (resizeObserver) window.ResizeObserver = ResizeObserver;
  const run = () => vm.runInContext(code, context);
  const flush = () => { const tasks = [...frames.values()]; frames.clear(); tasks.forEach(fn => fn()); };
  run();
  return {
    window, document, props, frames, observers, run, flush,
    get header() { return header; },
    replaceHeader() { header = { dataset: {}, getBoundingClientRect: () => ({ height }) }; return header; },
    resize(value) { height = value; window.emit('resize'); flush(); },
    scrollTo(value) { window.scrollY = value; window.emit('scroll'); flush(); },
  };
}

test('header starts transparent at the top and publishes its real height', () => {
  const h = harness();
  assert.equal(h.header.dataset.scrolled, 'false');
  assert.equal(h.header.dataset.scrollHeader, 'ready');
  assert.equal(h.props.get('--sticky-header-height'), '56px');
});
test('the solid surface appears only past eight pixels and disappears at the top', () => {
  const h = harness();
  for (const value of [-12, 0, 1, 8]) { h.scrollTo(value); assert.equal(h.header.dataset.scrolled, 'false'); }
  h.scrollTo(9); assert.equal(h.header.dataset.scrolled, 'true');
  h.scrollTo(250); assert.equal(h.header.dataset.scrolled, 'true');
  h.scrollTo(0); assert.equal(h.header.dataset.scrolled, 'false');
});
test('loading mid-page starts opaque, including on back-forward restoration', () => {
  const h = harness({scroll: 500});
  assert.equal(h.header.dataset.scrolled, 'true');
  h.window.scrollY = 0; h.window.emit('pageshow');
  assert.equal(h.header.dataset.scrolled, 'false');
});
test('scroll events are batched into a single animation frame', () => {
  const h = harness(); h.flush();
  for (let i = 0; i < 50; i++) h.window.emit('scroll');
  assert.equal(h.frames.size, 1);
  h.flush(); assert.equal(h.frames.size, 0);
});
test('responsive header measurement updates anchor and sidebar offsets', () => {
  const h = harness(); h.resize(133.2);
  assert.equal(h.props.get('--sticky-header-height'), '134px');
  h.resize(56); assert.equal(h.props.get('--sticky-header-height'), '56px');
});
test('mounting twice replaces listeners and observers rather than leaking them', () => {
  const h = harness(); h.run();
  assert.equal(h.window.count('scroll'), 1); assert.equal(h.window.count('resize'), 1);
  assert.equal(h.window.count('pageshow'), 1); assert.equal(h.document.count('astro:page-load'), 1);
  assert.equal(h.observers.filter(o => o.active).length, 1);
});
test('page swaps attach to the new header and release the old observer', () => {
  const h = harness(); const old = h.header;
  h.replaceHeader(); h.window.scrollY = 100; h.document.emit('astro:page-load');
  assert.equal(h.header.dataset.scrolled, 'true'); assert.equal(old.dataset.scrolled, 'false');
  assert.equal(h.observers.filter(o => o.active).length, 1);
});
test('DOMContentLoaded and no-ResizeObserver fallback are supported', () => {
  const h = harness({ loading: true, resizeObserver: false });
  assert.equal(h.header.dataset.scrollHeader, undefined);
  h.document.emit('DOMContentLoaded'); assert.equal(h.header.dataset.scrollHeader, 'ready');
  h.resize(120); assert.equal(h.props.get('--sticky-header-height'), '120px');
});
test('pages without navigation do not throw and still allow clean teardown', () => {
  const h = harness({present: false});
  h.scrollTo(25); h.window.emit('pageshow'); h.resize(100);
  h.window.__personalSiteScrollHeaderCleanup();
  assert.equal(h.window.count('scroll'), 0); assert.equal(h.frames.size, 0);
});

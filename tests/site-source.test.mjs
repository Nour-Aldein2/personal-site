import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir, access } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const text = name => readFile(join(root, name), 'utf8');
async function walk(dir) {
  const out = [];
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name); out.push(...(e.isDirectory() ? await walk(p) : [p]));
  }
  return out;
}
async function exists(path) { try { await access(path); return true; } catch { return false; } }

test('every content route uses the shared page or article layout', async () => {
  const files = (await walk(join(root, 'src/pages'))).filter(p => p.endsWith('.astro'));
  assert.ok(files.length >= 6);
  for (const p of files) assert.match(await readFile(p, 'utf8'), /<(?:BaseLayout|ArticleLayout)\b/, p);
  assert.match(await text('src/layouts/ArticleLayout.astro'), /<BaseLayout\b/);
});
test('the base loads each shared stylesheet exactly once and has one header', async () => {
  const s = await text('src/layouts/BaseLayout.astro');
  assert.equal((s.match(/<SiteHeader\s*\/>/g) || []).length, 1);
  for (const file of ['theme.css','global.css','header.css','entries.css','canvas.css','scroll-header.css']) assert.equal(s.split(`styles/${file}`).length - 1, 1);
  assert.match(s, /theme-preference/); assert.match(s, /themechange/); assert.match(s, /entry-thumbnails/);
});
test('one theme file owns the approved white-and-brown light palette', async () => {
  const light = (await text('src/styles/theme.css')).split(':root[data-theme="dark"]')[0];
  for (const token of ['--accent: #7C2710;', '--accent-soft: #95543f;', '--bg: #ffffff;', '--ink: #302d2c;', '--header-text: var(--ink);', '--header-wash-opacity: 0;']) assert.ok(light.includes(token), token);
  assert.doesNotMatch(light, /#(?:29484d|57726b|88d8b0|9d97b8|b49fbb)\b/i);
});
test('dark mode retains the agreed warm palette without page-specific colour overrides', async () => {
  const dark = (await text('src/styles/theme.css')).split(':root[data-theme="dark"]')[1];
  for (const token of ['--accent: #CC9075;', '--bg: #181210;', '--surface: #211916;', '--ink: #f3eeeb;', '--header-wash-left: rgba(191, 79, 24, 0.16);']) assert.ok(dark.includes(token), token);
  const components = await Promise.all(['ArticleCard','ProjectCard'].map(name => text(`src/components/${name}.astro`)));
  components.forEach(s => assert.doesNotMatch(s, /<style\b|#[a-f0-9]{6}\b/i));
});
test('matte layers fade horizontally at both outer edges without desaturating text', async () => {
  const s = await text('src/styles/header.css');
  assert.match(s, /linear-gradient\(90deg,\s*transparent 0%,[\s\S]*?transparent 100%\)/);
  assert.match(s, /--header-fade-left: 27%/);
  assert.match(s, /--header-fade-right: 67%/);
  assert.doesNotMatch(s, /linear-gradient\((?:180deg|to bottom)/);
  for (const selector of ['.site-header::before {', '.site-header::after {']) assert.match(s.split(selector)[1].split('}')[0], /pointer-events: none/);
  assert.match(s, /--header-grain-strength: 0\.65/);
});
test('embedded texture is static and removed for print and forced colours', async () => {
  const s = await text('src/styles/header.css');
  const encoded = s.match(/background-image: url\("data:image\/svg\+xml,([^"\n]+)"\)/)?.[1];
  assert.ok(encoded);
  const svg = decodeURIComponent(encoded);
  assert.match(svg, /<feTurbulence[^>]*type="fractalNoise"/);
  assert.doesNotMatch(svg, /<(?:animate|script|image)\b/);
  for (const rule of ['@media print','@media (forced-colors: active)']) assert.match(s.split(rule)[1], /\.site-header::before, \.site-header::after \{ display: none; \}/);
});
test('compact name, role, navigation, PDF and theme controls are retained', async () => {
  const s = await text('src/components/SiteHeader.astro');
  assert.match(s, />Noor Aldeen<\/span>/);
  for (const label of ['Home','CV','Publications','Articles','Contact']) assert.match(s, new RegExp(`>${label}<\\/a>`));
  assert.match(s, /site-brand__role/); assert.match(s, /isArticleSection/); assert.match(s, /<ThemeToggle/);
});
test('article and project rows use the same borderless styles without type labels', async () => {
  for (const component of ['ArticleCard','ProjectCard']) {
    const s = await text(`src/components/${component}.astro`);
    assert.match(s, /entry-row/); assert.match(s, /data-entry-thumbnail/);
    assert.doesNotMatch(s, /\{(?:project\.status|articleTemplateLabels\[|article\.data\.templateType)/);
  }
  const css = await text('src/styles/entries.css');
  assert.match(css, /border: 0/); assert.match(css, /background: transparent/);
  assert.match(css, /color: var\(--metadata\)/); assert.match(css, /font-size: 0\.8rem/);
  assert.match(await text('src/components/ArticleCard.astro'), /<time\b/);
});
test('the introduction remains unboxed and its name exactly four points smaller', async () => {
  const s = await text('src/pages/index.astro');
  assert.match(s, /calc\(clamp\(2\.2rem, 5vw, 4rem\) - 4pt\)/);
  assert.match(s, /background: transparent/); assert.match(s, /box-shadow: none/);
});
test('all relative source and stylesheet imports exist in this archive', async () => {
  for (const p of (await walk(join(root, 'src'))).filter(p => /\.(astro|ts)$/.test(p))) {
    const s = await readFile(p, 'utf8');
    for (const m of s.matchAll(/(?:from\s+|import\s*)["'](\.[^"']+)["']/g)) {
      const target = resolve(dirname(p), m[1]);
      assert.ok((await Promise.all(['','.ts','.js','.astro','.css','/index.ts'].map(ext => exists(target + ext)))).some(Boolean), `${p}: ${m[1]}`);
    }
  }
});
test('the original article, publication records and legacy routes are retained', async () => {
  assert.match(await text('src/content/articles/bert-lime-disaster-tweets.md'), /publishDate:.*2024-04-12/);
  assert.equal(((await text('src/pages/publications/index.astro')).match(/doi:/g) || []).length, 3);
  for (const name of ['public/cv.html','public/post_explain_BERT_LIME.html','src/pages/robots.txt.ts','src/pages/sitemap.xml.ts']) assert.ok(await exists(join(root,name)));
});
test('dependencies and original asset source are pinned, not fetched from a moving branch', async () => {
  const pkg = JSON.parse(await text('package.json')); assert.equal(pkg.dependencies.astro, '6.2.1');
  const manifest = JSON.parse(await text('scripts/asset-manifest.json'));
  assert.match(manifest.revision, /^[a-f0-9]{40}$/); assert.ok(manifest.files.includes('public/Noor_CV.pdf'));
  assert.doesNotMatch(manifest.files.join('\n'), /\.(?:ttf|otf|woff2?)$/m);
});
test('GitHub validates PRs and only deploys successful non-PR builds', async () => {
  const s = await text('.github/workflows/deploy.yml');
  for (const step of ['npm run setup','npm ci','npm test','npm run build']) assert.ok(s.includes(step));
  assert.match(s, /needs: build/); assert.match(s, /if: github\.event_name != 'pull_request'/);
  assert.match(s, /pages: write/); assert.match(s, /id-token: write/);
});
test('404 is not indexed and canonical base-path helpers remain available', async () => {
  assert.match(await text('src/pages/404.astro'), /noindex/);
  assert.match(await text('src/components/SEOHead.astro'), /noindex \? "noindex, follow"/);
  assert.match(await text('src/lib/site.ts'), /export function absoluteSiteUrl/);
});

test('the seamless header controller is loaded once through BaseLayout', async () => {
  const s = await text('src/layouts/BaseLayout.astro');
  assert.equal(s.split('import "../scripts/scroll-header.js"').length - 1, 1);
  const css = await text('src/styles/scroll-header.css');
  assert.match(css, /data-scroll-header="ready"/);
  assert.match(css, /data-scrolled="true"/);
  assert.match(css, /background: transparent/);
  assert.match(css, /box-shadow: none/);
  assert.match(css, /prefers-reduced-motion/);
});
test('the continuous canvas contains light and dark static grids with edge fades', async () => {
  const css = await text('src/styles/canvas.css');
  assert.match(css, /body::before/); assert.match(css, /body::after/);
  assert.match(css, /--canvas-grid-line: rgb\(var\(--accent-rgb\) \/ 0\.055\)/);
  assert.match(css, /--canvas-grid-line: rgb\(var\(--accent-rgb\) \/ 0\.075\)/);
  assert.match(css, /mask-composite: intersect/); assert.match(css, /pointer-events: none/);
  assert.match(css, /forced-colors/); assert.match(css, /@media print/);
  assert.doesNotMatch(css, /animation:|@keyframes/);
});
test('the removed writing introduction does not reappear in any page', async () => {
  for (const p of (await walk(join(root, 'src'))).filter(p => p.endsWith('.astro'))) {
    assert.doesNotMatch(await readFile(p, 'utf8'), /Selected writing and technical case studies, including the current published/);
  }
});
test('no superseded blue, teal, mint or lavender theme values remain in styles', async () => {
  for (const p of (await walk(join(root, 'src/styles'))).filter(p => p.endsWith('.css'))) {
    assert.doesNotMatch(await readFile(p, 'utf8'), /#(?:29484d|57726b|88d8b0|9d97b8|b49fbb)\b/i, p);
  }
});

test('browser/app chrome and legacy fallbacks contain no superseded theme colours', async () => {
  for (const name of ['public/site.webmanifest','public/images/favicon_io_home/site.webmanifest']) {
    const manifest = JSON.parse(await text(name));
    assert.equal(manifest.theme_color, '#ffffff');
    assert.equal(manifest.background_color, '#ffffff');
  }
  for (const name of ['public/cv.html','public/post_explain_BERT_LIME.html']) {
    const s = await text(name);
    assert.match(s, /--accent: #7C2710/); assert.match(s, /--accent: #CC9075/);
    assert.doesNotMatch(s, /#(?:29484d|57726b|88d8b0|9d97b8|b49fbb)\b/i);
  }
});

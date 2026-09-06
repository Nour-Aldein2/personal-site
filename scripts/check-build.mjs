#!/usr/bin/env node
/** Check the real Astro output, not a screenshot fixture. No extra dependencies. */
import { access, readdir, readFile, writeFile, rm } from 'node:fs/promises';
import { resolve, dirname, relative, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
async function walk(dir) {
  const paths = [];
  for (const item of await readdir(dir, { withFileTypes: true })) {
    const path = resolve(dir, item.name);
    paths.push(...(item.isDirectory() ? await walk(path) : [path]));
  }
  return paths;
}
async function exists(path) { try { await access(path); return true; } catch { return false; } }
const unescape = value => value.replace(/&amp;/g, '&').replace(/&#39;/g, "'").replace(/&quot;/g, '"');
export async function checkBuild({ dist = resolve(root, 'dist'), site = process.env.SITE_URL || 'https://nourd.uk', base = process.env.BASE_PATH || '/' } = {}) {
  const origin = new URL(site).origin;
  base = '/' + base.replace(/^\/+|\/+$/g, '');
  if (base !== '/') base += '/';
  const files = await walk(dist);
  const htmlFiles = files.filter(file => file.endsWith('.html'));
  const failures = [];
  for (const required of ['index.html', 'articles/index.html', 'publications/index.html', 'cv/index.html', '404.html', 'robots.txt', 'sitemap.xml', 'Noor_CV.pdf']) {
    if (!await exists(resolve(dist, required))) failures.push(`Missing output: ${required}`);
  }
  let checkedLinks = 0;
  for (const file of htmlFiles) {
    const name = relative(dist, file).split(sep).join('/');
    const html = await readFile(file, 'utf8');
    const legacy = ['cv.html', 'post_explain_BERT_LIME.html'].includes(name);
    if (!legacy) {
      for (const pattern of [/<html\b[^>]*lang="en"/, /class="[^"]*site-header/, /data-site-design="seamless-brown-grid"/, /id="main-content"/, /theme-preference/, /<title>[^<]+<\/title>/]) {
        if (!pattern.test(html)) failures.push(`${name}: missing shared layout requirement ${pattern}`);
      }
    }
    if (/Selected writing and technical case studies, including the current published\s*post on model explainability with BERT and LIME\./.test(html)) {
      failures.push(`${name}: removed writing-section sentence reappeared`);
    }
    const pageUrl = new URL(base + name.replace(/index\.html$/, ''), origin);
    for (const match of html.matchAll(/\b(?:href|src)\s*=\s*["']([^"']+)["']/g)) {
      const raw = unescape(match[1]);
      if (/^(?:#|mailto:|tel:|data:|javascript:)/i.test(raw)) continue;
      let url;
      try { url = new URL(raw, pageUrl); } catch { failures.push(`${name}: malformed URL ${raw}`); continue; }
      if (url.origin !== origin) continue;
      if (!url.pathname.startsWith(base)) { failures.push(`${name}: URL escapes BASE_PATH: ${raw}`); continue; }
      let path;
      try { path = decodeURIComponent(url.pathname.slice(base.length)); } catch { failures.push(`${name}: invalid URL encoding ${raw}`); continue; }
      const target = resolve(dist, path || '.');
      if (target !== dist && !target.startsWith(dist + sep)) { failures.push(`${name}: unsafe path ${raw}`); continue; }
      const found = await exists(target) || await exists(resolve(target, 'index.html'));
      if (!found) failures.push(`${name}: missing local target ${raw}`);
      checkedLinks++;
    }
  }
  if (failures.length) throw new Error('Build validation failed:\n' + [...new Set(failures)].join('\n'));
  // A project-pages build must not accidentally carry the custom-domain CNAME.
  const host = new URL(site).hostname;
  if (host.endsWith('.github.io')) await rm(resolve(dist, 'CNAME'), { force: true });
  else if (base === '/') await writeFile(resolve(dist, 'CNAME'), host + '\n');
  const summary = { htmlPages: htmlFiles.length, checkedLocalLinks: checkedLinks, site, base, result: 'passed' };
  console.log(JSON.stringify(summary, null, 2));
  return summary;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  checkBuild().catch(error => { console.error(error.message); process.exitCode = 1; });
}

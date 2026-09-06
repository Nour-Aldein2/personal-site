#!/usr/bin/env node
/** Keep the two old .html URLs useful and themed without maintaining old pages. */
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const esc = value => value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
export async function finalizeBuild({ source = root, dist = resolve(root, 'dist') } = {}) {
  const theme = await readFile(resolve(source, 'src/styles/theme.css'), 'utf8');
  const canvas = await readFile(resolve(source, 'src/styles/canvas.css'), 'utf8');
  const layout = await readFile(resolve(source, 'src/layouts/BaseLayout.astro'), 'utf8');
  const themeScript = layout.match(/<script is:inline>([\s\S]*?)<\/script>/)?.[1];
  if (!themeScript) throw new Error('Shared theme initializer could not be found.');
  const routes = [
    ['cv.html', 'Noor_CV.pdf', 'Curriculum vitae', 'Continue to the CV PDF'],
    ['post_explain_BERT_LIME.html', 'articles/bert-lime-disaster-tweets/', 'Article moved', 'Continue to the article'],
  ];
  for (const [file, target, title, label] of routes) {
    const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, follow"><meta name="theme-color" content="#ffffff">
<meta http-equiv="refresh" content="0;url=${esc(target)}"><link rel="canonical" href="${esc(target)}">
<title>${esc(title)} | Noor Aldeen</title>
<style>${theme}\n${canvas}\nbody{margin:0;color:var(--ink);font-family:var(--font-body);line-height:1.6}main{max-width:50rem;margin:4rem auto;padding:1rem}a{color:var(--accent);text-underline-offset:.2em}a:focus-visible{outline:2px solid var(--focus-outline);outline-offset:4px}</style>
<script>${themeScript}</script></head>
<body><main><h1>${esc(title)}</h1><p><a href="${esc(target)}">${esc(label)}</a>.</p></main></body></html>\n`;
    await writeFile(resolve(dist, file), html);
  }
  console.log('Legacy URLs now use the shared theme and relative, base-safe redirects.');
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  finalizeBuild().catch(error => { console.error(error.message); process.exitCode = 1; });
}

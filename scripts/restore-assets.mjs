#!/usr/bin/env node
/** Restore only the original binary assets and dependency lockfile.
 *  Every website source file is already included in this package.
 *  Existing valid files are preserved; failed/partial downloads never replace them.
 *  Usage: node scripts/restore-assets.mjs [--check] [--from /path/to/original-repo]
 */
import { readFile, writeFile, mkdir, rename, lstat, rm } from 'node:fs/promises';
import { resolve, dirname, relative, isAbsolute, sep, extname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const manifest = JSON.parse(await readFile(new URL('./asset-manifest.json', import.meta.url), 'utf8'));

export function safePath(base, name) {
  if (typeof name !== 'string' || isAbsolute(name) || name.includes('\\') || name.includes('\0')) throw new Error('Invalid asset path');
  const target = resolve(base, name);
  const rel = relative(base, target);
  if (!rel || rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw new Error(`Unsafe asset path: ${name}`);
  return target;
}
export function validateAsset(name, data) {
  if (!Buffer.isBuffer(data) || data.length < 4) throw new Error(`Empty or truncated file: ${name}`);
  const ext = extname(name).toLowerCase();
  if (ext === '.json') {
    const lock = JSON.parse(data.toString('utf8'));
    if (!lock.lockfileVersion || !lock.packages?.['node_modules/astro']) throw new Error('Not an Astro dependency lockfile');
    if (!['latest', '6.2.1'].includes(lock.packages?.['']?.dependencies?.astro) || lock.packages['node_modules/astro'].version !== '6.2.1') throw new Error('Expected the upstream Astro 6.2.1 lockfile or its exact-version normalisation');
  } else if (ext === '.png') {
    if (!data.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) throw new Error(`Invalid PNG: ${name}`);
  } else if (ext === '.jpg' || ext === '.jpeg') {
    if (data[0] !== 255 || data[1] !== 216 || data[2] !== 255) throw new Error(`Invalid JPEG: ${name}`);
  } else if (ext === '.pdf') {
    if (data.subarray(0,5).toString() !== '%PDF-') throw new Error(`Invalid PDF: ${name}`);
  } else if (ext === '.ico') {
    if (!data.subarray(0,4).equals(Buffer.from([0,0,1,0]))) throw new Error(`Invalid icon: ${name}`);
  } else throw new Error(`Unexpected asset extension: ${name}`);
  return true;
}
async function checkNoSymlinks(base, name) {
  const target = safePath(base, name);
  let parent = target;
  while (parent !== base) {
    try { if ((await lstat(parent)).isSymbolicLink()) throw new Error(`Refusing symbolic link: ${parent}`); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    parent = dirname(parent);
  }
  return target;
}
export async function restore({ base = root, from, check = false, fetchImpl = fetch, log = console.log, files = manifest.files } = {}) {
  base = resolve(base);
  const missing = [];
  for (const name of files) {
    const destination = await checkNoSymlinks(base, name);
    try {
      const existing = await readFile(destination);
      validateAsset(name, existing);
      log(`Kept ${name}`);
      continue;
    } catch (error) {
      if (error.code !== 'ENOENT') throw new Error(`Existing file is invalid; inspect it before replacing: ${name}. ${error.message}`);
    }
    if (check) { missing.push(name); continue; }
    let data;
    if (from) {
      const source = await checkNoSymlinks(resolve(from), name);
      data = await readFile(source);
    } else {
      const url = `https://raw.githubusercontent.com/${manifest.repository}/${manifest.revision}/${name.split('/').map(encodeURIComponent).join('/')}`;
      let lastError;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const response = await fetchImpl(url, {signal: AbortSignal.timeout(30000), headers:{'User-Agent':'personal-site-asset-restore'}});
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const length = Number(response.headers.get('content-length') || 0);
          if (length > 40 * 1024 * 1024) throw new Error('File exceeds the 40 MB safety limit');
          data = Buffer.from(await response.arrayBuffer());
          if (data.length > 40 * 1024 * 1024) throw new Error('File exceeds the 40 MB safety limit');
          validateAsset(name, data);
          lastError = undefined;
          break;
        } catch (error) {
          lastError = error;
          if (attempt < 2) await new Promise(resolve => setTimeout(resolve, 350 * (attempt + 1)));
        }
      }
      if (lastError) throw new Error(`Could not download ${name}: ${lastError.message}. Use --from with an existing repository checkout instead.`);
    }
    validateAsset(name, data);
    await mkdir(dirname(destination), {recursive:true});
    const temp = `${destination}.${process.pid}.part`;
    try { await writeFile(temp, data, {flag:'wx'}); await rename(temp, destination); }
    finally { await rm(temp, {force:true}); }
    log(`Restored ${name}`);
  }
  if (missing.length) throw new Error(`Missing required assets:\n${missing.join('\n')}\nRun npm run setup, or copy the original files from your repository. No build has run.`);
}
async function main() {
  const args = process.argv.slice(2);
  let check = false, from;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--check') check = true;
    else if (args[i] === '--from' && args[i+1] && !args[i+1].startsWith('--')) from = args[++i];
    else throw new Error(`Unknown or incomplete argument: ${args[i]}`);
  }
  await restore({check, from});
  console.log(check ? 'All required original assets are present.' : 'Original media and lockfile are ready. Next: npm ci');
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}

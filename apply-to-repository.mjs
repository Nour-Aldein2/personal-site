#!/usr/bin/env node
/** Safely merge this release into an existing Git checkout. Does not push. */
import { access, readFile, readdir, mkdir, writeFile, lstat } from 'node:fs/promises';
import { resolve, relative, dirname, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const source = dirname(fileURLToPath(import.meta.url));
const excluded = new Set(['.git', 'node_modules', 'dist', '.astro', '.site-update-backups', '.env', '.DS_Store', 'test-results', 'playwright-report']);
async function exists(path) { try { await access(path); return true; } catch { return false; } }
async function walk(dir, prefix = '') {
  const result = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (excluded.has(entry.name) || entry.name.endsWith('.part')) continue;
    if (entry.isSymbolicLink()) throw new Error(`Source contains a symbolic link: ${entry.name}`);
    const name = prefix + entry.name;
    if (entry.isDirectory()) result.push(...await walk(resolve(dir, entry.name), name + '/'));
    else result.push(name);
  }
  return result;
}
async function checkPath(root, name) {
  const path = resolve(root, name);
  if (!path.startsWith(root + sep)) throw new Error(`Unsafe destination: ${name}`);
  let parent = path;
  while (parent !== root) {
    try { if ((await lstat(parent)).isSymbolicLink()) throw new Error(`Refusing symbolic link: ${parent}`); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    parent = dirname(parent);
  }
  return path;
}
export async function applyRelease({ from = source, target, dryRun = false, log = console.log } = {}) {
  if (!target) throw new Error('Usage: node apply-to-repository.mjs /path/to/personal-site [--dry-run]');
  from = resolve(from); target = resolve(target);
  if (target === from || target.startsWith(from + sep) || from.startsWith(target + sep)) throw new Error('Extract the release outside the target repository before applying it.');
  if (!await exists(resolve(target, '.git')) || !await exists(resolve(target, 'astro.config.mjs'))) throw new Error('The destination must be the root of an existing Astro Git checkout.');
  const manifest = JSON.parse(await readFile(resolve(from, 'scripts/asset-manifest.json'), 'utf8'));
  const protectedFiles = new Set(manifest.files);
  const files = await walk(from);
  const changed = [];
  for (const name of files) {
    const destination = await checkPath(target, name);
    const present = await exists(destination);
    if (present && protectedFiles.has(name)) continue;
    let data = await readFile(resolve(from, name));
    let before;
    if (present) {
      before = await readFile(destination);
      if (name === '.gitignore') {
        const rules = new Set(before.toString('utf8').split(/\r?\n/));
        data = Buffer.from([...rules, ...data.toString('utf8').split(/\r?\n/).filter(line => !rules.has(line))].join('\n') + '\n');
      }
      if (before.equals(data)) continue;
    }
    changed.push({ name, destination, data, before });
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backup = resolve(target, '.site-update-backups', stamp);
  if (!dryRun && changed.length) {
    // Validate the backup path too; never follow a pre-existing backup symlink.
    await checkPath(target, relative(target, resolve(backup, 'manifest.json')));
    for (const item of changed) {
      if (!item.before) continue;
      const saved = resolve(backup, 'files', item.name);
      await mkdir(dirname(saved), { recursive: true });
      await writeFile(saved, item.before);
    }
    await mkdir(backup, { recursive: true });
    await writeFile(resolve(backup, 'manifest.json'), JSON.stringify(changed.map(({ name, before }) => ({ path: name, existedBefore: Boolean(before) })), null, 2) + '\n');
    for (const item of changed) {
      await mkdir(dirname(item.destination), { recursive: true });
      await writeFile(item.destination, item.data);
    }
  }
  for (const { name } of changed) log(`${dryRun ? 'Would update' : 'Updated'} ${name}`);
  log(`${changed.length} file(s) ${dryRun ? 'would be updated' : 'updated'}. Existing media, lockfile, Git history and unrelated files were preserved.`);
  if (!dryRun && changed.length) log(`Backups: ${backup}`);
  return { changed: changed.map(item => item.name), backup: dryRun ? undefined : backup };
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const paths = args.filter(arg => arg !== '--dry-run');
  if (paths.length !== 1 || paths[0].startsWith('--')) {
    console.error('Usage: node apply-to-repository.mjs /path/to/personal-site [--dry-run]'); process.exitCode = 1;
  } else {
    applyRelease({ target: paths[0], dryRun }).then(() => {
      if (!dryRun) console.log('\nIn your repository, run:\n  npm run setup\n  npm ci\n  npm run check\n  npm run preview\nThen review git diff, commit and push. Nothing has been pushed or deployed.');
    }).catch(error => { console.error(error.message); process.exitCode = 1; });
  }
}

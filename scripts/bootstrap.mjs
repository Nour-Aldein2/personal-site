#!/usr/bin/env node
/** Prepare the source archive or an existing checkout without changing site content.
 * Only missing upstream media/lockfile are fetched, from one immutable revision.
 * The upstream lock's root dependency label is normalised to the exact installed
 * Astro version; its transitive versions and integrity hashes are not changed.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { restore, root } from './restore-assets.mjs';

export async function pinLockfile(base = root) {
  const packagePath = resolve(base, 'package.json');
  const lockPath = resolve(base, 'package-lock.json');
  const pkg = JSON.parse(await readFile(packagePath, 'utf8'));
  const raw = await readFile(lockPath, 'utf8');
  const lock = JSON.parse(raw);
  const expected = pkg.dependencies.astro;
  if (lock.packages?.['node_modules/astro']?.version !== expected) {
    throw new Error(`This release uses Astro ${expected}, but the existing lockfile resolves another version. Keep a backup and review the lockfile; it has not been replaced.`);
  }
  if (lock.packages[''].dependencies.astro !== expected) {
    lock.packages[''].dependencies.astro = expected;
    await writeFile(lockPath, JSON.stringify(lock, null, 2) + '\n');
  }
}

export async function bootstrap(args = []) {
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (major < 22 || (major === 22 && minor < 12)) throw new Error('Use Node.js 22.12 or newer.');
  let from;
  if (args.length) {
    if (args.length !== 2 || args[0] !== '--from') throw new Error('Usage: npm run setup -- [--from /path/to/existing/repository]');
    from = resolve(args[1]);
  }
  await restore({ from });
  await pinLockfile();
  console.log('Original media and the pinned lockfile are ready. Next: npm ci');
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  bootstrap(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1; });
}

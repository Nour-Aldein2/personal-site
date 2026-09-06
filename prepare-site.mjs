#!/usr/bin/env node
// Run from the extracted package. This helper never pushes or deploys.
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { existsSync } from 'node:fs';
const root = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
if (args.length && !(args.length === 2 && args[0] === '--from')) {
  console.error('Usage: node prepare-site.mjs [--from "/path/to/original-repository"]');
  process.exit(1);
}
const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 22 || (major === 22 && minor < 12)) {
  console.error('Please use Node.js 22.12 or newer.'); process.exit(1);
}
function run(command, commandArgs) {
  console.log(`\n> ${command} ${commandArgs.join(' ')}`);
  const result = spawnSync(command, commandArgs, {cwd:root, stdio:'inherit', shell:process.platform === 'win32' && command === 'npm'});
  if (result.error || result.status !== 0) {
    console.error(result.error?.message || `Stopped: command exited with status ${result.status}. Nothing was deployed.`);
    process.exit(result.status || 1);
  }
}
run(process.execPath, ['scripts/bootstrap.mjs', ...args]);
run('npm', ['ci']);
run('npm', ['test']);
run('npm', ['run', 'build']);
if (!existsSync(resolve(root,'dist/index.html'))) {
  console.error('Build did not produce dist/index.html. Nothing was deployed.'); process.exit(1);
}
console.log('\nBuild complete. Run npm run preview to review it. The contents of dist/ are ready for static hosting. Nothing has been pushed or deployed.');

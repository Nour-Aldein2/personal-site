import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, readdir, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { pinLockfile } from '../scripts/bootstrap.mjs';
import { applyRelease } from '../apply-to-repository.mjs';
import { checkBuild } from '../scripts/check-build.mjs';
import { finalizeBuild } from '../scripts/finalize-build.mjs';
async function temporary(t) { const dir = await mkdtemp(join(tmpdir(), 'site-release-')); t.after(() => rm(dir, { recursive: true, force: true })); return dir; }
async function put(base, name, data) { const file = join(base, name); await mkdir(dirname(file), { recursive: true }); await writeFile(file, data); }
const quiet = () => {};

test('lock normalisation pins only the root label and preserves transitive integrity', async t => {
  const base = await temporary(t);
  const packages = { '': { dependencies: { astro: 'latest' } }, 'node_modules/astro': { version: '6.2.1', integrity: 'original-integrity' }, 'node_modules/example': { version: '1.2.3', integrity: 'unchanged' } };
  await put(base, 'package.json', JSON.stringify({ dependencies: { astro: '6.2.1' } }));
  await put(base, 'package-lock.json', JSON.stringify({ lockfileVersion: 3, packages }));
  await pinLockfile(base);
  const updated = JSON.parse(await readFile(join(base, 'package-lock.json'), 'utf8'));
  assert.equal(updated.packages[''].dependencies.astro, '6.2.1');
  assert.deepEqual(updated.packages['node_modules/astro'], packages['node_modules/astro']);
  assert.deepEqual(updated.packages['node_modules/example'], packages['node_modules/example']);
  const first = await readFile(join(base, 'package-lock.json'));
  await pinLockfile(base); assert.deepEqual(await readFile(join(base, 'package-lock.json')), first);
});
test('an incompatible customised lock is not overwritten', async t => {
  const base = await temporary(t);
  await put(base, 'package.json', JSON.stringify({ dependencies: { astro: '6.2.1' } }));
  const lock = JSON.stringify({ packages: { '': { dependencies: { astro: '7.0.0' } }, 'node_modules/astro': { version: '7.0.0' } } });
  await put(base, 'package-lock.json', lock);
  await assert.rejects(pinLockfile(base), /another version/);
  assert.equal(await readFile(join(base, 'package-lock.json'), 'utf8'), lock);
});
async function fixture(t) {
  const from = await temporary(t), target = await temporary(t);
  await put(from, 'scripts/asset-manifest.json', JSON.stringify({ files: ['public/photo.jpg', 'package-lock.json'] }));
  await put(from, 'src/styles/theme.css', 'new theme');
  await put(from, '.github/workflows/deploy.yml', 'workflow');
  await put(from, '.gitignore', 'node_modules/\n.site-update-backups/\n');
  await put(from, 'public/photo.jpg', 'upstream media');
  await put(from, 'package-lock.json', 'upstream lock');
  await put(target, '.git/HEAD', 'ref: refs/heads/main');
  await put(target, 'astro.config.mjs', 'existing config');
  await put(target, 'src/styles/theme.css', 'old theme');
  await put(target, 'public/photo.jpg', 'custom media');
  await put(target, 'package-lock.json', 'custom lock');
  await put(target, '.gitignore', 'private-notes/\n');
  await put(target, 'dashboard/readme.txt', 'unrelated project');
  return { from, target };
}
test('applying the release preserves originals, history and unrelated files with backups', async t => {
  const { from, target } = await fixture(t);
  const result = await applyRelease({ from, target, log: quiet });
  assert.equal(await readFile(join(target, 'src/styles/theme.css'), 'utf8'), 'new theme');
  assert.equal(await readFile(join(result.backup, 'files/src/styles/theme.css'), 'utf8'), 'old theme');
  assert.equal(await readFile(join(target, 'public/photo.jpg'), 'utf8'), 'custom media');
  assert.equal(await readFile(join(target, 'package-lock.json'), 'utf8'), 'custom lock');
  assert.equal(await readFile(join(target, '.git/HEAD'), 'utf8'), 'ref: refs/heads/main');
  assert.equal(await readFile(join(target, 'dashboard/readme.txt'), 'utf8'), 'unrelated project');
  assert.match(await readFile(join(target, '.gitignore'), 'utf8'), /private-notes\//);
  assert.match(await readFile(join(target, '.gitignore'), 'utf8'), /site-update-backups/);
});
test('dry run reports changes without writing or backing up anything', async t => {
  const { from, target } = await fixture(t); const before = await readdir(target);
  await applyRelease({ from, target, dryRun: true, log: quiet });
  assert.deepEqual(await readdir(target), before);
  assert.equal(await readFile(join(target, 'src/styles/theme.css'), 'utf8'), 'old theme');
});
test('installer rejects nested target locations and non-repositories', async t => {
  const { from, target } = await fixture(t);
  await assert.rejects(applyRelease({ from, target: from, log: quiet }), /outside/);
  await assert.rejects(applyRelease({ from, target: join(from, 'nested'), log: quiet }), /outside/);
  const empty = await temporary(t);
  await assert.rejects(applyRelease({ from, target: empty, log: quiet }), /existing Astro Git checkout/);
});
test('installer refuses a target symlink before making changes', async t => {
  const { from, target } = await fixture(t); const outside = await temporary(t);
  await rm(join(target, 'src'), { recursive: true });
  await symlink(outside, join(target, 'src'));
  await assert.rejects(applyRelease({ from, target, log: quiet }), /symbolic link/);
  assert.deepEqual(await readdir(outside), []);
});
async function buildFixture(t, base = '/') {
  const dist = await temporary(t);
  const html = path => `<!doctype html><html lang="en" data-site-design="seamless-brown-grid"><head><title>Page</title></head><body><header class="site-header"></header><main id="main-content"><a href="${base}articles/">Articles</a></main><script>/* theme-preference */</script></body></html>`;
  for (const name of ['index.html','articles/index.html','publications/index.html','cv/index.html','404.html']) await put(dist, name, html(name));
  for (const name of ['robots.txt','sitemap.xml','Noor_CV.pdf']) await put(dist, name, 'fixture');
  return dist;
}
test('generated-page validator accepts custom-domain and project-base fixtures', async t => {
  const root = await buildFixture(t); const result = await checkBuild({ dist: root });
  assert.equal(result.htmlPages, 5); assert.equal(await readFile(join(root, 'CNAME'), 'utf8'), 'nourd.uk\n');
  const dist = await buildFixture(t, '/personal-site/');
  await put(dist, 'CNAME', 'nourd.uk');
  await checkBuild({ dist, site: 'https://Nour-Aldein2.github.io', base: '/personal-site/' });
  assert.ok(!(await readdir(dist)).includes('CNAME'));
});
test('generated-page validator fails on missing local images instead of deploying broken pages', async t => {
  const dist = await buildFixture(t); const file = join(dist, 'index.html');
  await writeFile(file, (await readFile(file, 'utf8')).replace('</main>', '<img src="/missing.png"></main>'));
  await assert.rejects(checkBuild({ dist }), /missing local target/);
});
test('generated-page validator catches a root link that escapes a project base', async t => {
  const dist = await buildFixture(t, '/personal-site/'); const file = join(dist, 'index.html');
  await writeFile(file, (await readFile(file, 'utf8')).replace('/personal-site/articles/', '/articles/'));
  await assert.rejects(checkBuild({ dist, site: 'https://example.github.io', base: '/personal-site/' }), /escapes BASE_PATH/);
});
test('legacy redirects receive the actual shared theme and saved-theme initializer', async t => {
  const dist = await temporary(t); await finalizeBuild({ dist });
  for (const name of ['cv.html','post_explain_BERT_LIME.html']) {
    const html = await readFile(join(dist, name), 'utf8');
    assert.match(html, /--accent: #7C2710/); assert.match(html, /--accent: #CC9075/);
    assert.match(html, /theme-preference/); assert.match(html, /noindex, follow/);
  }
  assert.match(await readFile(join(dist, 'cv.html'), 'utf8'), /url=Noor_CV\.pdf/);
});

test('generated pages must use this release shared layout, not the old one', async t => {
  const dist = await buildFixture(t);
  const file = join(dist, 'publications/index.html');
  await writeFile(file, (await readFile(file, 'utf8')).replace(' data-site-design="seamless-brown-grid"', ''));
  await assert.rejects(checkBuild({ dist }), /seamless-brown-grid/);
});

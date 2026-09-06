import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, writeFile, readdir, rm, symlink} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, dirname} from 'node:path';
import {safePath, validateAsset, restore, manifest} from '../scripts/restore-assets.mjs';
const png = Buffer.from([137,80,78,71,13,10,26,10,0,0,0,0]);
const pdf = Buffer.from('%PDF-1.7\noriginal');
const lock = Buffer.from(JSON.stringify({lockfileVersion:3,packages:{'':{dependencies:{astro:'latest'}},'node_modules/astro':{version:'6.2.1'}}}));
const quiet = () => {};
async function temporary(t) { const dir=await mkdtemp(join(tmpdir(),'noor-site-test-')); t.after(()=>rm(dir,{recursive:true,force:true})); return dir; }
async function put(base,name,data) {const p=join(base,name);await mkdir(dirname(p),{recursive:true});await writeFile(p,data);}

test('safePath permits nested files and rejects traversal/absolute paths',()=>{
 const base=join(tmpdir(),'site');
 assert.equal(safePath(base,'public/images/a.png'),join(base,'public','images','a.png'));
 for(const bad of ['../escape.png',join(base,'absolute.png'),'.', 'public\\bad.png','bad\0.png']) assert.throws(()=>safePath(base,bad));
});
test('checks original binary signatures and the matching dependency lock',()=>{
 for(const [name,data] of [['a.png',png],['a.jpg',Buffer.from([255,216,255,224])],['a.pdf',pdf],['a.ico',Buffer.from([0,0,1,0])],['package-lock.json',lock]]) assert.equal(validateAsset(name,data),true);
});
test('rejects HTML error pages, truncated files and the wrong lock',()=>{
 for(const name of ['a.png','a.jpg','a.pdf','a.ico']) assert.throws(()=>validateAsset(name,Buffer.from('<html>Not found</html>')));
 assert.throws(()=>validateAsset('a.png',Buffer.from([137])));
 assert.throws(()=>validateAsset('package-lock.json',Buffer.from('{"lockfileVersion":3,"packages":{}}')));
 assert.throws(()=>validateAsset('unexpected.js',pdf));
});
test('copies missing originals from a local checkout without network',async t=>{
 const base=await temporary(t),from=await temporary(t);await put(from,'public/a.png',png);await put(from,'package-lock.json',lock);
 await restore({base,from,files:['public/a.png','package-lock.json'],log:quiet,fetchImpl:()=>{throw Error('network not expected')}});
 assert.deepEqual(await readFile(join(base,'public/a.png')),png);
 assert.deepEqual(await readFile(join(base,'package-lock.json')),lock);
});
test('preserves existing valid media, including customised files',async t=>{
 const base=await temporary(t);const custom=Buffer.concat([png,Buffer.from('custom')]);await put(base,'public/a.png',custom);
 await restore({base,files:['public/a.png'],log:quiet,fetchImpl:()=>{throw Error('network not expected')}});
 assert.deepEqual(await readFile(join(base,'public/a.png')),custom);
});
test('check mode fails cleanly without writing or fetching missing files',async t=>{
 const base=await temporary(t);
 await assert.rejects(restore({base,check:true,files:['public/a.png'],log:quiet,fetchImpl:()=>{throw Error('network not expected')}}),/Missing required assets/);
 assert.deepEqual(await readdir(base),[]);
});
test('check mode accepts a complete local asset set',async t=>{
 const base=await temporary(t);await put(base,'public/a.png',png);
 await restore({base,check:true,files:['public/a.png'],log:quiet});
});
test('downloads from the pinned revision and commits a validated file atomically',async t=>{
 const base=await temporary(t);let calls=0;
 await restore({base,files:['public/a.png'],log:quiet,fetchImpl:async(url)=>{
   calls++;assert.ok(url.includes(`/${manifest.revision}/public/a.png`));return new Response(png,{status:200});
 }});
 assert.equal(calls,1);assert.deepEqual(await readFile(join(base,'public/a.png')),png);
 assert.deepEqual(await readdir(join(base,'public')),['a.png']);
});
test('a failed download never creates a production asset',async t=>{
 const base=await temporary(t);let calls=0;
 await assert.rejects(restore({base,files:['public/a.png'],log:quiet,fetchImpl:async()=>{calls++;return new Response('<html>no</html>',{status:404})}}),/Could not download/);
 assert.equal(calls,3);assert.deepEqual(await readdir(base),[]);
});
test('invalid existing files are never silently overwritten',async t=>{
 const base=await temporary(t);const invalid=Buffer.from('not an image');await put(base,'public/a.png',invalid);
 await assert.rejects(restore({base,files:['public/a.png'],log:quiet}),/Existing file is invalid/);
 assert.deepEqual(await readFile(join(base,'public/a.png')),invalid);
});
test('refuses a destination symlink rather than writing outside the project',async t=>{
 const base=await temporary(t),outside=await temporary(t);await symlink(outside,join(base,'public'),process.platform==='win32'?'junction':'dir');
 await assert.rejects(restore({base,check:true,files:['public/a.png'],log:quiet}),/Refusing symbolic link/);
});
test('an invalid local source is rejected before writing',async t=>{
 const base=await temporary(t),from=await temporary(t);await put(from,'public/a.png',Buffer.from('invalid png'));
 await assert.rejects(restore({base,from,files:['public/a.png'],log:quiet}),/Invalid PNG/);
 assert.deepEqual(await readdir(base),[]);
});

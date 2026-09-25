// CPU-only browser regression. All API responses are test doubles, not model evidence.
// Run: node tests/creator_selected_diff_browser.cjs [screenshot.png]
// Optional: CREATOR_PLAYWRIGHT_MODULE, CREATOR_BROWSER_EXECUTABLE, CREATOR_TEST_PYTHON.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const {chromium} = require(process.env.CREATOR_PLAYWRIGHT_MODULE || 'playwright');
const source = JSON.parse(execFileSync(process.env.CREATOR_TEST_PYTHON || 'python', ['-c',
  'import json; from training.creator.page import HTML,CSS,JS; print(json.dumps(dict(HTML=HTML,CSS=CSS,JS=JS)))'],
  {cwd:path.resolve(__dirname, '..'), encoding:'utf8'}));
const plan = frames => ({status:'ready', explanation:'CPU UI TEST ONLY',
  action_segments:[{frames:240-frames, keys:['W']},{frames, keys:['W','I']}], goals:[]});
// Each completed version retries a failed parent with identical input. The two
// completed versions differ by 60 I frames even though their parent diffs are 0.
const versions = [
  {version_id:'test-original-failed', plan:plan(120), job_status:'failed'},
  {version_id:'test-original-retry', parent_version:'test-original-failed', plan:plan(120), job_status:'completed'},
  {version_id:'test-edited-failed', parent_version:'test-original-retry', plan:plan(60), job_status:'failed'},
  {version_id:'test-edited-retry', parent_version:'test-edited-failed', plan:plan(60), job_status:'completed'},
].map((v, i) => ({...v, job_id:'test-job-'+i, text:'CPU UI TEST ONLY — not a model run'}));
const fixture = {session_id:'test-session', scene_id:'test-scene', seed:42, versions};

(async () => {
  const browser = await chromium.launch({headless:true, args:['--disable-gpu'],
    ...(process.env.CREATOR_BROWSER_EXECUTABLE ? {executablePath:process.env.CREATOR_BROWSER_EXECUTABLE} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:950}});
    const posts = [], errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => localStorage.setItem('interactworld.creator.session', 'test-session'));
    await page.route('**/*', async route => {
      const request = route.request(), pathname = new URL(request.url()).pathname;
      if (request.method() === 'POST') {posts.push(pathname); return route.fulfill({status:500, json:{error:'No mutation expected'}})}
      const assets = {'/':['text/html',source.HTML], '/style.css':['text/css',source.CSS], '/app.js':['text/javascript',source.JS]};
      if (assets[pathname]) return route.fulfill({contentType:assets[pathname][0],body:assets[pathname][1]});
      if (pathname === '/api/config') return route.fulfill({json:{csrf_token:'test-token', generation_enabled:false,
        scenes:[{scene_id:'test-scene',prompt:'CPU UI TEST ONLY',initial_url:'/test-image.svg'}], planner_kind:'rule_fallback',
        observer_kind:'manual', visual_revision_enabled:false}});
      if (pathname === '/api/sessions') return route.fulfill({json:{sessions:[fixture]}});
      if (pathname === '/api/jobs') return route.fulfill({json:{jobs:versions.map(v=>({job_id:v.job_id,status:v.job_status}))}});
      if (pathname === '/test-image.svg') return route.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="832" height="480"/>'});
      // No test video is fabricated; this regression only checks input comparison.
      return route.fulfill({status:404,body:'UI test does not supply generated media'});
    });
    await page.goto('http://127.0.0.1:9859/');
    await page.waitForFunction(() => document.querySelectorAll('.version:not([hidden])').length === 2);
    const panel = page.locator('#selected-input-diff');
    assert.match(await panel.innerText(), /0 \/ 240/);
    await page.locator('#compare-left').selectOption('test-original-retry');
    assert.match(await panel.innerText(), /60 \/ 240/);
    assert.match(await panel.innerText(), /120 → 60 帧/);
    assert.match(await panel.innerText(), /\+0 \/ −60/);
    assert.match(await panel.innerText(), /移动输入 W\/A\/S\/D：逐帧不变/);
    for (const id of ['test-original-retry','test-edited-retry']) {
      assert.match(await page.locator(`[data-version-id="${id}"] .input-diff`).innerText(), /0 \/ 240/);
    }
    await page.locator('#compare-right').selectOption('test-original-failed');
    assert.match(await panel.innerText(), /0 \/ 240/);
    assert.equal(await panel.locator('tbody tr').count(), 0);
    await page.locator('#compare-left').selectOption('test-edited-retry');
    assert.match(await panel.innerText(), /60 → 120 帧/);
    assert.match(await panel.innerText(), /\+60 \/ −0/);
    await page.locator('#compare-right').selectOption('');
    assert.match(await panel.innerText(), /请选择左右两个版本/);
    await page.locator('#compare-left').selectOption('test-original-retry');
    await page.locator('#compare-right').selectOption('test-edited-retry');
    await page.evaluate(() => refresh());
    assert.match(await panel.innerText(), /60 \/ 240/);
    assert.deepEqual(posts, []);
    assert.deepEqual(errors, []);
    let screenshot = null;
    if (process.argv[2]) {
      screenshot = path.resolve(process.argv[2]);
      await panel.evaluate(node => {
        const note = document.createElement('p');
        note.textContent = 'CPU UI TEST · retry-chain fixture · not generation evidence';
        note.style.color = '#e8c997';
        node.prepend(note);
      });
      await panel.screenshot({path:screenshot});
    }
    console.log(JSON.stringify({test_only:true,passed:true,checks:['retry parent=0 / selected pair=60','selection changes and reverses delta','missing side clears delta','poll preserves selected diff','no POST'],screenshot,errors,posts}));
  } finally {await browser.close()}
})().catch(error => {console.error(error);process.exitCode=1});

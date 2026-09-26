// CPU-only browser contract check. API responses are doubles, not model evidence.
// Run: node tests/creator_manual_review_browser.cjs
// CREATOR_PLAYWRIGHT_MODULE / CREATOR_BROWSER_EXECUTABLE / CREATOR_TEST_PYTHON
// may point to existing installed tooling; no browser download is required.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const {chromium} = require(process.env.CREATOR_PLAYWRIGHT_MODULE || 'playwright');
const source = JSON.parse(execFileSync(process.env.CREATOR_TEST_PYTHON || 'python', ['-c',
  'import json; from training.creator.page import HTML,CSS,JS; print(json.dumps(dict(HTML=HTML,CSS=CSS,JS=JS)))'],
  {cwd:path.resolve(__dirname, '..'), encoding:'utf8'}));
const model = {source:'model_assessment', verdict:'uncertain', evidence:'MODEL TEST OBSERVATION',
  decision:'ask_user', created_at:1, reference_used:false};
const oldHuman = {source:'human', verdict:'unsatisfied', evidence:'OLDER HUMAN TEST OBSERVATION',
  decision:'ask_user', created_at:2, previous_assessment:model};
const versions = ['original','edited'].map((id, i) => ({version_id:id, job_id:'job-'+i,
  text:'CPU UI TEST ONLY', job_status:'completed', created_at:i+1,
  plan:{status:'ready', explanation:'CPU UI TEST ONLY', action_segments:[{frames:240,keys:['W']}],goals:[]},
  review:i===1?oldHuman:null, review_history:i===1?[model]:[]}));
const fixture = {session_id:'manual-test', scene_id:'test-scene', seed:42, versions};

(async () => {
  const browser = await chromium.launch({headless:true,args:['--disable-gpu'],
    ...(process.env.CREATOR_BROWSER_EXECUTABLE?{executablePath:process.env.CREATOR_BROWSER_EXECUTABLE}:{})});
  try {
    const page = await browser.newPage(), posts = [], errors = [];
    let reviewAttempts = 0, plannerKind = 'external_proposal_validated';
    page.on('pageerror', error=>errors.push(error.message));
    await page.addInitScript(()=>localStorage.setItem('interactworld.creator.session','manual-test'));
    await page.route('**/*', async route=>{
      const request=route.request(), pathname=new URL(request.url()).pathname;
      if(request.method()==='POST'){
        const body=request.postDataJSON();posts.push({pathname,body});
        if(pathname==='/api/review'){
          reviewAttempts++;
          if(reviewAttempts===1)return route.fulfill({status:503,json:{error:'SIMULATED SAVE FAILURE'}});
          const version=versions.find(v=>v.version_id===body.version_id);
          version.review_history.push(version.review);
          version.review={source:'human',verdict:body.verdict,evidence:body.evidence,criteria:body.criteria,
            decision:'ask_user',created_at:3,previous_assessment:model};
          return route.fulfill({json:fixture});
        }
        return route.fulfill({status:503,json:{error:'TEST ONLY — no model task launched'}});
      }
      const assets={'/':['text/html',source.HTML],'/style.css':['text/css',source.CSS],'/app.js':['text/javascript',source.JS]};
      if(assets[pathname])return route.fulfill({contentType:assets[pathname][0],body:assets[pathname][1]});
      if(pathname==='/api/config')return route.fulfill({json:{csrf_token:'test-token',generation_enabled:false,
        rule_planner_available:true,planner_kind:plannerKind,observer_kind:'local_qwen',
        visual_revision_enabled:false,scenes:[{scene_id:'test-scene',prompt:'CPU UI TEST ONLY',initial_url:'/initial.svg'}]}});
      if(pathname==='/api/sessions')return route.fulfill({json:{sessions:[fixture]}});
      if(pathname==='/api/jobs')return route.fulfill({json:{jobs:versions.map(v=>({job_id:v.job_id,status:v.job_status}))}});
      if(pathname==='/initial.svg')return route.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="832" height="480"/>'});
      return route.fulfill({status:404,body:'No generated media in this UI test'});
    });
    await page.goto('http://127.0.0.1:9860/');
    await page.waitForFunction(()=>document.querySelectorAll('.version:not([hidden])').length===2);
    assert.equal(await page.locator('#planner-mode').inputValue(),'local_model');
    assert.equal(await page.locator('#planner-mode option[value="local_model"]').isDisabled(),false);
    const card=page.locator('[data-version-id="edited"]');
    assert.match(await card.locator('.human-review-panel').innerText(),/OLDER HUMAN TEST OBSERVATION/);
    assert.match(await card.locator('.model-review-panel').innerText(),/MODEL TEST OBSERVATION/);
    assert.equal(await card.locator('[data-criterion="movement_response"]').inputValue(),'');
    assert.equal(await card.locator('[data-criterion="camera_response"]').inputValue(),'');
    const evidence='5–8 秒观察测试；不是实际人工评分。', instruction='保留前进，只缩短抬头';
    await card.locator('.human-verdict').selectOption('unsatisfied');
    await card.locator('.human-evidence').fill(evidence);
    await card.locator('[data-criterion="camera_response"]').selectOption('unsatisfied');
    await card.locator('.review-edit').fill(instruction);
    await page.evaluate(()=>refresh());
    assert.equal(await card.locator('.human-evidence').inputValue(),evidence);
    await card.locator('[data-action="review"]').click();
    await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('SIMULATED SAVE FAILURE'));
    assert.equal(await card.locator('.human-evidence').inputValue(),evidence);
    assert.equal(await card.locator('.review-edit').inputValue(),instruction);
    await page.reload();
    await page.waitForFunction(()=>document.querySelectorAll('.version:not([hidden])').length===2);
    assert.equal(await card.locator('.human-evidence').inputValue(),evidence);
    assert.equal(await card.locator('.review-edit').inputValue(),instruction);
    assert.equal(await card.locator('[data-criterion="camera_response"]').inputValue(),'unsatisfied');
    await card.locator('[data-action="review"]').click();
    await page.waitForFunction(()=>document.querySelector('#notice').textContent.startsWith('人工反馈已保存'));
    assert.match(await card.locator('.human-review-panel').innerText(),/5–8 秒观察测试/);
    assert.match(await card.locator('.model-review-panel').innerText(),/MODEL TEST OBSERVATION/);
    assert.equal(await card.locator('.human-review-panel .historical-review').count(),1);
    assert.deepEqual(posts[1].body.criteria,{camera_response:'unsatisfied'});
    const count=posts.length;
    await card.locator('[data-action="use-review-edit"]').click();
    assert.equal(await page.locator('#instruction').inputValue(),instruction);
    assert.equal(posts.length,count);
    assert.match(await page.locator('#notice').innerText(),/尚未编排，也未提交生成/);
    const original=page.locator('[data-version-id="original"]');
    await original.locator('.review-edit').fill('保留前进，取消抬头');
    await original.locator('[data-action="use-review-edit"]').click();
    assert.match(await page.locator('#notice').innerText(),/编排会基于版本 2/);
    assert.equal(posts.length,count);
    await page.locator('#planner-mode').selectOption('rule_fallback');
    assert.equal(posts.length,count);
    await page.locator('#plan').click();
    await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('TEST ONLY — no model task launched'));
    assert.equal(posts.at(-1).pathname,'/api/plan');
    assert.equal(posts.at(-1).body.planner,'rule_fallback');
    plannerKind='rule_fallback';
    await page.reload();
    await page.waitForFunction(()=>document.querySelectorAll('.version:not([hidden])').length===2);
    assert.equal(await page.locator('#planner-mode').inputValue(),'rule_fallback');
    assert.equal(await page.locator('#planner-mode option[value="local_model"]').isDisabled(),true);
    await page.locator('#instruction').fill('一直前进');
    await page.locator('#plan').click();
    await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('TEST ONLY — no model task launched'));
    assert.equal(posts.at(-1).body.planner,'rule_fallback');
    assert.equal(posts.some(item=>item.pathname==='/api/generate'),false);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({test_only:true,passed:true,checks:[
      'separate human/model source panels and history','criteria initially unset','failed save preserves draft',
      'reload preserves draft','review success keeps old model observation','edit-fill never POSTs',
      'historical edit warns last-ready baseline','explicit rule planner request; no implicit generation',
      'model deployment defaults local model; CPU-only deployment defaults rule and disables model'],posts:posts.map(p=>p.pathname),errors}));
  } finally {await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});

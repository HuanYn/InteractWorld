// CPU-only real-browser UI contract test. Every network response is a TEST DOUBLE.
// No API request reaches a server; no model, video generation or GPU job is run.
// Use existing tooling via CREATOR_PLAYWRIGHT_MODULE, CREATOR_BROWSER_EXECUTABLE,
// and optionally CREATOR_TEST_PYTHON. This script does not download a browser.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const {chromium} = require(process.env.CREATOR_PLAYWRIGHT_MODULE || 'playwright');
const source = JSON.parse(execFileSync(process.env.CREATOR_TEST_PYTHON || 'python', ['-c',
  'import json; from training.creator.page import HTML,CSS,JS; print(json.dumps(dict(HTML=HTML,CSS=CSS,JS=JS)))'],
  {cwd:path.resolve(__dirname, '..'), encoding:'utf8'}));

const plan = keys => ({status:'ready', explanation:'CPU BROWSER TEST ONLY',
  action_segments:[{frames:240,keys}], goals:['TEST GOAL'], edit_scope:'all', preserved:false});
const version = (id, index, keys=['W'], completed=true) => ({version_id:id,
  created_at:index, text:'CPU BROWSER TEST ONLY', plan:plan(keys),
  job_id:completed?'job-'+id:null, ...(completed?{job_status:'completed'}:{})});
const first = {session_id:'base-test', scene_id:'test-scene', seed:42,
  versions:[version('v-a',1),version('v-b',2,['W','I']),
    {version_id:'v-clarify',created_at:3,text:'TEST CLARIFICATION',
      plan:{status:'clarify',explanation:'TEST ONLY',action_segments:[],goals:[]}}]};
const second = {session_id:'other-test',scene_id:'test-scene',seed:43,
  versions:[version('other-a',1)]};

(async()=>{
  const browser = await chromium.launch({headless:true,args:['--disable-gpu'],
    ...(process.env.CREATOR_BROWSER_EXECUTABLE?{executablePath:process.env.CREATOR_BROWSER_EXECUTABLE}:{})});
  try {
    const page = await browser.newPage(), posts=[], errors=[], leakedRequests=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.addInitScript(()=>localStorage.setItem('interactworld.creator.session','base-test'));
    await page.route('**/*',async route=>{
      const request=route.request(),url=new URL(request.url()),pathname=url.pathname;
      if(url.origin!=='http://127.0.0.1:9861'){
        leakedRequests.push(request.url());return route.abort('blockedbyclient');
      }
      if(request.method()==='POST'){
        const body=request.postDataJSON();posts.push({pathname,body});
        if(pathname==='/api/plan'){
          assert.equal(body.base_version_id,'v-a');
          const result={...version('v-planned',5,['W'],false),base_version_id:'v-a',parent_version:'v-a',
            planning_trace:{schema_version:1,max_revisions:1,model_load_count:1,revision_count:1,outcome:'repaired',
              attempts:[{attempt:1,status:'clarify',elapsed_seconds:1.2,
                feedback:{code:'missing_actions',repairable:true,message:'<img src=x onerror="globalThis.traceInjected=true">'}},
                {attempt:2,status:'ready',elapsed_seconds:1.3,feedback:{code:'ok',repairable:false,message:'TEST input check passed'}}]},
            plan:{...plan(['W']),edit_scope:'camera',preserved:true,edit_patch:{schema_version:1,fps:16,
              total_frames:240,edits:[{op:'replace_intervals',key:'I',intervals:[{start_seconds:8,end_seconds:10}]}],
              protected_keys:['W','A','S','D','J','K','L']}}};
          first.versions.push(result,version('v-newer',6,['D'],false));
          return route.fulfill({json:{...first,planned_version_id:'v-planned'}});
        }
        if(pathname==='/api/generate'){
          // Verify the explicit button's payload, then reject the TEST submission.
          return route.fulfill({status:503,json:{error:'TEST ONLY — no generation submitted'}});
        }
        return route.fulfill({status:503,json:{error:'TEST ONLY — unexpected operation blocked'}});
      }
      const assets={'/':['text/html',source.HTML],'/style.css':['text/css',source.CSS],'/app.js':['text/javascript',source.JS]};
      if(assets[pathname])return route.fulfill({contentType:assets[pathname][0],body:assets[pathname][1]});
      if(pathname==='/api/config')return route.fulfill({json:{csrf_token:'test-token',generation_enabled:true,
        rule_planner_available:true,planner_kind:'local_model',observer_kind:'human_only',visual_revision_enabled:false,
        scenes:[{scene_id:'test-scene',prompt:'CPU BROWSER TEST ONLY',initial_url:'/initial.svg'}]}});
      if(pathname==='/api/sessions')return route.fulfill({json:{sessions:[first,second]}});
      if(pathname==='/api/jobs')return route.fulfill({json:{jobs:[first,second].flatMap(session=>session.versions)
        .filter(v=>v.job_id).map(v=>({job_id:v.job_id,status:'completed'}))}});
      if(pathname==='/initial.svg')return route.fulfill({contentType:'image/svg+xml',
        body:'<svg xmlns="http://www.w3.org/2000/svg" width="832" height="480"/>'});
      return route.fulfill({status:404,body:'No generated media in this CPU UI test'});
    });
    await page.goto('http://127.0.0.1:9861/');
    await page.waitForFunction(()=>document.querySelector('#edit-base-version').options.length===3);
    assert.equal(await page.locator('#edit-base-version').inputValue(),'');
    assert.match(await page.locator('#edit-base').innerText(),/动态跟随.*版本 2/);
    assert.deepEqual(await page.locator('#edit-base-version option').evaluateAll(nodes=>nodes.map(n=>n.value)),['','v-a','v-b']);

    await page.locator('#edit-base-version').selectOption('v-a');
    first.versions.push(version('v-poll',4,['S'],false));
    await page.evaluate(()=>refresh());
    assert.equal(await page.locator('#edit-base-version').inputValue(),'v-a');
    assert.match(await page.locator('#edit-base').innerText(),/已固定.*版本 1/);
    await page.reload();
    await page.waitForFunction(()=>document.querySelector('#edit-base-version').value==='v-a');
    await page.locator('#edit-base-version').selectOption('');
    assert.match(await page.locator('#edit-base').innerText(),/动态跟随.*版本 4/);
    await page.locator('#edit-base-version').selectOption('v-a');
    await page.locator('#compare-left').selectOption('v-b');
    await page.locator('#compare-right').selectOption('v-poll');
    assert.equal(await page.locator('#edit-base-version').inputValue(),'v-a');

    await page.locator('#session').selectOption('other-test');
    assert.equal(await page.locator('#edit-base-version').inputValue(),'');
    assert.match(await page.locator('#edit-base').innerText(),/other-a/);
    assert.deepEqual(await page.locator('#edit-base-version option').evaluateAll(nodes=>nodes.map(n=>n.value)),['','other-a']);
    await page.locator('#new-session').click();
    assert.equal(await page.locator('#edit-base-version').inputValue(),'');
    assert.equal(await page.locator('#edit-base-version').isDisabled(),true);
    await page.locator('#session').selectOption('base-test');
    assert.equal(await page.locator('#edit-base-version').inputValue(),'v-a');

    await page.locator('#compare-left').selectOption('v-a');
    await page.locator('#compare-right').selectOption('v-b');
    await page.locator('#edit-base-version').selectOption('v-b');
    const card=page.locator('[data-version-id="v-a"]');
    const instruction='保持前进不变，只在第8到10秒抬头';
    await card.locator('.review-edit').fill(instruction);
    await card.locator('[data-action="use-review-edit"]').click();
    assert.equal(await page.locator('#edit-base-version').inputValue(),'v-a');
    assert.equal(await page.locator('#instruction').inputValue(),instruction);
    assert.match(await page.locator('#notice').innerText(),/固定为此历史版本/);
    assert.equal(posts.length,0);
    await page.evaluate(()=>refresh());
    assert.equal(await card.locator('.review-edit').inputValue(),instruction);

    await page.locator('#plan').click();
    await page.waitForFunction(()=>document.querySelector('#notice').textContent.startsWith('动作计划已保存'));
    assert.equal(posts.length,1);
    assert.equal(posts[0].pathname,'/api/plan');
    assert.equal(posts[0].body.base_version_id,'v-a');
    assert.equal(posts[0].body.text,instruction);
    assert.match(await page.locator('#plan-details').innerText(),/v-planne/);
    assert.match(await page.locator('#plan-details').innerText(),/实际编辑基线\s*v-a/);
    assert.match(await page.locator('#plan-details').innerText(),/replace_intervals/);
    assert.doesNotMatch(await page.locator('#plan-details').innerText(),/v-newer/);
    const trace=page.locator('#plan-trace');
    assert.match(await trace.innerText(),/初次检查/);
    assert.match(await trace.innerText(),/一次修订检查/);
    assert.match(await trace.innerText(),/服务端最终计划：可执行/);
    assert.match(await trace.innerText(),/不是视频效果/);
    assert.match(await trace.innerText(),/<img src=x/);
    assert.equal(await trace.locator('img').count(),0);
    assert.equal(await page.evaluate(()=>globalThis.traceInjected===undefined),true);
    await page.locator('#compare-right').selectOption('v-planned');
    assert.match(await page.locator('[data-version-id="v-planned"] .planning-trace').innerText(),/一次修订检查/);
    assert.equal(posts.length,1);
    await page.evaluate(()=>refresh());
    await page.reload();
    await page.waitForFunction(()=>document.querySelector('#plan-details').textContent.includes('v-planne'));
    assert.equal(await page.locator('#edit-base-version').inputValue(),'v-a');
    assert.equal(await page.locator('#generate').isEnabled(),true);
    await page.locator('#generate').click();
    await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('TEST ONLY — no generation submitted'));
    assert.deepEqual(posts.map(p=>p.pathname),['/api/plan','/api/generate']);
    assert.equal(posts[1].body.version_id,'v-planned');
    assert.deepEqual(errors,[]);
    assert.deepEqual(leakedRequests,[]);
    console.log(JSON.stringify({test_only:true,real_browser:true,passed:true,browser:browser.version(),
      checks:['ready-only baseline options','dynamic latest versus fixed historical baseline',
        'poll/reload preserve fixed selection','comparison independent from editing baseline',
        'session switch and new session do not leak baseline IDs','historical edit button chooses that exact version',
        'historical edit creates no API POST','manual draft survives polling','plan sends displayed base_version_id',
        'planned_version_id controls preview across polling/reload','explicit generation payload uses displayed planned version',
        'repair trace appears in plan and version card without auto generation','feedback HTML remains text; no XSS execution',
        'all requests intercepted; no real API/model/GPU invocation'],posts:posts.map(p=>({path:p.pathname,
          base_version_id:p.body.base_version_id,version_id:p.body.version_id})),errors,leakedRequests}));
  }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});

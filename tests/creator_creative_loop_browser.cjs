// CPU-only browser fixture: every request is intercepted. No real video/model/GPU evidence.
// Uses the same existing CREATOR_* tooling variables as creator_base_browser.cjs.
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const {chromium} = require(process.env.CREATOR_PLAYWRIGHT_MODULE || 'playwright');
const source = JSON.parse(execFileSync(process.env.CREATOR_TEST_PYTHON || 'python', ['-c',
  'import json; from training.creator.page import HTML,CSS,JS; print(json.dumps(dict(HTML=HTML,CSS=CSS,JS=JS)))'],
  {cwd:path.resolve(__dirname, '..'), encoding:'utf8'}));
const ready = {status:'ready', explanation:'CPU FIXTURE ONLY',
  action_segments:[{frames:240,keys:['W']}], goals:[]};
const version = (id, completed=false) => ({version_id:id, text:'CPU FIXTURE ONLY', plan:ready,
  ...(completed ? {job_id:'job-'+id,job_status:'completed'} : {})});

(async()=>{
  const browser = await chromium.launch({headless:true,args:['--disable-gpu'],
    ...(process.env.CREATOR_BROWSER_EXECUTABLE ? {executablePath:process.env.CREATOR_BROWSER_EXECUTABLE} : {})});
  try {
    const checks=[];
    for(const outcome of ['ready','clarify','missing-id','mismatch','error']){
      const fixture={session_id:'creative-test',scene_id:'mountain-fixture',seed:42,
        versions:[version('original',true),version('recent-video',true),version('old-preview')]};
      const page=await browser.newPage({viewport:{width:1100,height:900}}),posts=[],errors=[],leaks=[];
      page.on('pageerror',error=>errors.push(error.message));
      await page.addInitScript(()=>localStorage.setItem('interactworld.creator.session','creative-test'));
      await page.route('**/*',async route=>{
        const request=route.request(),url=new URL(request.url()),pathname=url.pathname;
        if(url.origin!=='http://127.0.0.1:9862'){
          leaks.push(request.url());return route.abort('blockedbyclient');
        }
        if(request.method()==='POST'){
          const body=request.postDataJSON();posts.push({pathname,body});
          if(pathname==='/api/plan'){
            if(outcome==='error')return route.fulfill({status:503,json:{error:'SIMULATED PLANNING FAILURE'}});
            const planned={...version('this-edit'),text:body.text,base_version_id:body.base_version_id,
              plan:outcome==='clarify'?{status:'clarify',explanation:'TEST: clarify direction',action_segments:[]}:ready};
            if(outcome==='mismatch')planned.base_version_id='recent-video';
            fixture.versions.push(planned,version('newer-unrelated'));
            return route.fulfill({json:{...fixture,...(outcome==='missing-id'?{}:{planned_version_id:'this-edit'})}});
          }
          if(pathname==='/api/generate'){
            const v=fixture.versions.find(item=>item.version_id===body.version_id);
            v.job_id='job-this-edit';v.job_status='queued';
            return route.fulfill({json:fixture});
          }
          return route.fulfill({status:500,json:{error:'Unexpected fixture mutation'}});
        }
        const assets={'/':['text/html',source.HTML],'/style.css':['text/css',source.CSS],'/app.js':['text/javascript',source.JS]};
        if(assets[pathname])return route.fulfill({contentType:assets[pathname][0],body:assets[pathname][1]});
        if(pathname==='/api/config')return route.fulfill({json:{csrf_token:'fixture-token',generation_enabled:true,
          planner_kind:'local_model',observer_kind:'human_only',visual_revision_enabled:false,
          scenes:[{scene_id:'mountain-fixture',prompt:'FIXED MOUNTAIN FIXTURE',initial_url:'/initial.svg'}]}});
        if(pathname==='/api/sessions')return route.fulfill({json:{sessions:[fixture]}});
        if(pathname==='/api/jobs')return route.fulfill({json:{jobs:fixture.versions.filter(v=>v.job_id)
          .map(v=>({job_id:v.job_id,status:v.job_status}))}});
        if(pathname==='/initial.svg')return route.fulfill({contentType:'image/svg+xml',
          body:'<svg xmlns="http://www.w3.org/2000/svg" width="832" height="480"/>'});
        return route.fulfill({status:404,body:'CPU fixture supplies no generated video'});
      });
      await page.goto('http://127.0.0.1:9862/');
      await page.waitForFunction(()=>document.querySelector('#edit-base-version').value==='recent-video');
      assert.equal(await page.locator('#generate').isEnabled(),true);
      assert.match(await page.locator('#editing-banner').innerText(),/重新生成整段；不是末帧续写/);
      await page.locator('#compare-left').selectOption('original');
      await page.locator('[data-version-id="original"] [data-action="edit-video-base"]').click();
      assert.equal(await page.locator('#edit-base-version').inputValue(),'original');
      assert.equal(await page.locator('#generate').isDisabled(),true);
      for(const id of ['example-camera','example-timing']){
        await page.locator('#'+id).click();
        assert.equal(await page.locator('#instruction').inputValue(),await page.locator('#'+id).innerText());
        assert.equal(await page.locator('#edit-base-version').inputValue(),'original');
      }
      await page.locator('#instruction').fill('只改镜头，先向左看再向右看');
      await page.locator('#compare-left').selectOption('recent-video');
      await page.locator('#compare-right').selectOption('old-preview');
      await page.evaluate(()=>refresh());
      assert.equal(await page.locator('#edit-base-version').inputValue(),'original');
      assert.equal(await page.locator('#generate').isDisabled(),true);
      assert.deepEqual(posts,[]);
      const fixedImage=await page.locator('#initial').getAttribute('src');
      const fixedPrompt=await page.locator('#prompt').inputValue();
      await page.locator('#plan-generate').click();
      await page.waitForFunction(()=>!busy && /已编排并提交生成|已停止|编排／生成未完成/.test(document.querySelector('#notice').textContent));
      assert.equal(posts[0].pathname,'/api/plan');
      assert.equal(posts[0].body.base_version_id,'original');
      assert.equal(posts[0].body.seed,42);
      assert.equal(posts[0].body.scene_id,'mountain-fixture');
      assert.equal(posts[0].body.text,'只改镜头，先向左看再向右看');
      assert.equal(typeof posts[0].body.request_id,'string');
      assert.equal(await page.locator('#edit-base-version').inputValue(),'original');
      assert.equal(await page.locator('#initial').getAttribute('src'),fixedImage);
      assert.equal(await page.locator('#prompt').inputValue(),fixedPrompt);
      if(outcome==='ready'){
        assert.deepEqual(posts.map(item=>item.pathname),['/api/plan','/api/generate']);
        assert.deepEqual(posts[1].body,{session_id:'creative-test',version_id:'this-edit'});
        assert.equal(await page.locator('#compare-left').inputValue(),'original');
        assert.equal(await page.locator('#compare-right').inputValue(),'this-edit');
        assert.match(await page.locator('#plan-details').innerText(),/this-edi/);
        fixture.versions.find(v=>v.version_id==='this-edit').job_status='completed';
        await page.evaluate(()=>refresh());
        await page.reload();
        await page.waitForFunction(()=>document.querySelector('#compare-right').value==='this-edit');
        assert.equal(await page.locator('#compare-left').inputValue(),'original');
        assert.equal(await page.locator('#edit-base-version').inputValue(),'original');
        assert.equal(posts.length,2);
      }else{
        assert.deepEqual(posts.map(item=>item.pathname),['/api/plan']);
        assert.equal(await page.locator('#generate').isDisabled(),true);
        if(outcome==='clarify')assert.match(await page.locator('#plan-status').innerText(),/需要澄清/);
        if(outcome==='missing-id')assert.match(await page.locator('#notice').innerText(),/未返回明确的计划版本/);
        if(outcome==='mismatch')assert.match(await page.locator('#notice').innerText(),/编辑基线不一致/);
        if(outcome==='error')assert.match(await page.locator('#notice').innerText(),/SIMULATED PLANNING FAILURE/);
      }
      assert.deepEqual(errors,[]);assert.deepEqual(leaks,[]);
      checks.push({outcome,posts:posts.map(item=>item.pathname),passed:true});
      await page.close();
    }
    console.log(JSON.stringify({test_only:true,real_browser:true,passed:true,checks,
      coverage:['completed-video default and explicit base','example buttons never POST','edits lock stale preview',
        'one-click generates exact returned ID despite newer version','clarify/missing ID/mismatch/error stop',
        'fixed scene/seed/image/prompt','base/new version comparison survives polling and reload'],
      evidence:'All requests intercepted; no real model, generated video, or GPU invocation.'}));
  }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});

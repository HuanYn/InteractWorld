"""Presentation/API wiring only: a tiny DOM substitute never runs a model."""
import json
import shutil
import subprocess

import pytest

from training.demo import page


def test_showcase_fixed_input_strip_and_honest_async_labels():
    assert 'readonly' in page.HTML and '异步提交与回看，不是实时游戏' in page.HTML
    assert 'grid-template-columns:152px minmax(0,1fr)' in page.CSS
    assert '左侧 WASD' in page.HTML and '右侧方向键' in page.HTML
    assert '不能套用到当前模式' in page.JS and '仅供参考' in page.JS
    assert 'innerHTML' not in page.JS and 'insertAdjacentHTML' not in page.JS
    assert '未提供完成百分比' in page.JS


def test_showcase_browser_mock_preserves_request_contract_and_video_playback():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js needed for the dependency-free browser DOM mock')
    script = r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const page=JSON.parse(fs.readFileSync(0,'utf8'));
const nodes=new Map();
class Node {
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.style={};this.scrollHeight=180;this._value='';this._text=''}
  set id(value){this._id=value;nodes.set(value,this)} get id(){return this._id}
  set value(value){this._value=String(value)} get value(){return this._value}
  set textContent(value){this._text=String(value);this.children=[]}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('')}
  setAttribute(name,value){this.attrs[name]=String(value)}
  append(...children){this.children.push(...children);if(this.tag==='select'&&!this.value)this.value=children[0].value}
  prepend(child){this.children.unshift(child)}
  replaceChildren(...children){this.children=children;this._text=''}
}
for(const match of page.HTML.matchAll(/<([a-z]+)[^>]* id="([^"]+)"/g)){const n=new Node(match[1]);n.id=match[2]}
function walk(node,tag){return [...(node.tag===tag?[node]:[]),...node.children.flatMap(n=>walk(n,tag))]}
const scene={scene_id:'courtyard',source_episode_id:'held-out',prompt:'<script>not markup</script>',seed:42,
 initial_url:'/api/scenes/courtyard/initial.png',initial_sha256:'a'.repeat(64),
 method:'action_teacher_chunked_ar15s_ui_v1',action_segments:[{frames:240,keys:['W']}]};
const config={scenes:[scene],csrf_token:'test-csrf',generation_enabled:true};
const old={job_id:'1'.repeat(32),scene_id:'courtyard',seed:42,status:'completed',created_unix:10,started_unix:20,finished_unix:200};
const running={job_id:'2'.repeat(32),scene_id:'courtyard',seed:44,status:'running',created_unix:300,started_unix:310};
const failed={job_id:'3'.repeat(32),scene_id:'courtyard',seed:45,status:'failed',created_unix:400,finished_unix:420,error:'GPU budget reservation denied'};
let rows=[failed,running,old],posts=[],deny=false,intervals=[],events={};
const context=vm.createContext({document:{getElementById:id=>nodes.get(id)||null,createElement:tag=>new Node(tag)},
 window:{addEventListener:(name,callback)=>events[name]=callback},
 structuredClone,Date,console,setInterval:(fn,ms)=>intervals.push([fn,ms]),
 fetch:async(url,options)=>{
   if(options?.method==='POST'){posts.push(options);return {ok:!deny,json:async()=>deny?{error:'队列已满'}:{status:'queued'}}}
   return {ok:true,json:async()=>url==='/api/config'?structuredClone(config):{jobs:structuredClone(rows)}};
 }});
(async()=>{
 await vm.runInContext(page.JS,context);
 assert.equal(nodes.get('prompt').value,scene.prompt);assert.equal(nodes.get('initial').src,scene.initial_url);
 assert.equal(nodes.get('prompt').style.height,'180px');nodes.get('prompt').scrollHeight=240;events.resize();
 assert.equal(nodes.get('prompt').style.height,'240px');assert.equal(nodes.get('prompt').value,scene.prompt);
 assert.equal(nodes.get('submit').disabled,false);assert.match(nodes.get('timing').textContent,/不能套用/);
 assert.equal(posts.length,0);assert.equal(intervals[0][1],2000);
 const cards=nodes.get('jobs').children;assert.deepEqual(cards.map(n=>n.id),[failed,running,old].map(j=>'job-'+j.job_id));
 const card=nodes.get('job-'+old.job_id),video=walk(card,'video')[0];assert.equal(video.src,`/api/jobs/${old.job_id}/files/inputs.mp4`);
 assert.equal(walk(card,'a').length,3);assert.equal(walk(nodes.get('job-'+failed.job_id),'video').length,0);
 assert.match(nodes.get('job-'+failed.job_id).textContent,/GPU budget reservation denied/);
 const progress=walk(nodes.get('job-'+running.job_id),'progress')[0];assert.equal(progress.value,'');assert.equal(progress.attrs.value,undefined);
 await context.jobs();assert.equal(walk(card,'video')[0],video);
 const segment=nodes.get('timeline').children[0],groups=segment.children[1].children;
 assert.match(groups[0].textContent,/WASD/);assert.match(groups[1].textContent,/方向键/);
 const labels=walk(segment,'label'),w=labels.find(n=>n.dataset.key==='W').children[0],s=labels.find(n=>n.dataset.key==='S').children[0];
 s.checked=true;s.onchange();assert.equal(w.checked,false);assert.equal(nodes.get('prompt').value,scene.prompt);
 assert.equal(vm.runInContext('segments[0].keys.join()',context),'S');
 const count=walk(segment,'input').find(n=>n.type==='number');count.value=239.5;count.oninput();assert.equal(nodes.get('submit').disabled,true);
 count.value=240;count.oninput();nodes.get('seed').value='';nodes.get('seed').oninput();assert.equal(nodes.get('submit').disabled,true);
 nodes.get('seed').value=88;nodes.get('seed').oninput();assert.equal(nodes.get('submit').disabled,false);
 await nodes.get('submit').onclick();assert.equal(posts.length,1);
 const body=JSON.parse(posts[0].body);assert.deepEqual(Object.keys(body).sort(),['action_segments','scene_id','seed']);
 assert.deepEqual(body,{scene_id:'courtyard',seed:88,action_segments:[{frames:240,keys:['S']}]});
 assert.equal(posts[0].headers['X-InterActWorld-CSRF'],'test-csrf');
 deny=true;await nodes.get('submit').onclick();assert.match(nodes.get('notice').textContent,/提交失败：队列已满/);
 vm.runInContext('state.generation_enabled=false;duration()',context);await nodes.get('submit').onclick();assert.equal(posts.length,2);assert.equal(nodes.get('submit').disabled,true);
 vm.runInContext("state.scenes[0].method='action_teacher_joint61_15s_ui_v1';choose()",context);assert.match(nodes.get('timing').textContent,/仅供参考/);
 console.log('browser mock passed: immutable inputs, left/right controls, strict POST, honest progress, preserved video');
})().catch(error=>{console.error(error);process.exitCode=1});
'''
    result = subprocess.run([node, '-e', script], input=json.dumps(dict(HTML=page.HTML, JS=page.JS)),
                            text=True, encoding='utf-8', capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'browser mock passed' in result.stdout

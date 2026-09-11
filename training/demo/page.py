"""Dependency-free loopback showcase. User text uses textContent/value only."""
HTML = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>InterActWorld · 交互视频生成</title><link rel="stylesheet" href="/style.css"></head><body><main>
<header><div class="eyebrow">INTERACTWORLD / SELF-TRAINED MODEL DEMO</div><h1>给出动作，生成接下来的 <em>15 秒</em></h1>
<p>从固定初始画面出发，使用自训模型生成新视频。异步提交与回看，不是实时游戏。</p>
<div class="badges"><span>832 × 480 · 16 fps</span><span>240 步真实动作输入</span><span>画质与动作可靠性仍在验证</span></div></header>
<section><h2>01 / 固定模型输入</h2><div class="input-strip"><img id="initial" alt="模型实际使用的初始画面">
<label>场景提示词 · 只读<textarea id="prompt" readonly rows="3" aria-label="实际静态提示词"></textarea></label></div>
<details><summary>输入来源与当前生成方式</summary><p id="source"></p><p>静态提示词不会被动作编辑改变。动作作为独立数值张量输入，不拼进提示词，不读取未来真实视频。</p></details></section>
<section><h2>02 / 编排动作</h2><div class="setup-row"><label>预置场景<select id="scene"></select></label>
<label>随机种子<input id="seed" type="number" min="0" max="4294967295" step="1"></label><button id="preset">恢复场景预设</button></div>
<p class="muted">左侧 WASD 控制移动，右侧方向键控制视角。每段可组合按键，不勾选表示保持；16 帧 = 1 秒，总计 240 帧。</p>
<div id="timeline"></div><button id="add">＋ 添加动作段</button><div class="submit-row"><div><strong id="duration"></strong><p id="timing" class="muted"></p></div>
<button class="primary" id="submit" disabled>生成 15 秒视频 →</button></div><p id="notice" role="status" aria-live="polite"></p></section>
<section><div class="section-heading"><h2>03 / 生成与回看</h2><span id="queue-summary" class="muted">正在读取任务</span></div>
<p id="empty-jobs" class="empty">还没有生成任务。提交后会显示本次真实生成的视频，不会用示例视频替代。</p><div id="jobs"></div>
<p class="muted small">运行状态包括输入检查、模型生成与视频封装。暂不提供逐采样步进度或虚构的完成百分比。</p></section>
<footer>保持原始输出：不循环、不慢放、不插帧补时长。自定义动作没有配对真实未来，不提供伪造的 GT 误差。</footer>
</main><script src="/app.js"></script></body></html>'''

CSS = '''*{box-sizing:border-box}body{margin:0;background:#090f18;color:#e8eef6;font:15px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:38px 24px}.eyebrow{font-size:12px;font-weight:750;letter-spacing:2px;color:#96b8bd}h1{font-size:clamp(27px,4vw,42px);line-height:1.25;margin:20px 0 14px}h1 em{font-style:normal;color:#a7ebda}header p,.muted,footer{color:#92a4b7}.badges{display:flex;gap:9px;flex-wrap:wrap}.badges span{font-size:12px;border:1px solid #2a3a48;border-radius:20px;padding:4px 11px;color:#acbac9}section{margin-top:24px;border:1px solid #273443;border-radius:15px;padding:23px;background:#111b28}h2{font-size:17px;margin:0 0 18px}.section-heading{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.small,.section-heading .muted{font-size:12px}.input-strip{display:grid;grid-template-columns:152px minmax(0,1fr);gap:18px;align-items:start}#initial{display:block;width:152px;aspect-ratio:832/480;object-fit:contain;border-radius:8px;background:#080e17}.input-strip label{min-width:0;font-size:12px;color:#91a4b8}#prompt{display:block;width:100%;margin-top:6px;border:0;background:transparent;padding:0;color:#e8eef6;resize:vertical;min-height:72px;line-height:1.55}details{margin-top:12px;font-size:12px;color:#8296aa}summary{cursor:pointer}details p{overflow-wrap:anywhere}label{display:block}input,select,textarea,button{font:inherit;border-radius:7px;border:1px solid #394d62;background:#182638;color:#e8eef6;padding:8px 11px}input:focus,select:focus,textarea:focus,button:focus-visible{outline:2px solid #82d7c5;outline-offset:2px}button{cursor:pointer}button:hover:not(:disabled){border-color:#82d7c5}button:disabled{opacity:.45;cursor:not-allowed}.primary{background:#a7ebda;color:#0c2828;border-color:#a7ebda;font-weight:750;padding:12px 20px;white-space:nowrap}.setup-row{display:flex;align-items:end;gap:16px;flex-wrap:wrap}.setup-row label{font-size:12px;color:#9fb0c1}.setup-row input,.setup-row select{display:block;margin-top:5px;min-width:200px}#timeline{display:grid;gap:12px;max-height:560px;overflow:auto;margin:16px 0}.segment{padding:16px;border:1px solid #2c3c4e;border-radius:10px;background:#0e1723}.segment-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.segment-head strong{font-size:13px}.segment input[type=number]{width:82px;padding:5px 8px}.remove{margin-left:auto;font-size:12px;padding:4px 9px;color:#a4b3c3;background:transparent}.segment-time{color:#9cb0c3;font-size:12px}.keys-layout{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:12px}.key-group{display:flex;align-items:center;justify-content:space-between;gap:12px}.key-group-title{font-size:12px;color:#91a4b8}.keyboard{display:grid;grid-template-columns:repeat(3,42px);grid-template-rows:repeat(2,36px);gap:5px}.key{position:relative}.key input{position:absolute;opacity:0;width:100%;height:100%;margin:0;cursor:pointer}.key span{display:flex;align-items:center;justify-content:center;width:100%;height:100%;border:1px solid #41526a;background:#182638;border-radius:6px;pointer-events:none;font-size:13px}.key input:checked+span{background:#214f49;border-color:#85d9c6;color:#c4fff0}.key input:focus-visible+span{outline:2px solid #a7ebda}.key[data-key=W],.key[data-key=I]{grid-column:2;grid-row:1}.key[data-key=A],.key[data-key=J]{grid-column:1;grid-row:2}.key[data-key=S],.key[data-key=K]{grid-column:2;grid-row:2}.key[data-key=D],.key[data-key=L]{grid-column:3;grid-row:2}.submit-row{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-top:22px;border-top:1px solid #273443;padding-top:20px}.submit-row p{font-size:12px;max-width:690px;margin:4px 0}#notice{white-space:pre-wrap;color:#eacb93;font-size:13px;margin-bottom:0}.empty{text-align:center;padding:35px 16px;border:1px dashed #344254;border-radius:10px;color:#91a4b8}.job{border:1px solid #304154;border-radius:10px;padding:18px;margin-top:15px}.job-head{display:flex;justify-content:space-between;gap:15px;align-items:center}.job-title{font-weight:650;font-size:14px}.status{border-radius:20px;padding:3px 10px;font-size:12px;background:#24384d;color:#bdd3ed;white-space:nowrap}.status.completed{background:#203e38;color:#a7ebda}.status.failed{background:#452d32;color:#f3b1af}.job-meta,.job-time,.job-note{font-size:12px;color:#91a4b8}.job progress{display:block;width:100%;height:7px;margin:17px 0;accent-color:#a7ebda}.job-error{white-space:pre-wrap;overflow-wrap:anywhere;color:#f3b1af;background:#281e28;padding:12px;border-radius:7px;font-size:13px}.job video{display:block;width:100%;border-radius:8px;background:#05090f;margin:15px 0}.downloads{display:flex;gap:11px;flex-wrap:wrap}.downloads a{font-size:12px;text-decoration:none;color:#b9d9f4;border:1px solid #3b5069;border-radius:6px;padding:6px 10px}footer{margin:28px 0;font-size:12px}@media(max-width:740px){main{padding:22px 12px}section{padding:16px}.input-strip{grid-template-columns:96px minmax(0,1fr);gap:12px}#initial{width:96px}.keys-layout{gap:12px}.key-group{flex-direction:column;align-items:flex-start;gap:5px}.keyboard{grid-template-columns:repeat(3,36px);grid-template-rows:repeat(2,32px)}.submit-row{flex-direction:column;align-items:stretch}.primary{width:100%}.setup-row{gap:12px}.setup-row input,.setup-row select{min-width:0;width:170px}.job{padding:12px}.job-head{align-items:flex-start}}'''

JS = ''''use strict';
const $ = id => document.getElementById(id);
const labels = {W:'W',A:'A',S:'S',D:'D',I:'↑',J:'←',K:'↓',L:'→'};
const opposite = {W:'S',S:'W',A:'D',D:'A',I:'K',K:'I',J:'L',L:'J'};
const statuses = {queued:'等待中',running:'正在处理',completed:'生成完成',failed:'生成失败'};
let state,segments=[],token='',submitting=false;
function current(){return state.scenes.find(s=>s.scene_id===$('scene').value)}
function el(tag,cls,text){const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node}
function timelineError(){
  if(!segments.length||segments.length>240)return '至少保留 1 段动作，最多 240 段。';
  if(segments.some(s=>!Number.isInteger(s.frames)||s.frames<1||s.frames>240))return '每段帧数必须是 1–240 的整数。';
  if(segments.some(s=>s.keys.some(k=>s.keys.includes(opposite[k]))))return '同一动作段不能同时按下相反方向。';
  if(segments.reduce((n,s)=>n+s.frames,0)!==240)return '请将动作时间线调整为 240 帧（15 秒）。';
  const seed=Number($('seed').value);
  if($('seed').value===''||!Number.isInteger(seed)||seed<0||seed>4294967295)return '随机种子必须是 0–4294967295 的整数。';
  return '';
}
function duration(){const n=segments.reduce((n,s)=>n+s.frames,0),error=timelineError();
  $('duration').textContent=`${n} / 240 帧 · ${(n/16).toFixed(2)} 秒${error?' · '+error:' · 可以提交'}`;
  $('duration').title=error;$('submit').disabled=!!error||!state.generation_enabled||submitting;
  $('submit').textContent=submitting?'正在提交…':'生成 15 秒视频 →';$('add').disabled=segments.length>=240;
}
function draw(){const box=$('timeline');box.replaceChildren();let start=0;
  segments.forEach((s,i)=>{const row=el('div','segment'),head=el('div','segment-head'),time=el('span','segment-time');
    time.id='segment-time-'+i;time.textContent=`${(start/16).toFixed(2)}–${((start+s.frames)/16).toFixed(2)} 秒`;start+=s.frames;
    const num=el('input');num.type='number';num.min=1;num.max=240;num.step=1;num.value=s.frames;num.setAttribute('aria-label',`第 ${i+1} 段帧数`);
    num.oninput=()=>{s.frames=Number(num.value);let offset=0;segments.forEach((v,j)=>{$('segment-time-'+j).textContent=`${(offset/16).toFixed(2)}–${((offset+v.frames)/16).toFixed(2)} 秒`;offset+=v.frames});duration()};
    const remove=el('button','remove','删除');remove.onclick=()=>{segments.splice(i,1);draw()};head.append(el('strong','',`第 ${i+1} 段`),num,el('span','muted','帧'),time,remove);
    const layout=el('div','keys-layout'),checks={};
    for(const [title,keys] of [['移动 · WASD',['W','A','S','D']],['视角 · 方向键',['I','J','K','L']]]){
      const group=el('div','key-group'),keyboard=el('div','keyboard');
      for(const k of keys){const label=el('label','key'),check=el('input');label.dataset.key=k;check.type='checkbox';check.checked=s.keys.includes(k);
        check.setAttribute('aria-label',`第 ${i+1} 段 ${labels[k]} (${k})`);
        check.onchange=()=>{s.keys=s.keys.filter(v=>v!==k&&(!check.checked||v!==opposite[k]));if(check.checked){s.keys.push(k);checks[opposite[k]].checked=false}duration()};
        checks[k]=check;label.append(check,el('span','',labels[k]));keyboard.append(label)}
      group.append(el('span','key-group-title',title),keyboard);layout.append(group)}
    row.append(head,layout);box.append(row)});duration();
}
function sizePrompt(){const prompt=$('prompt');prompt.style.height='auto';prompt.style.height=prompt.scrollHeight+'px'}
window.addEventListener('resize',sizePrompt);
function choose(){const s=current();$('initial').src=s.initial_url;$('prompt').value=s.prompt;sizePrompt();$('seed').value=s.seed;
  $('source').textContent=`场景 ${s.scene_id} · 来源 ${s.source_episode_id} · 初帧 SHA ${s.initial_sha256.slice(0,12)} · ${s.method}`;
  $('timing').textContent=s.method==='action_teacher_joint61_15s_ui_v1'
    ?'本项目整段联合候选曾在单张 RTX 5090 实测约 2–3 分钟/条，仅供参考，不代表任意模型或场景；排队与加载另计，不是实时。'
    :'当前为分段生成，耗时以本次任务为准。整段联合候选曾实测约 2–3 分钟/条，不能套用到当前模式。';
  segments=structuredClone(s.action_segments);draw();
}
async function api(url,options){const r=await fetch(url,options),body=await r.json();if(!r.ok)throw Error(body.error||r.statusText);return body}
function elapsed(seconds){const n=Math.max(0,Math.floor(seconds));return n<60?`${n} 秒`:`${Math.floor(n/60)} 分 ${n%60} 秒`}
function updateTime(j){const now=Date.now()/1000,line=$('job-time-'+j.job_id);
  if(j.status==='queued')line.textContent=`已等待 ${elapsed(now-j.created_unix)} · 队列串行处理，请勿重复提交。`;
  else if(j.status==='running')line.textContent=`已处理 ${elapsed(now-(j.started_unix||j.created_unix))} · 含输入检查、模型生成和视频封装。`;
  else line.textContent=`本次处理耗时 ${elapsed((j.finished_unix||now)-(j.started_unix||j.created_unix))}`;
}
function renderJob(j,box){let card=$('job-'+j.job_id);if(!card){card=el('article','job');card.id='job-'+j.job_id;box.prepend(card)}
  if(card.dataset.status!==j.status){card.dataset.status=j.status;card.replaceChildren();const head=el('div','job-head');
    head.append(el('span','job-title',`场景 ${j.scene_id} · seed ${j.seed}`),el('span','status '+j.status,statuses[j.status]||j.status));
    const timer=el('p','job-time');timer.id='job-time-'+j.job_id;card.append(head,el('p','job-meta',`任务 ${j.job_id.slice(0,8)} · 独立生成请求`),timer);
    if(j.status==='running'){const progress=el('progress');progress.setAttribute('aria-label','正在处理，未提供完成百分比');card.append(progress)}
    if(j.error)card.append(el('p','job-error',`失败原因：${j.error}`),el('p','job-note','没有复用旧视频或生成替代结果。修复后可重新提交一个新任务。'));
    if(j.status==='completed'){const video=el('video');video.controls=true;video.preload='metadata';video.playsInline=true;video.src=`/api/jobs/${j.job_id}/files/inputs.mp4`;
      const links=el('div','downloads');for(const [name,label] of [['raw.mp4','↓ 原始视频'],['inputs.mp4','↓ 带输入栏视频'],['actions.npy','↓ 实际动作张量']]){
        const link=el('a','',label);link.href=`/api/jobs/${j.job_id}/files/${name}?download=1`;link.download=name;links.append(link)}card.append(video,links)}}updateTime(j);
}
async function jobs(){try{const data=await api('/api/jobs'),box=$('jobs');$('empty-jobs').hidden=data.jobs.length>0;
  const count=status=>data.jobs.filter(j=>j.status===status).length;$('queue-summary').textContent=`${count('running')} 个处理中 · ${count('queued')} 个等待 · ${count('completed')} 个已完成`;
  // Newest-first API: prepend only newly seen cards in reverse order. Unchanged
  // completed cards keep their video element so polling never restarts playback.
  for(const j of [...data.jobs].reverse())renderJob(j,box);
}catch(e){$('notice').textContent=`任务状态读取失败：${e.message}。页面会继续重试，不会自动重放任务。`}}
$('scene').onchange=choose;$('preset').onclick=choose;$('seed').oninput=()=>{if(state)duration()};
$('add').onclick=()=>{segments.push({frames:16,keys:[]});draw()};
$('submit').onclick=async()=>{const error=timelineError();if(error||submitting||!state.generation_enabled){if(error)$('notice').textContent=error;return}
  submitting=true;duration();try{await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json','X-InterActWorld-CSRF':token},
    body:JSON.stringify({scene_id:current().scene_id,seed:Number($('seed').value),action_segments:segments})});
    $('notice').textContent='已提交新任务：将实际运行自训模型。请查看下方状态，避免重复提交。';await jobs();
  }catch(e){$('notice').textContent=`提交失败：${e.message}`}finally{submitting=false;duration()}};
(async()=>{try{state=await api('/api/config');token=state.csrf_token;for(const s of state.scenes){const o=el('option','',s.scene_id);o.value=s.scene_id;$('scene').append(o)}
  if(!state.generation_enabled)$('notice').textContent='预览模式：操作员尚未配置 GPU 授权与预算门禁，当前不能提交生成。';choose();await jobs();setInterval(jobs,2000);
}catch(e){$('notice').textContent=`页面初始化失败：${e.message}`}})();
'''

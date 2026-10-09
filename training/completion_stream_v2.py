"""Bounded localhost live action/JPEG-SSE tool with browser presentation ACK.

No MP4 input, replay branch or prerecorded output interface exists. CPU fixtures
exercise contracts only; they never populate project GPU/browser measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
import base64
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import queue
import secrets
import threading
import time
from urllib.parse import urlsplit, parse_qs

from training.completion_runtime_v2 import require, finite, PhaseTimer, ACTION_KEYS


@dataclass(frozen=True)
class ActionSnapshot:
    action_id: str
    bits: tuple[int, ...]
    received: float
    browser_clock_id: str | None
    browser_action_ms: float | None
    origin: str


class ActionMailbox:
    """Newest state is sampled once at chunk start; in-flight chunks are immutable."""
    def __init__(self, *, clock=time.perf_counter):
        self.clock, self.lock = clock, threading.Lock()
        self.latest = ActionSnapshot('initial-neutral', (0,)*8, clock(), None, None, 'initial_neutral')
        self.accepted = {self.latest.action_id:self.latest}

    def put(self, payload):
        identity,bits=payload.get('action_id'),payload.get('bits')
        browser_clock=payload.get('browser_clock_id');browser_ms=payload.get('browser_action_ms')
        require(isinstance(identity,str) and 1<=len(identity)<=128
            and isinstance(bits,list) and len(bits)==8 and all(type(v) is int and v in (0,1) for v in bits)
            and isinstance(browser_clock,str) and 1<=len(browser_clock)<=128
            and finite(browser_ms) and browser_ms>=0
            and payload.get('origin') in ('keyboard','scripted_browser_keyboard'),
            'online keyboard action identity/eightbits/browser monotonic clock required')
        with self.lock:
            require(identity not in self.accepted and len(self.accepted)<10000,'duplicate action or finite input limit')
            require(self.latest.browser_clock_id!=browser_clock or browser_ms>self.latest.browser_action_ms,
                    'stale out-of-order browser action cannot replace newer keyboard state')
            self.latest=ActionSnapshot(identity,tuple(bits),self.clock(),browser_clock,float(browser_ms),payload['origin'])
            self.accepted[identity]=self.latest
            return self.latest

    def snapshot(self):
        with self.lock: return self.latest


class Trace:
    def __init__(self, *, path=None, clock=time.perf_counter):
        self.path=Path(path) if path else None; self.clock=clock; self.events=[]; self.lock=threading.RLock()
        if self.path:
            require(not self.path.exists(),'new runtime trace cannot overwrite prior evidence')
            self.path.parent.mkdir(parents=True,exist_ok=True)

    def add(self, kind, **values):
        row=dict(event=kind,server_monotonic=self.clock(),**values)
        with self.lock:
            self.events.append(row)
            if self.path:
                with self.path.open('a',encoding='utf-8') as stream: stream.write(json.dumps(row,allow_nan=False)+'\n')
        return row


def jpeg_frames(output, *, quality=85):
    """Actual pixels toJPEG; host transfer and encoding are timed separately by caller."""
    import numpy as np
    from PIL import Image
    require(output.device.type=='cpu' and output.ndim==5 and output.shape[0]==1 and output.shape[2]==3,
            'explicit completed host RGB transfer required before encoding')
    pixels=((output[0].float().clamp(-1,1)+1)*127.5).round().to(dtype=__import__('torch').uint8)
    rows=[]
    for frame in pixels:
        buffer=BytesIO();Image.fromarray(np.asarray(frame.permute(1,2,0))).save(buffer,format='JPEG',quality=quality)
        rows.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
    return rows


def percentile(values, fraction):
    if not values:return None
    values=sorted(values);index=(len(values)-1)*fraction;lo=int(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


class StreamSession:
    def __init__(self,engine,decoder,*,data_origin,producer_binding=None,max_seconds=600,
                 max_chunks=8,idle_seconds=180,max_inflight_chunks=1,clock=time.perf_counter,trace_path=None,timer=None):
        require(data_origin in ('cpu_fixture','native_gpu') and type(max_chunks) is int and 1<=max_chunks<=80
            and type(max_seconds) is int and 1<=max_seconds<=3600
            and type(idle_seconds) is int and 1<=idle_seconds<=180,'finite runtime/idle/chunk bounds required')
        require(type(max_inflight_chunks) is int and max_inflight_chunks in (1,2),'presentation-bounded1/2chunk pipeline only')
        if data_origin=='native_gpu':
            require(isinstance(producer_binding,dict) and producer_binding.get('schema')=='completion_runtime_actual_student_v2',
                    'actual self-trained checkpoint binding mandatory for native session')
        self.engine,self.decoder,self.origin=engine,decoder,data_origin
        self.binding=producer_binding; self.max_seconds,self.max_chunks,self.idle_seconds=max_seconds,max_chunks,idle_seconds
        self.clock,self.timer=clock,timer or PhaseTimer(clock=clock)
        self.mailbox=ActionMailbox(clock=clock); self.trace=Trace(path=trace_path,clock=clock)
        self.queue=queue.Queue(maxsize=2);self.stop=threading.Event();self.connected=threading.Event()
        self.chunks={};self.acks={};self.lock=threading.RLock();self.failure=None
        self.token=secrets.token_urlsafe(24);self.started=self.last_activity=clock();self.primed=False
        self.browser_clock_id=None; self.browser_build_sha256=None; self.client_attached=False
        self.max_inflight=max_inflight_chunks;self.input_open=True

    def check(self):
        require(not self.stop.is_set(),'runtime stopped')
        require(self.clock()-self.started<self.max_seconds,'finite runtime watchdog reached')
        require(self.clock()-self.last_activity<self.idle_seconds,'finite runtime idle bound reached')

    def action(self,payload):
        self.check()
        require(self.input_open,'generation complete; no new input can be applied')
        require(payload.get('browser_clock_id')==self.browser_clock_id,'action must belong to active same-clock browser')
        value=self.mailbox.put(payload);self.last_activity=self.clock()
        self.trace.add('action_received',action_id=value.action_id,bits=list(value.bits),
            browser_clock_id=value.browser_clock_id,browser_action_ms=value.browser_action_ms,origin=value.origin)
        return value

    def attach_browser(self,clock_id,build_hash):
        require(not self.client_attached and isinstance(clock_id,str) and 1<=len(clock_id)<=128
            and build_hash==browser_build_hash(),'one matching live browser implementation required')
        self.browser_clock_id,self.browser_build_sha256=clock_id,build_hash
        self.client_attached=True;self.last_activity=self.clock();self.connected.set()
        self.trace.add('browser_attached',browser_clock_id=clock_id,browser_build_sha256=build_hash)

    def prime(self):
        self.check();require(not self.primed,'fresh session may only prime once')
        initial=self.engine.initialize();_,decode=self.decoder.warmup(self.engine.first)
        self.trace.add('initial_warmup',**initial,**decode)
        self.primed=True

    def step(self):
        self.check();require(self.primed and len(self.chunks)<self.max_chunks,'primefirst and respect finite chunk count')
        require(self.inflight_chunks()<self.max_inflight,'browser must present/ACK previous chunk before further pre-generation')
        action=self.mailbox.snapshot();start=self.clock()
        self.trace.add('denoise_started',chunk_id=len(self.chunks),action_id=action.action_id,
            action_received=action.received,online_snapshot_at_chunk_boundary=True)
        latent,dit=self.engine.generate_chunk(action_id=action.action_id,action_bits=action.bits)
        self.trace.add('denoise_and_commit_finished',**dit)
        pixels,vae=self.decoder.decode_chunk(latent)
        decode_end=self.clock();self.trace.add('decode_finished',chunk_id=dit['chunk_id'],action_id=action.action_id,**vae)
        host,transfer_ms=self.timer.call(lambda:pixels.detach().to(device='cpu',non_blocking=False))
        frames,encode_ms=self.timer.call(lambda:jpeg_frames(host))
        require(len(frames)==12,'actual3latent decode must yield12 displayed frames')
        require(dit['new_frame_count']==vae['new_frame_count']==12,'DiT/VAE frame contracts must agree')
        row=dict(dit|vae,transfer_ms=transfer_ms,encode_ms=encode_ms,
            start_monotonic=start,decode_end_monotonic=decode_end,encoded_monotonic=self.clock(),
            browser_clock_id=action.browser_clock_id,browser_action_ms=action.browser_action_ms,
            action_origin=action.origin,bytes_sent_monotonic=None,frame_ids=[f"{dit['chunk_id']}:{i}" for i in range(12)])
        with self.lock:self.chunks[dit['chunk_id']]=row
        if len(self.chunks)==self.max_chunks:self.input_open=False
        payload=dict(chunk_id=dit['chunk_id'],action_id=action.action_id,frame_ids=row['frame_ids'],frames_jpeg=frames,
            browser_clock_id=action.browser_clock_id,browser_action_ms=action.browser_action_ms,fps=16,
            new_generated_pixels=True,data_origin=self.origin)
        self.trace.add('jpeg_encoded',chunk_id=dit['chunk_id'],action_id=action.action_id,transfer_ms=transfer_ms,
            encode_ms=encode_ms,jpeg_bytes=sum(len(base64.b64decode(v)) for v in frames))
        return payload

    def inflight_chunks(self):
        with self.lock:
            return sum(any(identity not in self.acks for identity in row['frame_ids']) for row in self.chunks.values())

    def mark_sent(self,payload):
        with self.lock:
            row=self.chunks[payload['chunk_id']]
            require(row['bytes_sent_monotonic'] is None,'chunk may be sent only once')
            row['bytes_sent_monotonic']=self.clock()
            self.trace.add('bytes_sent',chunk_id=payload['chunk_id'],action_id=payload['action_id'],
                note='SSE flush returned; not browser visibility')

    def acknowledge(self,payload):
        self.check()
        with self.lock:
            chunk=self.chunks.get(payload.get('chunk_id'));frame=payload.get('frame_id');presented=payload.get('browser_presented_ms')
            require(chunk is not None and chunk['bytes_sent_monotonic'] is not None
                and frame in chunk['frame_ids'] and frame not in self.acks
                and payload.get('action_id')==chunk['action_id']
                and payload.get('browser_clock_id')==self.browser_clock_id
                and finite(presented) and presented>=0,'ACK must match one sent generatedframe/action/browser clock')
            action_ms=chunk['browser_action_ms']
            require(action_ms is None or (chunk['browser_clock_id']==self.browser_clock_id and presented>=action_ms),
                'cannot subtract different browser clocks or negative input-to-presented time')
            row=dict(chunk_id=payload['chunk_id'],frame_id=frame,action_id=chunk['action_id'],
                browser_clock_id=self.browser_clock_id,browser_presented_ms=float(presented),
                browser_action_ms=action_ms,input_to_presented_ms=None if action_ms is None else presented-action_ms,
                send_to_ack_upper_ms=(self.clock()-chunk['bytes_sent_monotonic'])*1000,
                ack_origin=self.origin,browser_build_sha256=self.browser_build_sha256)
            self.acks[frame]=row;self.last_activity=self.clock();self.trace.add('browser_presentation_ack',**row)
            return row

    def results(self):
        with self.lock:rows=list(self.chunks.values());acks=list(self.acks.values())
        generated=None
        if rows:
            elapsed=max(r['encoded_monotonic'] for r in rows)-min(r['start_monotonic'] for r in rows)
            generated=12*len(rows)/elapsed if elapsed>0 else None
        presented=sorted(r['browser_presented_ms'] for r in acks)
        browser_fps=(len(presented)-1)*1000/(presented[-1]-presented[0]) if len(presented)>1 and presented[-1]>presented[0] else None
        first={}
        for ack in acks:
            if ack['input_to_presented_ms'] is not None:
                first[ack['action_id']]=min(first.get(ack['action_id'],math.inf),ack['input_to_presented_ms'])
        metrics=dict(generation_fps=generated,browser_presented_fps=browser_fps,
            input_to_presented_p50_ms=percentile(list(first.values()),.5),
            input_to_presented_p95_ms=percentile(list(first.values()),.95),
            dit_mean_ms=sum(r['dit_ms'] for r in rows)/len(rows) if rows else None,
            kv_commit_mean_ms=sum(r['kv_commit_ms'] for r in rows)/len(rows) if rows else None,
            vae_mean_ms=sum(r['vae_ms'] for r in rows)/len(rows) if rows else None,
            transfer_mean_ms=sum(r['transfer_ms'] for r in rows)/len(rows) if rows else None,
            encode_mean_ms=sum(r['encode_ms'] for r in rows)/len(rows) if rows else None,
            send_to_ack_upper_p95_ms=percentile([r['send_to_ack_upper_ms'] for r in acks],.95))
        fixture=dict(metrics) if self.origin=='cpu_fixture' else None
        if fixture is not None:metrics={key:None for key in metrics}
        used={r['action_id'] for r in rows}
        return dict(schema='completion_stream_measurements_v2',data_origin=self.origin,metrics=metrics,
            cpu_fixture_timing_not_project_measurements=fixture,generated_chunks=len(rows),acknowledged_frames=len(acks),
            distinct_acked_keyboard_actions=len(first),accepted_keyboard_actions=len(self.mailbox.accepted)-1,
            superseded_or_unprocessed_action_ids=sorted(set(self.mailbox.accepted)-used-{'initial-neutral'}),
            max_inflight_chunks=self.max_inflight,input_acceptance_open=self.input_open,
            browser_clock_subtracted_only_within_browser=True,server_send_is_not_visibility=True,
            decoder_warmup_excluded=True,sampling_steps=4,external_cfg_scale=1.,
            native_gpu_with_browser_reported_ack=self.origin=='native_gpu' and bool(acks),
            presentation_measurement='browser double-requestAnimationFrame proxy, not optical panel scanout',
            input_origins=sorted({row['action_origin'] for row in rows}),
            realtime_claim_supported=False,failure=self.failure)

    def produce(self):
        try:
            while not self.connected.wait(.25):self.check()
            self.prime()
            for unused in range(self.max_chunks):
                while self.inflight_chunks()>=self.max_inflight:
                    self.check();self.stop.wait(.01)
                item=self.step()
                while True:
                    self.check()
                    try:self.queue.put(item,timeout=.25);break
                    except queue.Full:pass
            self.trace.add('generation_finished',generated_chunks=len(self.chunks))
        except BaseException as error:
            self.failure=f'{type(error).__name__}: {error}';self.trace.add('failure',error=self.failure);self.stop.set()
        finally:
            self.engine.close();self.decoder.close()


BROWSER = r'''<!doctype html><meta charset="utf-8"><title>Live causal student (experimental)</title>
<style>body{background:#111;color:#ddd;font:16px system-ui}canvas{max-width:96vw}pre{white-space:pre-wrap}</style>
<h2>Live causal 4-step student — experimental, not a realtime claim</h2>
<p>W A S D I J K L. New input applies at the next block boundary. No prerecorded video.</p>
<canvas id="view" width="832" height="480"></canvas><pre id="status">Connecting…</pre>
<script>
const token='__TOKEN__', build='__BUILD__', clock=crypto.randomUUID(), keys=['W','A','S','D','I','J','K','L'];
const held=new Set(), pending=[];let drawing=false,nextFrame=0,finished=false;
const scripted=new URLSearchParams(location.search).get('scripted')==='1';
const post=(path,value)=>fetch(path+'?token='+encodeURIComponent(token),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
async function action(e){let k=e.key.toUpperCase();if(finished||!keys.includes(k))return;e.preventDefault();
 if(e.type==='keydown'){if(held.has(k))return;held.add(k);}else held.delete(k);
 const response=await post('/action',{action_id:crypto.randomUUID(),bits:keys.map(k=>held.has(k)?1:0),browser_clock_id:clock,browser_action_ms:performance.now(),origin:scripted?'scripted_browser_keyboard':'keyboard'});
 if(!response.ok)document.querySelector('#status').textContent='Action rejected';}
addEventListener('keydown',action);addEventListener('keyup',action);
async function draw(){if(drawing)return;drawing=true;const canvas=document.querySelector('#view'),ctx=canvas.getContext('2d');
 while(pending.length){const f=pending.shift(),target=Math.max(nextFrame,performance.now());const wait=Math.max(0,target-performance.now());nextFrame=target+1000/16;await new Promise(r=>setTimeout(r,wait));
 const blob=await(await fetch('data:image/jpeg;base64,'+f.jpeg)).blob(), bitmap=await createImageBitmap(blob);
 ctx.drawImage(bitmap,0,0,canvas.width,canvas.height);bitmap.close();
 // Double RAF is a browser-reported presentation proxy, not optical panel scanout.
 await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));const presented=performance.now();
 post('/ack',{chunk_id:f.chunk_id,frame_id:f.frame_id,action_id:f.action_id,browser_clock_id:clock,browser_presented_ms:presented})
 .then(response=>{if(!response.ok)document.querySelector('#status').textContent='ACK rejected';});
 document.querySelector('#status').textContent='Presented '+f.frame_id+' / action '+f.action_id;}
 drawing=false;}
(async()=>{const hello=await post('/hello',{browser_clock_id:clock,browser_build_sha256:build});if(!hello.ok)throw Error('Browser binding rejected');
 if(scripted){await action(new KeyboardEvent('keydown',{key:'W'}));await action(new KeyboardEvent('keydown',{key:'J'}));
   let left=true;setInterval(async()=>{if(finished)return;await action(new KeyboardEvent('keyup',{key:left?'J':'L'}));left=!left;
     await action(new KeyboardEvent('keydown',{key:left?'J':'L'}));},750);}
 const stream=new EventSource('/events?token='+encodeURIComponent(token));stream.addEventListener('chunk',event=>{const c=JSON.parse(event.data);
 c.frames_jpeg.forEach((jpeg,i)=>pending.push({jpeg,chunk_id:c.chunk_id,frame_id:c.frame_ids[i],action_id:c.action_id}));draw();});
 stream.addEventListener('done',()=>{finished=true;stream.close();document.querySelector('#status').textContent+=' / generation complete';});})();
</script>'''


def browser_build_hash():
    import hashlib
    return hashlib.sha256(BROWSER.encode()).hexdigest()


class LocalStreamServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,session,*,port=0):
        self.session=session;self.events_claimed=False
        super().__init__(('127.0.0.1',port),StreamHandler)


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self,*unused):pass

    def _authorize(self):
        query=parse_qs(urlsplit(self.path).query)
        require(query.get('token')==[self.server.session.token],'session token required; localhost only')
        origin=self.headers.get('Origin')
        require(origin is None or origin==f'http://127.0.0.1:{self.server.server_port}','cross-origin requests prohibited')

    def _reply(self,status,content,kind='application/json'):
        data=content.encode() if isinstance(content,str) else json.dumps(content,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

    def do_GET(self):
        streaming=False
        try:
            self._authorize();path=urlsplit(self.path).path;s=self.server.session
            if path=='/':return self._reply(200,BROWSER.replace('__TOKEN__',s.token).replace('__BUILD__',browser_build_hash()),'text/html; charset=utf-8')
            if path=='/results':return self._reply(200,s.results())
            require(path=='/events','unknown streaming endpoint')
            with s.lock:
                require(s.client_attached and not self.server.events_claimed,'one live SSE consumer only; no reconnect replay')
                self.server.events_claimed=True
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-store');self.end_headers()
            streaming=True
            sent=0
            while sent<s.max_chunks:
                s.check()
                try:item=s.queue.get(timeout=.25)
                except queue.Empty:continue
                with s.lock:
                    self.wfile.write(('event: chunk\ndata: '+json.dumps(item)+'\n\n').encode());self.wfile.flush();s.mark_sent(item)
                sent+=1
            self.wfile.write(b'event: done\ndata: {}\n\n');self.wfile.flush()
        except BaseException as error:
            if streaming:
                self.server.session.failure=f'{type(error).__name__}: {error}'
                self.server.session.trace.add('stream_connection_failed',error=str(error))
                self.server.session.stop.set()
                return
            if not self.wfile.closed:
                try:self._reply(400,dict(error=str(error)))
                except OSError:pass

    def do_POST(self):
        try:
            self._authorize();s=self.server.session
            size=int(self.headers.get('Content-Length','0'));require(0<size<=16384,'bounded JSON body required')
            data=json.loads(self.rfile.read(size));path=urlsplit(self.path).path
            if path=='/hello':s.attach_browser(data.get('browser_clock_id'),data.get('browser_build_sha256'));answer=dict(attached=True)
            elif path=='/action':answer=dict(accepted=s.action(data).action_id)
            elif path=='/ack':answer=s.acknowledge(data)
            else:raise ValueError('unknown action/ACK endpoint')
            self._reply(200,answer)
        except (ValueError,RuntimeError,TypeError,KeyError) as error:self._reply(400,dict(error=str(error)))


def serve_bounded(session,*,port=0,browser_launcher=None):
    """Blocks only for this finite runtime. Caller retains originalGPU watchdog/lease."""
    server=LocalStreamServer(session,port=port)
    producer=threading.Thread(target=session.produce,name='owned-live-student',daemon=True)
    producer.start();server.timeout=.25
    session.trace.add('localhost_service_started',url=f'http://127.0.0.1:{server.server_port}/?token={session.token}')
    browser=None
    try:
        if browser_launcher is not None:
            browser=browser_launcher(f'http://127.0.0.1:{server.server_port}/?token={session.token}&scripted=1')
        while True:
            session.check();server.handle_request()
            if not session.input_open and session.chunks and session.inflight_chunks()==0:
                session.trace.add('all_generated_frames_presented');break
    except RuntimeError as error:
        session.trace.add('bounded_service_stopped',reason=str(error))
    finally:
        session.stop.set();server.server_close();producer.join(timeout=5)
        if browser is not None:browser.close()
    return session.results()


def run_native_stream(request,directory,lease,*,check):
    """Real worker entry after actual gates; not registered/deployed by thisCPU task.

    The coordinator must supply a complete bounded runtime protocol/native-gate
    record and original guarded reservation. This never prepares/reserves itself.
    """
    from training.completion_runtime_v2 import DATA_ROOT,build_native_runtime,file_sha256,_read
    from training.completion_ode_jobs_v2 import _generation_inputs
    directory=Path(directory)
    require(DATA_ROOT in directory.parents and directory.name==request.get('job_id')
        and request.get('schema')=='completion_runtime_stream_request_v2'
        and _read(directory/'request.json')==request and callable(check),'actual fixed guarded runtime request required')
    require(request.get('case_id') in ('pair-j','pair-l','holdout-j','holdout-l')
        and request.get('seed')==42 and request.get('condition')=='online_keyboard'
        and request.get('sampling_steps')==4 and request.get('external_cfg_scale')==1.,
        'same fixed original input/student/seed4step CFG1 runtime comparison required')
    first,positive,*unused=_generation_inputs(request['case_id'],request['baseline_binding'],'original')
    engine,decoder=build_native_runtime(binding=request['student_binding'],ledger_path=DATA_ROOT/'creator-runtime/budget.json',
        lease=lease,request_sha256=file_sha256(directory/'request.json'),max_seconds=request['max_seconds'],
        first=first,conditions=positive,max_chunks=request['max_chunks'],
        local_window_latents=request.get('local_window_latents'),native_gate=request['native_gate'],check=check)
    import torch
    torch.cuda.reset_peak_memory_stats()
    session=StreamSession(engine,decoder,data_origin='native_gpu',producer_binding=request['student_binding'],
        max_chunks=request['max_chunks'],max_seconds=request['max_seconds'],idle_seconds=180,
        max_inflight_chunks=request.get('max_inflight_chunks',1),
        trace_path=directory/'runtime-events.jsonl',timer=engine.timer)
    try:
        launcher=None
        if request.get('browser_driver')=='headless_firefox_scripted_keyboard':
            from training.completion_runtime_jobs_v4 import HeadlessFirefox
            launcher=lambda url:HeadlessFirefox(url,directory/'browser-profile',directory/'browser.log')
        result=serve_bounded(session,port=request.get('port',0),browser_launcher=launcher)
        result.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(),actual_student=request['student_binding'],
            actual_input_baseline=request['baseline_binding'],native_gate=request['native_gate'])
        with (directory/'runtime-measurements.json').open('x',encoding='utf-8') as output:
            json.dump(result,output,indent=2,allow_nan=False)
        return result
    finally:
        session.stop.set()  # Original finite outer worker owns GPU shutdown/settlement.

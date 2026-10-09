"""Actual runtime tail handlers; no reserve, deployment, or invented measurements.

The fixed ODE80 producer and the native gate are late-bound from settled records.
Firefox is a real headless browser using canvas / double-RAF ACKs and scripted
keys. Its timing is explicitly not a human or optical-display measurement.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from training import completion_runtime_v2 as rt

MODES=('completion_runtime_v2_probe','completion_runtime_v2_stream')
ROOT=rt.DATA_ROOT/'creator-runtime/abot-completion-20261007/runtime-v2'
REVISION='creator-completion-deadline-v4'
require=rt.require


def source_hashes():
    from training import completion_stream_v2 as stream
    return {Path(p).name:rt.file_sha256(p) for p in (__file__,rt.__file__,stream.__file__)}


def build_tail_jobs(ode80_producer, *, gpu_index=0):
    """Four concrete slots may overlap DMD: none waits for DMD32 or its videos.

    Work ceilings are safety bounds, not predicted speeds. The enclosing
    coordinator must add its actual reserve/check/settle fee bound to each slot.
    They spend at most6000 work seconds of the existing21600-second envelope.
    Missing Table2 assets stay explicit gaps rather than fake GPU jobs.
    """
    require(ode80_producer.get('mode')=='completion_ode_v2_train'
        and ode80_producer.get('arguments',{}).get('stop_after_step')==80
        and gpu_index in (0,1,2,3,4),'exact finalODE80 producer/card required')
    producer=ode80_producer['job_id'];jobs=[]
    for label,window in (('full',None),('local7',7)):
        probe=f'd30-runtime-{label}-probe-v4';stream=f'd30-runtime-{label}-stream-v4'
        common=dict(student_producer_from_job_id=producer,case_id='pair-j',local_window_latents=window,
                    sampling_steps=4,external_cfg_scale=1.,seed=42)
        for mode,identity,seconds,args,depends in (
            (MODES[0],probe,1200,dict(common,probe_chunks=4),[producer]),
            (MODES[1],stream,1800,dict(common,native_gate_from_job_id=probe,max_chunks=8,
                condition='online_keyboard',max_inflight_chunks=1,
                browser_driver='headless_firefox_scripted_keyboard'),[producer,probe])):
            jobs.append(dict(job_id=identity,mode=mode,group='runtime',gpu_index=gpu_index,
                source_revision=REVISION,max_seconds=seconds,cleanup_seconds=60,
                directory=str(ROOT/identity),depends_on=depends,arguments=args))
    return jobs


def _producer(manifest,spec,identity,*,mode):
    rows=[row for row in manifest['jobs'] if row['job_id']==identity]
    require(len(rows)==1 and identity in spec['depends_on'] and rows[0]['mode']==mode,
            'exact explicit actual producer dependency required')
    return rows[0]


def _native_gate(producer,binding,window):
    directory=Path(producer['directory'])
    result=rt._read(directory/'result.json')
    gate=copy.deepcopy(result.get('execution',{}).get('native_runtime_gate'))
    require(isinstance(gate,dict),'actual native probe has no gate result')
    gate['actual_profile_records']={name:dict(path=str(directory/(name+'.json')),
        sha256=rt.file_sha256(directory/(name+'.json'))) for name in ('request','result','settlement')}
    rt.validate_native_gate(gate,binding,ledger_path=rt.DATA_ROOT/'creator-runtime/budget.json',
                            local_window_latents=window)
    return gate


def browser_preflight(directory,*,timeout=30):
    """Actually exercise headless canvas/RAF/HTTP on CPU BEFORE reservation.

    This tiny image is explicitly a readiness fixture, not generated pixels or
    project latency. No GPU/CUDA/model is loaded and the browser is own-process.
    """
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    directory=Path(directory);ready=threading.Event();observed={}
    page=b'''<!doctype html><canvas id="c" width="2" height="2"></canvas><script>
const c=document.querySelector('canvas'),x=c.getContext('2d');x.fillStyle='#12ab34';x.fillRect(0,0,2,2);
requestAnimationFrame(()=>requestAnimationFrame(()=>fetch('/ack',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({canvas_pixel:Array.from(x.getImageData(0,0,1,1).data),double_raf:true,clock_ms:performance.now()})})))
</script>'''
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*unused):pass
        def do_GET(self):
            self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(page)))
            self.end_headers();self.wfile.write(page)
        def do_POST(self):
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:self.send_error(400);return
            value=json.loads(self.rfile.read(length))
            if value.get('canvas_pixel')!=[18,171,52,255] or value.get('double_raf') is not True:
                self.send_error(400);return
            observed.update(value);ready.set();self.send_response(204);self.end_headers()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();browser=None;began=time.monotonic()
    try:
        browser=HeadlessFirefox(f'http://127.0.0.1:{server.server_port}/?token=cpu-readiness-fixture&scripted=1',
            directory/'browser-preflight-profile',directory/'browser-preflight.log')
        require(ready.wait(timeout),'actual existing Firefox canvas/doubleRAF/HTTP preflight failed before reserve')
        result=dict(status='completed',scope='CPU_browser_readiness_fixture_not_GPU_or_project_latency',
            browser_binary=shutil.which('firefox'),software_rendering_requested=True,canvas_double_raf_http_verified=True,
            cpu_elapsed_seconds=time.monotonic()-began,cuda_initialized=False)
        with (directory/'browser-preflight.json').open('x',encoding='utf-8') as output:json.dump(result,output,indent=2)
        return result
    finally:
        if browser is not None:browser.close()
        server.shutdown();server.server_close();thread.join(timeout=2)


def prepare_job(spec,directory):
    """Validate all known assets/parents/browser before taking a card."""
    import torch
    from training import completion_ode_jobs_v2 as jobs
    from training.studio import balanced_lora_pair_eval as baseline
    directory=Path(directory)
    require(directory.is_dir() and directory.name==spec.get('job_id') and ROOT in directory.parents
        and not (directory/'request.json').exists() and not torch.cuda.is_initialized()
        and spec.get('mode') in MODES and spec.get('source_revision')==REVISION,
        'fresh immutable CPU-only runtime job required')
    manifest=jobs.parallel_manifest(spec['parallel_budget_binding'])
    own=next((row for row in manifest['jobs'] if row['job_id']==spec['job_id']),None)
    require(own is not None and all(spec.get(k)==v for k,v in own.items()),'runtime specification changed')
    args=spec['arguments'];window=args.get('local_window_latents')
    require(window in (None,7) and args.get('case_id')=='pair-j' and args.get('seed')==42
        and args.get('sampling_steps')==4 and args.get('external_cfg_scale')==1.,
        'fixed same-student/input/seed four-step comparison only')
    producer=_producer(manifest,spec,args['student_producer_from_job_id'],mode='completion_ode_v2_train')
    require(producer['arguments']['stop_after_step']==80,'no intermediate best-parent selection')
    binding=rt.student_binding_from_job(producer['directory'])
    # Schema/optimizer/lineage validation of the real mmap checkpoint is CPU
    # work too. Do not discover a known incompatible student after reserve.
    actual_request=rt._pin(binding['records']['request'])
    endpoint_arguments=copy.deepcopy(actual_request['resolved_arguments'])
    endpoint_arguments.update(endpoint_step=80,endpoint_profile=copy.deepcopy(binding['records']))
    jobs._endpoint(endpoint_arguments,parent=endpoint_arguments['parent'],binding=endpoint_arguments['cache_binding'])
    baseline_binding=baseline.baseline_records(args['case_id'])[2]
    first,positive,*unused=jobs._generation_inputs(args['case_id'],baseline_binding,'original')
    require(first.shape==(1,1,48,30,52) and first.dtype==torch.bfloat16
        and bool(torch.isfinite(first).all()) and bool(torch.isfinite(positive).all()),
        'actual z0/positive contract invalid before reserve')
    resolved=dict(copy.deepcopy(args),student_binding=binding,baseline_binding=baseline_binding)
    if spec['mode']==MODES[0]:
        require(spec['max_seconds']==1200 and args.get('probe_chunks')==4,'exact fixed native probe bound')
        schema='completion_runtime_probe_request_v2'
    else:
        require(spec['max_seconds']==1800 and args.get('max_chunks')==8
            and args.get('browser_driver')=='headless_firefox_scripted_keyboard'
            and args.get('condition')=='online_keyboard' and args.get('max_inflight_chunks')==1,
            'exact finite real-browser stream contract')
        require(shutil.which('firefox') is not None,'existing real Firefox missing before reserve; no fake ACK')
        browser_preflight(directory)
        probe=_producer(manifest,spec,args['native_gate_from_job_id'],mode=MODES[0])
        resolved['native_gate']=_native_gate(probe,binding,window)
        schema='completion_runtime_stream_request_v2'
    request=dict(copy.deepcopy(spec),**resolved,schema=schema,contract_max_seconds=spec['max_seconds'],
        source_hashes=source_hashes(),derived_timeout=dict(schema='creator_measured_timeout_v1',
        kind='fixed',max_seconds=spec['max_seconds']),generation_reads_future_clean=False,
        browser_measurement='headless scripted browser canvas/RAF proxy; not human/optical' if spec['mode']==MODES[1] else None)
    require(not torch.cuda.is_initialized(),'runtime CPU prepare initialized CUDA')
    jobs.cache.write_new(directory/'request.json',request)
    return directory/'request.json'


class HeadlessFirefox:
    """Own finite localhost browser process, software rendering, data-root profile."""
    def __init__(self,url,profile,log):
        require(url.startswith('http://127.0.0.1:') and '&scripted=1' in url,'local fixed browser driver only')
        profile,log=Path(profile),Path(log)
        require(ROOT in profile.parents and not profile.exists() and ROOT in log.parents,
                'fresh data-root browser profile/log required')
        binary=shutil.which('firefox');require(binary is not None,'existing actual Firefox missing')
        profile.mkdir()
        for name in ('tmp','cache','config','data','runtime','crash','mesa'):
            (profile/name).mkdir(mode=0o700)
        self.output=log.open('xb')
        env=dict(os.environ,MOZ_HEADLESS='1',MOZ_DISABLE_GPU_PROCESS='1',LIBGL_ALWAYS_SOFTWARE='1',
            TMPDIR=str(profile/'tmp'),XDG_CACHE_HOME=str(profile/'cache'),XDG_CONFIG_HOME=str(profile/'config'),
            XDG_DATA_HOME=str(profile/'data'),XDG_RUNTIME_DIR=str(profile/'runtime'),
            MESA_SHADER_CACHE_DIR=str(profile/'mesa'),MOZ_CRASHREPORTER_DATA_DIRECTORY=str(profile/'crash'),
            MOZ_CRASHREPORTER_DISABLE='1',MOZ_CRASHREPORTER_NO_REPORT='1')
        try:
            self.process=subprocess.Popen([binary,'--headless','--no-remote','--profile',str(profile),url],
                stdout=self.output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        except BaseException:self.output.close();raise

    def close(self):
        if self.process.poll() is None:
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.process.wait(timeout=5)
        self.output.close()


def run_job(request,directory,lease):
    from training.demo.backend import validate_lease,validate_active_authority
    from training.studio.completion_causal_sufficiency_train import ledger_snapshot
    from training import completion_ode_jobs_v2 as jobs
    import torch
    directory=Path(directory)
    require(rt._read(directory/'request.json')==request and request['source_hashes']==source_hashes()
        and request['mode'] in MODES and lease.get('gpu_index')==request['gpu_index'],
        'prepared exact runtime/source/card changed')
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=request['job_id'],
                   request_sha256=rt.file_sha256(directory/'request.json'))
    ledger,_=ledger_snapshot();accounting=jobs.reconcile_parallel_budget(request['parallel_budget_binding'],
        ledger=ledger,lease=lease,max_seconds=request['max_seconds'])
    require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1'
        and torch.cuda.is_available(),'actual offline CUDA worker required')
    began=time.monotonic()
    def check():
        require(time.monotonic()-began<request['max_seconds']-30,'finite runtime job deadline; no retry')
        validate_active_authority(lease)
    if request['mode']==MODES[0]:execution=rt.run_native_probe(request,directory,lease,check=check)
    else:
        from training.completion_stream_v2 import run_native_stream
        execution=run_native_stream(request,directory,lease,check=check)
        require(execution['failure'] is None and execution['generated_chunks']==8
            and execution['acknowledged_frames']==96 and execution['distinct_acked_keyboard_actions']>0
            and execution['native_gpu_with_browser_reported_ack'] is True,
            'live real-browser run incomplete; no invented latency/FPS')
    return dict(status='completed',execution=execution,accounting_at_start=accounting,
        elapsed_seconds=time.monotonic()-began,quality_verified=False,realtime_verified=False)

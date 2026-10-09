"""Portable loopback Wan service; run under your own exclusive GPU scheduler.

The public branch does not ship the original private lease or launch authority.
Explicit device binding, launch acknowledgement and a whole-process limit are
required. This limit does not acquire a GPU or protect against other users.
"""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8878)
    p.add_argument('--snapshot',required=True,help='Operator-owned asset root with unchanged original lineage')
    p.add_argument('--prompt-cache',required=True)
    p.add_argument('--runtime-root',required=True,help='Dedicated directory for initial images and session outputs')
    p.add_argument('--max-seconds',type=int,required=True,help='Whole process bound, including model loading')
    p.add_argument('--idle-seconds',type=int,default=180)
    p.add_argument('--launch',action='store_true',help='Acknowledge your own exclusive allocation; not an allocator')
    a=p.parse_args()
    if not a.launch or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Use --launch with an explicit CUDA_VISIBLE_DEVICES under your own exclusive GPU allocation')
    if not 1<=a.max_seconds<=14400 or not 1<=a.idle_seconds<=a.max_seconds:
        raise ValueError('Require 1 <= idle_seconds <= max_seconds <= 14400')
    root=Path(a.runtime_root).resolve()
    if not root.is_dir(): raise ValueError('Create a dedicated runtime-root first')
    watchdog=threading.Timer(a.max_seconds,lambda:os._exit(124))
    watchdog.daemon=True; watchdog.start()
    from interactive.lab_video import LaboratoryVideoModel
    model=LaboratoryVideoModel(a.snapshot,a.prompt_cache)
    lock=threading.Lock()
    last_activity=time.time()
    def scoped(value):
        path=Path(value).resolve()
        if not path.is_relative_to(root): raise ValueError('Path outside runtime root')
        return path
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,status=200):
            blob=json.dumps(data,ensure_ascii=False).encode(); self.send_response(status)
            self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(blob)))
            self.end_headers(); self.wfile.write(blob)
        def do_GET(self):
            if self.path=='/health': self.send({'ready':True,'busy':lock.locked(),'visual_mode':'wan_laboratory',
                                              'conditioning_extensions':['current_frame_zoom_1p08'],
                                              'last_activity_unix':last_activity})
            else: self.send({'error':'Not found'},404)
        def do_POST(self):
            nonlocal last_activity
            if self.path!='/generate': self.send({'error':'Not found'},404); return
            if not lock.acquire(blocking=False): self.send({'error':'Renderer busy; do not resubmit until current job ends'},409); return
            last_activity=time.time()
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<16000: raise ValueError('Invalid payload size')
                data=json.loads(self.rfile.read(size)); data['initial']=str(scoped(data['initial'])); data['output_dir']=str(scoped(data['output_dir']))
                result=model.generate(**data); self.send(result)
            except Exception as exc:
                import traceback
                traceback.print_exc(); self.send({'error':type(exc).__name__+': '+str(exc)},500)
            finally:
                last_activity=time.time()
                lock.release()
    print(json.dumps({'ready':True,'port':a.port,'renderer':'wan_laboratory'}),flush=True)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler); server.timeout=1
    try:
        while lock.locked() or time.time()-last_activity<a.idle_seconds:
            server.handle_request()
    finally:
        server.server_close(); watchdog.cancel()

if __name__=='__main__': main()

"""Local Qwen3-VL OpenAI-compatible adapter, text/image only, loopback only.

CPU mode is available without a GPU grant. CUDA mode requires --launch and an
operator-selected CUDA_VISIBLE_DEVICES; run it under the project's shared lease.
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import time
from contextlib import nullcontext


def release_request_cuda_cache(torch_module, device):
    """Release only unused request allocations, never model weights or KV inputs."""
    if device != 'cuda':
        return {'enabled': True, 'device': device, 'cuda_cache_released': False}
    import gc
    cuda = torch_module.cuda
    before = {'allocated_bytes': cuda.memory_allocated(), 'reserved_bytes': cuda.memory_reserved()}
    gc.collect()
    cuda.empty_cache()
    after = {'allocated_bytes': cuda.memory_allocated(), 'reserved_bytes': cuda.memory_reserved()}
    receipt = {'event': 'planner_request_cuda_cache_release', 'device': device,
               'before': before, 'after': after, 'model_weights_offloaded': False,
               'sampling_or_content_changed': False}
    print(json.dumps(receipt), flush=True)
    return receipt


def completion_response(complete, cleanup=None):
    """The inference frame and exception reference end before optional cleanup/HTTP."""
    try:
        response, status = complete(), 200
    except Exception as exc:
        response, status = {'error': str(exc)}, 400
    # Exiting the except suite clears exc and its traceback before collecting.
    if cleanup is not None:
        try:
            response['request_memory_release'] = cleanup()
        except Exception as exc:
            response, status = {'error': 'Request CUDA cache release failed: ' + str(exc),
                                'inference_error': response.get('error')}, 500
    return response, status


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True)
    p.add_argument('--adapter',help='Optional qualified local planner LoRA; base remains unchanged')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--launch',action='store_true')
    p.add_argument('--port',type=int,default=8874); p.add_argument('--threads',type=int,default=8)
    p.add_argument('--ready-file',help='Optional new runtime JSON file for an ephemeral port')
    p.add_argument('--release-request-cache',action='store_true',
                   help='Opt-in release of unused request CUDA allocations before each HTTP response')
    args=p.parse_args()
    # Reserve the listener before loading weights; occupied ports must fail cheaply.
    server=HTTPServer(('127.0.0.1',args.port),BaseHTTPRequestHandler)
    actual_port=server.server_address[1]
    began=time.monotonic()
    def milestone(stage): print(json.dumps({'startup_stage':stage,'seconds':time.monotonic()-began}),flush=True)
    milestone('importing_runtime')
    if args.device=='cuda' and (not args.launch or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('CUDA needs an explicit card binding and --launch under a granted lease')
    if args.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    milestone('runtime_imported')
    torch.set_num_threads(args.threads)
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    milestone('processor_loaded')
    model=AutoModelForImageTextToText.from_pretrained(args.model,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':args.device})
    milestone('weights_loaded_on_target')
    if args.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,args.adapter,local_files_only=True)
    model=model.eval()
    last_activity=time.time()

    class Handler(BaseHTTPRequestHandler):
        def send(self,value,status=200):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path=='/health':
                self.send({'status':'ready','model':str(args.model),'adapter':args.adapter,'device':args.device,
                           'last_activity_unix':last_activity})
            else: self.send({'error':'not found'},404)

        def do_POST(self):
            nonlocal last_activity
            if self.path!='/v1/chat/completions': self.send({'error':'not found'},404); return
            last_activity=time.time()
            def complete():
                size=int(self.headers.get('Content-Length','0'))
                if size<=0 or size>4_000_000: raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size)); messages=payload['messages']
                if not isinstance(messages,list) or len(messages)>12: raise ValueError('Invalid messages')
                # Only inline images are supported: no server-side HTTP/file fetching.
                converted=[]; has_image=False
                for msg in messages:
                    content=msg['content']
                    if isinstance(content,list):
                        items=[]
                        for item in content:
                            if item['type']=='text': items.append(item)
                            elif item['type']=='image_url':
                                has_image=True
                                uri=item['image_url']['url']
                                if not uri.startswith('data:image/'): raise ValueError('Inline images only')
                                items.append({'type':'image','image':uri})
                            else: raise ValueError('Only text/images supported')
                        content=items
                    converted.append({'role':msg['role'],'content':content})
                started=time.monotonic()
                inputs=processor.apply_chat_template(converted,tokenize=True,add_generation_prompt=True,
                    return_dict=True,return_tensors='pt').to(args.device)
                if inputs['input_ids'].shape[1]>8192: raise ValueError('Context limit exceeded')
                # The new LoRA is text-planner-only; image observation uses the frozen base.
                adapter_active=bool(args.adapter and not has_image and payload.get('model')!='base-general')
                adapter_context=model.disable_adapter() if args.adapter and not adapter_active else nullcontext()
                with torch.inference_mode(),adapter_context:
                    tokens=model.generate(**inputs,max_new_tokens=min(int(payload.get('max_tokens',600)),1800),do_sample=False)
                answer=processor.batch_decode(tokens[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
                return {'choices':[{'message':{'role':'assistant','content':answer}}],
                    'usage':{'prompt_tokens':inputs['input_ids'].shape[1],'completion_tokens':tokens.shape[1]-inputs['input_ids'].shape[1]},
                    'latency_seconds':time.monotonic()-started,
                    'adapter_active':adapter_active}
            cleanup=(lambda: release_request_cuda_cache(torch,args.device)) if args.release_request_cache else None
            try:
                response,status=completion_response(complete,cleanup)
                self.send(response,status)
            finally: last_activity=time.time()

    server.RequestHandlerClass=Handler
    ready={'status':'ready','host':'127.0.0.1','port':actual_port,'device':args.device,'adapter':args.adapter}
    if args.ready_file:
        with Path(args.ready_file).open('x') as stream: json.dump(ready,stream)
    print(json.dumps(ready),flush=True)
    server.serve_forever()


if __name__=='__main__': main()

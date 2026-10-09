"""Produce real Base or LoRA planner predictions with the same prompt/decoding."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from interactive.dataset import messages_for


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--adapter')
    p.add_argument('--eval',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=700,help='Use 1800 for the runtime multi-step goal contract')
    p.add_argument('--prompt-contract',help='Saved v1 contract for exact historical replay')
    args=p.parse_args()
    if not 1<=args.max_new_tokens<=1800: raise ValueError('Token cap must be in 1..1800')
    requested_adapter=args.adapter
    if args.adapter:
        adapter=Path(args.adapter)
        # Trainer output contains resumable checkpoints plus the final adapter/.
        if not (adapter/'adapter_config.json').is_file() and (adapter/'adapter/adapter_config.json').is_file():
            adapter=adapter/'adapter'
        if not (adapter/'adapter_config.json').is_file(): raise FileNotFoundError('No saved local adapter at requested output')
        args.adapter=str(adapter)
    contract=json.loads(Path(args.prompt_contract).read_text()) if args.prompt_contract else None
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    architecture=AutoConfig.from_pretrained(args.model,local_files_only=True)
    model_class=AutoModelForImageTextToText if architecture.model_type.startswith('qwen3_vl') else AutoModelForCausalLM
    model=model_class.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16 if args.device=='cuda' else torch.float32,
        attn_implementation='sdpa')
    if args.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,args.adapter,local_files_only=True)
    model.to(args.device).eval()
    rows=[json.loads(l) for l in Path(args.eval).read_text(encoding='utf-8').splitlines() if l.strip()]
    with Path(args.output).open('x',encoding='utf-8') as stream:
        for offset in range(0,len(rows),args.batch_size):
            batch=rows[offset:offset+args.batch_size]; started=time.monotonic()
            prompts=[tokenizer.apply_chat_template(messages_for(row,contract),tokenize=False,add_generation_prompt=True) for row in batch]
            inputs=tokenizer(prompts,padding=True,return_tensors='pt').to(args.device)
            with torch.inference_mode(): outputs=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            for row,output in zip(batch,outputs):
                text=tokenizer.decode(output[inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                stream.write(json.dumps({'id':row['id'],'text':text,'model':args.model,'adapter':args.adapter,
                    'requested_adapter':requested_adapter,'prediction_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},ensure_ascii=False)+'\n')
            stream.flush(); print(json.dumps({'predicted':offset+len(batch),'total':len(rows),'batch_seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__': main()

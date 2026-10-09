"""Transformers + PEFT, assistant-only loss. No model downloads by default."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from interactive.dataset import messages_for,validate_row


def load_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows: validate_row(row)
    if not rows: raise ValueError('Empty training data')
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,help='Local small causal LM or Qwen3-VL (text-only planner training)')
    p.add_argument('--train',required=True); p.add_argument('--eval',required=True); p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=50); p.add_argument('--max-length',type=int,default=4096)
    p.add_argument('--epochs',type=float,help='Explicit epoch training instead of max steps')
    p.add_argument('--preserve-checkpoints',action='store_true',help='Do not rotate/delete this run checkpoints')
    p.add_argument('--profile-gate',action='store_true',help='After two optimizer steps stop if conservative runtime projection exceeds training budget')
    p.add_argument('--rank',type=int,default=8); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--training-seconds',type=int,default=600,help='Soft training-loop budget, excludes load/eval')
    p.add_argument('--prompt-contract',help='Saved v1 contract for exact historical replay')
    p.add_argument('--launch',action='store_true',help='Use only within an explicit GPU grant/lease')
    args=p.parse_args(); train=load_rows(args.train); eval_rows=load_rows(args.eval)
    contract=json.loads(Path(args.prompt_contract).read_text()) if args.prompt_contract else None
    if {r['seed_group'] for r in train}&{r['seed_group'] for r in eval_rows}: raise ValueError('Seed-family leakage')
    if args.dry_run:
        print(json.dumps({'status':'data_contract_passed','train':len(train),'eval':len(eval_rows),'training_run':False})); return
    if not args.launch: raise ValueError('Use --dry-run or explicit --launch after GPU authorization')
    if not Path(args.model).is_dir(): raise FileNotFoundError('Local planner model weights missing')
    if Path(args.output).exists(): raise FileExistsError('Choose a new adapter output directory')
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText,TrainingArguments,Trainer,TrainerCallback
    from peft import LoraConfig,get_peft_model,TaskType
    from torch.utils.data import Dataset
    if not torch.cuda.is_available(): raise RuntimeError('No authorized visible CUDA device')
    os.environ['WANDB_DISABLED']='true'
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    architecture=AutoConfig.from_pretrained(args.model,local_files_only=True)
    model_class=AutoModelForImageTextToText if architecture.model_type.startswith('qwen3_vl') else AutoModelForCausalLM
    model=model_class.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='sdpa')
    model.config.use_cache=False
    if hasattr(model.config,'text_config'): model.config.text_config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=args.rank,lora_alpha=args.rank*2,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=0.05))
    model.enable_input_require_grads()

    class Records(Dataset):
        def __init__(self,rows):
            self.records=[]
            for row in rows:
                prompt=tokenizer.apply_chat_template(messages_for(row,contract),tokenize=False,add_generation_prompt=True)
                prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
                answer=tokenizer(json.dumps(row['expected_output'],ensure_ascii=False)+tokenizer.eos_token,add_special_tokens=False)['input_ids']
                if len(prefix)+len(answer)>args.max_length:
                    raise ValueError('Sample exceeds max length; increase --max-length, never truncate labels silently')
                ids=prefix+answer
                self.records.append({'input_ids':ids,'attention_mask':[1]*len(ids),
                    'labels':[-100]*len(prefix)+answer})
        def __len__(self): return len(self.records)
        def __getitem__(self,index): return self.records[index]

    def collate(records):
        width=((max(len(r['input_ids']) for r in records)+7)//8)*8
        return {key:torch.tensor([r[key]+[pad]*(width-len(r[key])) for r in records])
                for key,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]}

    class Profile(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            self.started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step in (2,10) or state.global_step%20==0:
                torch.cuda.synchronize()
                report={'optimizer_steps':state.global_step,'wall_seconds':time.monotonic()-self.started,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                report['seconds_per_optimizer_step']=report['wall_seconds']/state.global_step
                Path(args.output_dir).mkdir(parents=True,exist_ok=True)
                (Path(args.output_dir)/f'profile-step-{state.global_step}.json').write_text(json.dumps(report,indent=2))
                print(json.dumps({'training_profile':report}),flush=True)
                if profile_gate and state.global_step==2:
                    projected=report['seconds_per_optimizer_step']*max(state.max_steps-2,0)*1.5+report['wall_seconds']
                    gate={'passed':projected<training_seconds and report['peak_reserved_bytes']<23*1024**3,
                        'estimated_training_seconds_conservative':projected,'soft_limit':training_seconds,
                        'profile':report,'scope':'Resource feasibility only, not training quality.'}
                    (Path(args.output_dir)/'resource-gate.json').write_text(json.dumps(gate,indent=2))
                    if not gate['passed']: control.should_training_stop=True
            elapsed=time.monotonic()-self.started
            mean_step=elapsed/max(state.global_step,1)
            if elapsed+mean_step*1.5>=training_seconds:
                control.should_training_stop=True
                print(json.dumps({'early_stop':'soft_training_wall_budget','completed_steps':state.global_step}),flush=True)

    config=TrainingArguments(output_dir=args.output,per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,learning_rate=1e-4,max_steps=-1 if args.epochs else args.steps,
        num_train_epochs=args.epochs or 3,bf16=True,
        gradient_checkpointing=True,logging_steps=5,save_steps=50 if args.preserve_checkpoints else 20,
        save_total_limit=None if args.preserve_checkpoints else 3,report_to=[],seed=42,
        dataloader_num_workers=0)
    kwargs={'model':model,'args':config,'train_dataset':Records(train),'eval_dataset':Records(eval_rows),
        'data_collator':collate,'callbacks':[Profile()]}
    name='processing_class' if 'processing_class' in inspect.signature(Trainer).parameters else 'tokenizer'
    kwargs[name]=tokenizer
    training_seconds=args.training_seconds
    profile_gate=args.profile_gate
    trainer=Trainer(**kwargs); result=trainer.train()
    model.save_pretrained(Path(args.output)/'adapter'); tokenizer.save_pretrained(Path(args.output)/'adapter')
    metrics=trainer.evaluate()
    (Path(args.output)/'training-report.json').write_text(json.dumps({'train':result.metrics,'eval_loss':metrics,
        'task_accuracy_measured':False,'actual_optimizer_steps':trainer.state.global_step,
        'requested_max_steps':None if args.epochs else args.steps,'requested_epochs':args.epochs,
        'actual_epoch':trainer.state.epoch,'soft_training_seconds':training_seconds},indent=2),encoding='utf-8')


if __name__=='__main__': main()

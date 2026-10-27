#!/usr/bin/env python3
"""Collect and screen train-only teacher traces on a deterministic row shard."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from train_pilot import gen_ids,strict_correct,extract_think,valid_trace,read_jsonl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True)
    ap.add_argument('--seed',type=int,default=20281028);ap.add_argument('--cap',type=int,default=768)
    a=ap.parse_args()
    train=read_jsonl(a.run/'data/train.jsonl')
    if not (0<=a.start<=a.end<=len(train)): raise ValueError('invalid row shard bounds')
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,use_fast=True)
    device=torch.device(f'cuda:{a.gpu}')
    model=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    open_id=tok.convert_tokens_to_ids('<think>');close_id=tok.convert_tokens_to_ids('</think>')
    if open_id is None or close_id is None or open_id==tok.unk_token_id or close_id==tok.unk_token_id:
        raise RuntimeError('Qwen think markers are missing from the tokenizer vocabulary')
    accepted=[];all_rows=[];start_t=time.time()
    for ix in range(a.start,a.end):
        row=dict(train[ix]);torch.manual_seed(a.seed+ix);torch.cuda.manual_seed_all(a.seed+ix)
        ids,text,n=gen_ids(model,tok,row['prompt'],True,a.cap,1.0,device)
        if not text.startswith('<think>'):
            raise RuntimeError(f'unexpected thinking-mode prefix for train index {ix}: {text[:80]!r}')
        raw=ids.tolist()
        try:
            oi=raw.index(open_id);ci=raw.index(close_id,oi+1)
        except ValueError:
            oi=ci=-1
        closed=oi>=0 and ci>oi
        solution_ids=raw[ci+1:] if ci>=0 else []
        trace_ids=raw[oi+1:ci] if oi>=0 and ci>oi else []
        trace,solution=extract_think(text)
        answer_ok=bool(solution and strict_correct(solution,row['answer']))
        screen=bool(closed and answer_ok and trace and valid_trace(text,row))
        rec={'id':row['id'],'family':row['family'],'original_train_index':ix,'max_new_tokens':a.cap,'generated_tokens':n,
             'closed':closed,'answer_correct':answer_ok,'trace_screen_pass':screen,'teacher_raw':text}
        all_rows.append(rec)
        if screen:
            row.update({'teacher_trace':trace,'teacher_solution':solution,'teacher_trace_token_ids':trace_ids,
                        'teacher_solution_token_ids':solution_ids,'teacher_output_token_ids':raw,'teacher_output_tokens':n,
                        'teacher_verified':True,'original_train_index':ix,'teacher_seed':a.seed+ix,'teacher_model_revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'})
            accepted.append(row)
        if (ix-a.start+1)%25==0:
            print(json.dumps({'gpu':a.gpu,'processed':ix-a.start+1,'shard_n':a.end-a.start,'accepted':len(accepted),'elapsed_s':time.time()-start_t}),flush=True)
    out=a.run/'results';out.mkdir(exist_ok=True,parents=True)
    base=f'teacher_train_shard_{a.start:04d}_{a.end:04d}'
    (out/f'{base}_accepted.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in accepted),encoding='utf-8')
    (out/f'{base}_all.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in all_rows),encoding='utf-8')
    summary={'shard':[a.start,a.end],'n_input':len(all_rows),'n_closed':sum(x['closed'] for x in all_rows),'n_correct':sum(x['answer_correct'] for x in all_rows),
             'n_screened':len(accepted),'screened_rate_all':len(accepted)/max(1,len(all_rows)),'accepted_by_family':{},
             'input_by_family':{},'elapsed_s':time.time()-start_t,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),'seed_base':a.seed,'cap':a.cap}
    for fam in sorted({x['family'] for x in all_rows}):
        summary['input_by_family'][fam]=sum(x['family']==fam for x in all_rows)
        summary['accepted_by_family'][fam]=sum(x['family']==fam for x in accepted)
    (out/f'{base}_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

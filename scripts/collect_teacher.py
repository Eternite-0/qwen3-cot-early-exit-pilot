#!/usr/bin/env python3
"""Measure direct vs explicit-CoT teacher on dev and collect filtered train traces."""
from __future__ import annotations
import argparse, json, re, time, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_pilot import evaluate, gen, gen_ids, prompt_message, read_jsonl, strict_correct, extract_think, valid_trace

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--max-new',type=int,default=256);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--train-limit',type=int,default=0);ap.add_argument('--seed',type=int,default=20261025);ap.add_argument('--dev-only',action='store_true');ap.add_argument('--skip-dev',action='store_true');ap.add_argument('--dev-gate-metrics',type=Path,default=None);a=ap.parse_args()
    random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device(f'cuda:{a.gpu}')
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    dev=read_jsonl(a.run/'data/dev.jsonl');train=read_jsonl(a.run/'data/train.jsonl')
    if a.limit: dev=dev[:a.limit];train=train[:a.limit]
    details=[];start=time.time()
    if a.skip_dev:
        if a.dev_gate_metrics is None: raise RuntimeError('--skip-dev requires frozen --dev-gate-metrics')
        metrics=json.loads(a.dev_gate_metrics.read_text(encoding='utf-8'))
        metrics={'n':metrics['n'],'explicit_cot_accuracy':metrics['answer_correct']/max(1,metrics['completed']),
                 'trace_screen_rate':metrics['trace_screen_pass']/max(1,metrics['completed']),
                 'trace_screen_rate_all':metrics['trace_screen_pass']/max(1,metrics['n']),'completed_rate':metrics['completed']/max(1,metrics['n']),
                 'gate_source':str(a.dev_gate_metrics),'elapsed_s':0.0}
        print(json.dumps({'frozen_dev_gate':metrics},ensure_ascii=False),flush=True)
    else:
        for i,row in enumerate(dev):
            d,dt=gen(model,tok,row['prompt'],False,128,1.0,device)
            cids,c,ct=gen_ids(model,tok,row['prompt'],True,a.max_new,1.0,device)
            trace,sol=extract_think(c)
            d_ok=strict_correct(d,row['answer']); c_ok=bool(sol and strict_correct(sol,row['answer']))
            reliable=bool(c_ok and trace and valid_trace(c,row))
            details.append({'id':row['id'],'family':row['family'],'gold':row['answer'],'direct':d,'direct_correct':d_ok,'direct_tokens':dt,'cot':c,'cot_correct':c_ok,'cot_tokens':ct,'trace_present':trace is not None,'trace_reliable_screen':reliable})
            if (i+1)%25==0: print(json.dumps({'dev_done':i+1,'direct_accuracy':sum(x['direct_correct'] for x in details)/len(details),'cot_accuracy':sum(x['cot_correct'] for x in details)/len(details),'trace_screen_rate':sum(x['trace_reliable_screen'] for x in details)/len(details),'elapsed_s':time.time()-start}),flush=True)
        metrics={'n':len(details),'direct_accuracy':sum(x['direct_correct'] for x in details)/max(1,len(details)),'explicit_cot_accuracy':sum(x['cot_correct'] for x in details)/max(1,len(details)),'trace_screen_rate':sum(x['trace_reliable_screen'] for x in details)/max(1,len(details)),'elapsed_s':time.time()-start}
    (a.run/'results').mkdir(exist_ok=True,parents=True)
    if details:
        (a.run/'results/teacher_dev.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in details),encoding='utf-8')
        (a.run/'results/teacher_dev_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(metrics),flush=True)
    if a.dev_only:
        print('DEV_ONLY_COMPLETE',flush=True);return
    # Only continue to train traces after the independent dev signal shows useful teacher CoT.
    # The dev-only completion diagnostic freezes the later train-collection gate.
    # Do not let an incomplete 256-token response censor this train-only gate.
    if metrics['trace_screen_rate'] < 0.5:
        print('STOP_TRAIN_TRACE_COLLECTION: teacher CoT did not pass frozen dev gate',flush=True);return
    accepted=[]; reject={};train=train[:a.train_limit] if a.train_limit else train
    collection_start=time.time()
    for i,row in enumerate(train):
        out_ids,out,n=gen_ids(model,tok,row['prompt'],True,a.max_new,1.0,device)
        trace,solution=extract_think(out)
        ok=bool(solution and strict_correct(solution,row['answer']) and valid_trace(out,row))
        if ok:
            close_id=tok.convert_tokens_to_ids('</think>')
            ids=out_ids.tolist()
            try:
                ci=ids.index(close_id)
            except ValueError:
                reject[row['family']]=reject.get(row['family'],0)+1
                continue
            row.update({'teacher_trace':trace,'teacher_solution':solution,'teacher_trace_token_ids':ids[:ci],
                        'teacher_solution_token_ids':ids[ci+1:],'teacher_output_token_ids':ids,
                        'teacher_output_tokens':n,'teacher_verified':True})
            accepted.append(row)
        else:
            reject[row['family']]=reject.get(row['family'],0)+1
        if (i+1)%50==0: print(json.dumps({'train_done':i+1,'accepted':len(accepted),'accept_rate':len(accepted)/(i+1),'reject_by_family':reject,'elapsed_s':time.time()-start}),flush=True)
    path=a.run/'data/train_teacher_verified.jsonl'
    path.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in accepted),encoding='utf-8')
    (a.run/'results/teacher_collection.json').write_text(json.dumps({'n_input':len(train),'n_accepted':len(accepted),'accepted_by_family':{f:sum(x['family']==f for x in accepted) for f in sorted({x['family'] for x in train})},'rejected_by_family':reject,'elapsed_s':time.time()-collection_start,'max_new_tokens':a.max_new,'quality_gate_basis':'dev-only 768-token stratified diagnostic; train rows require closed, correct and original executable trace screen'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()

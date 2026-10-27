#!/usr/bin/env python3
"""Five interleaved stochastic generations/item, active unmerged LoRA."""
from pathlib import Path
import argparse,gc,json,random,time,torch
from matched_common import read,write,load_model,load_tokenizer,sha,POLICY_PREFIX
from train_pilot import gen_ids,parse_direct_label,extract_think

GROUPS={0:['direct','matched_sft_answer_a1_seed20281028','matched_msd_a1_seed20281028'],1:['direct','matched_sft_trace_a0_seed20281028','matched_sft_trace_a0.5_seed20281028'],2:['direct','matched_sft_trace_a1_seed20281028','matched_msd_a0_seed20281028'],3:['direct','matched_msd_a0.5_seed20281028','program_answer_sft64_seed20261025']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--group',type=int,choices=range(4),required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--repeats',type=int,default=5);p.add_argument('--split',choices=['dev','v6','external'],required=True);a=p.parse_args()
    run=a.run;device=torch.device(f'cuda:{a.gpu}');tok=load_tokenizer(run);freeze=json.loads((run/'configs/matched_models_frozen.json').read_text())
    source={'v6':run/'data/synthetic_test_v6_unopened.jsonl','dev':run/'data/dev.jsonl','external':run/'data/external_regression/evaluation.jsonl'}[a.split]
    if a.split=='v6':
        assert sha(source)==freeze['test_sha256']
        for c in freeze['head_conditions']:assert (run/'configs'/f'{POLICY_PREFIX}{c}.json').exists()
        assert (run/'configs/final_evaluation_frozen.json').exists()
    rows=read(source);out=run/'results'/f'matched_{a.split}_generation_group{a.group}_predictions.jsonl'
    if out.exists():raise RuntimeError('refusing to rerun or overwrite held-out predictions')
    t=time.time();models={};cold={}
    for condition in GROUPS[a.group]:
        start=time.time();models[condition]=load_model(run,condition,device);torch.cuda.synchronize(device);cold[condition]=time.time()-start
    dev=read(run/'data/dev.jsonl')
    for cond,m in models.items():
        for row in dev[:10]:gen_ids(m,tok,row['prompt'],False,128,1.,device)
    start=time.time();count=0
    with out.open('w',encoding='utf-8') as stream:
        for rep in range(a.repeats):
            order=list(range(len(rows)));random.Random(20291031+rep).shuffle(order)
            for ix in order:
                row=rows[ix];conditions=list(models);random.Random(20291031+rep*100003+ix).shuffle(conditions)
                if a.split=='v6' and a.group==0 and ix%7==0:
                    conditions.append('explicit_cot')
                    random.Random(20291031+rep*100003+ix+1).shuffle(conditions)
                for cond in conditions:
                    torch.manual_seed(20291031+rep*100003+ix);torch.cuda.manual_seed_all(20291031+rep*100003+ix)
                    is_cot=cond=='explicit_cot';m=models['direct'] if is_cot else models[cond]
                    torch.cuda.synchronize(device);ts=time.perf_counter();ids,text,n=gen_ids(m,tok,row['prompt'],is_cot,768 if is_cot else 128,1.,device);torch.cuda.synchronize(device);dt=time.perf_counter()-ts
                    trace,solution=extract_think(text) if is_cot else (None,None)
                    label=parse_direct_label(solution,row['labels']) if is_cot and solution is not None else (None if is_cot else parse_direct_label(text,row['labels']))
                    rec={'id':row['id'],'family':row['family'],'split':a.split,'condition':cond,'group':a.group,'repeat':rep,'gold':row['answer'],'prediction':label,'prediction_raw':text,'correct':label==row['answer'],'schema_valid':label is not None,'generated_tokens':n,'latency_s':dt,'executed_layers':28,'input_characters':len(row['prompt']),'candidate_count':len(row['labels']),'input_difficulty':row.get('difficulty'), 'inference':'BF16 Transformers SDPA; active unmerged LoRA for trained models'}
                    if is_cot:rec.update(trace_closed=trace is not None,comparison_subset='indices_mod7_zero_n72')
                    stream.write(json.dumps(rec,ensure_ascii=False)+'\n');count+=1
            stream.flush();print(json.dumps({'group':a.group,'split':a.split,'repeat':rep+1,'elapsed_s':time.time()-start,'rows_written':count}),flush=True)
    write(out.with_name(out.name.replace('_predictions.jsonl','_runtime.json')),{'split':a.split,'group':a.group,'conditions':list(models),'n_items':len(rows),'repeats':a.repeats,'warm_request_elapsed_s':time.time()-start,'process_elapsed_s':time.time()-t,'cold_model_load_s':cold,'peak_memory_bytes':torch.cuda.max_memory_allocated(device),'prediction_sha256':sha(out),'source_sha256':sha(source),'seed':20291031,'parallel_conditions_in_request':False})
if __name__=='__main__':main()

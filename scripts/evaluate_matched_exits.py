#!/usr/bin/env python3
"""Serial, shuffled fixed exits and only dev-eligible incremental policies."""
from pathlib import Path
import argparse,json,random,time,torch
from matched_common import read,write,load_model,load_tokenizer,sha,DEPTHS,POLICY_PREFIX
from decision_paths import native_scores,load_heads,fixed_head,incremental_head

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--repeats',type=int,default=5);p.add_argument('--split',choices=['dev_smoke','v6'],required=True);a=p.parse_args()
    run=a.run;device=torch.device(f'cuda:{a.gpu}');freeze=json.loads((run/'configs/matched_models_frozen.json').read_text());conditions=freeze['head_conditions'];tok=load_tokenizer(run);rows=read(run/('data/dev.jsonl' if a.split=='dev_smoke' else 'data/synthetic_test_v6_unopened.jsonl'))
    if a.split=='dev_smoke':rows=rows[:6]
    else:
        assert sha(run/'data/synthetic_test_v6_unopened.jsonl')==freeze['test_sha256']
        assert (run/'configs/final_evaluation_frozen.json').exists()
    out=run/'results'/f'matched_{a.split}_exit_predictions.jsonl'
    if out.exists():raise RuntimeError('refusing to rerun or overwrite predictions')
    models={};heads={};policies={};cold={};ts=time.time()
    for c in conditions:
        t=time.time();models[c]=load_model(run,c,device);heads[c]=load_heads(run,c,device);torch.cuda.synchronize(device);cold[c]=time.time()-t
        policies[c]=json.loads((run/'configs'/f'{POLICY_PREFIX}{c}.json').read_text())['selected']
    configs=[(c,kind,d) for c in conditions for kind in ('native','trained') for d in DEPTHS]
    configs.extend((c,'dynamic',None) for c in conditions if policies[c] is not None)
    def call(c,kind,d,row):
        if kind=='native':return native_scores(models[c],tok,row,device,d)
        if kind=='trained':return fixed_head(models[c],heads[c],tok,row,device,d)
        return incremental_head(models[c],heads[c],tok,row,device,policies[c])
    with torch.inference_mode():
        warm=read(run/'data/dev.jsonl')[:10]
        for c,kind,d in configs:
            for row in warm:call(c,kind,d,row)
        start=time.time();count=0
        with out.open('w',encoding='utf-8') as stream:
            for rep in range(a.repeats):
                schedule=[(ix,c,kind,d) for ix in range(len(rows)) for c,kind,d in configs];random.Random(20291101+rep).shuffle(schedule)
                for ix,c,kind,d in schedule:
                    row=rows[ix];torch.cuda.synchronize(device);t=time.perf_counter();z=call(c,kind,d,row);torch.cuda.synchronize(device);dt=time.perf_counter()-t
                    rec={'id':row['id'],'family':row['family'],'split':a.split,'backbone':c,'condition':f'{c}/{kind}/L{d}' if d is not None else f'{c}/dynamic','readout':kind,'depth':d,'repeat':rep,'gold':row['answer'],'correct':z['prediction']==row['answer'],'schema_valid':True,'latency_s':dt,'input_characters':len(row['prompt']),'candidate_count':len(row['labels']),'input_difficulty':row.get('difficulty'),**z}
                    stream.write(json.dumps(rec,ensure_ascii=False)+'\n');count+=1
                stream.flush();print(json.dumps({'split':a.split,'repeat':rep+1,'rows_written':count,'elapsed_s':time.time()-start}),flush=True)
    write(out.with_name(out.name.replace('_predictions.jsonl','_runtime.json')),{'n_items':len(rows),'repeats':a.repeats,'n_conditions':len(configs),'conditions':configs,'warm_request_elapsed_s':time.time()-start,'process_elapsed_s':time.time()-ts,'cold_model_load_s':cold,'peak_memory_bytes':torch.cuda.max_memory_allocated(device),'prediction_sha256':sha(out),'model_weights':'BF16 original plus active unmerged LoRA where trained','latency_includes':'tokenization, candidate embeddings/scoring, head/stopper, actual forward execution, synchronization','seed':20291101})
if __name__=='__main__':main()

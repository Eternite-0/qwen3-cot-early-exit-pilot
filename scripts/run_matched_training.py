#!/usr/bin/env python3
"""Run the frozen, one-seed matched training jobs on one existing GPU."""
from pathlib import Path
import argparse,json,subprocess,sys,time,hashlib

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--group',choices=['sft','msd'],required=True);a=p.parse_args()
    run=a.run.resolve(); jobs=([('sft_answer',1.0),('sft_trace',0.0),('sft_trace',0.5),('sft_trace',1.0)] if a.group=='sft' else [('msd',0.0),('msd',0.5),('msd',1.0)])
    status=[]
    for method,alpha in jobs:
        name=f'matched_{method}_a{alpha:g}_seed20281028'
        log=run/'logs'/f'{name}.log';dest=run/'results'/name
        if (dest/'train_log.json').exists():
            status.append({'name':name,'status':'already_complete'});continue
        cmd=[sys.executable,str(run/'scripts/train_pilot.py'),'--model',str(run/'models/Qwen3-1.7B'),'--run',str(run),'--train-file','data/train_teacher_verified.jsonl','--method',method,'--alpha',str(alpha),'--steps','64','--seed','20281028','--student-response-cap','256','--gpu',str(a.gpu),'--run-name',name]
        t=time.time();print(json.dumps({'event':'start','name':name,'gpu':a.gpu,'command':cmd}),flush=True)
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,cwd=run)
        record={'name':name,'status':'complete' if result.returncode==0 else 'failed','returncode':result.returncode,'elapsed_s':time.time()-t,'gpu':a.gpu,'log':str(log.relative_to(run))}
        status.append(record); print(json.dumps(record),flush=True)
        (run/'results'/f'matched_training_status_{a.group}.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        if result.returncode:sys.exit(result.returncode)
if __name__=='__main__':main()

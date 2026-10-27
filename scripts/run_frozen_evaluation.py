#!/usr/bin/env python3
"""Resume only complete stages of the immutable, serial final evaluation."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();r=a.run.resolve()
    freeze=json.loads((r/'configs/final_evaluation_frozen.json').read_text())
    for rel,expected in freeze['hashes'].items():
        actual=hashlib.sha256((r/rel).read_bytes()).hexdigest()
        if actual!=expected:raise RuntimeError(f'freeze mismatch: {rel}')
    jobs=[]
    for group in range(4):jobs.append((f'v6_generation_group{group}','evaluate_matched_generation.py',['--group',str(group),'--split','v6'],r/'results'/f'matched_v6_generation_group{group}_runtime.json'))
    jobs.append(('v6_exits','evaluate_matched_exits.py',['--split','v6'],r/'results/matched_v6_exit_runtime.json'))
    for group in range(4):jobs.append((f'external_generation_group{group}','evaluate_matched_generation.py',['--group',str(group),'--split','external'],r/'results'/f'matched_external_generation_group{group}_runtime.json'))
    status=[]
    for name,script,args,completed in jobs:
        if completed.exists():
            status.append({'stage':name,'status':'already_complete'});continue
        cmd=[sys.executable,str(r/'scripts'/script),'--run',str(r),'--gpu',str(a.gpu),'--repeats','5',*args];start=time.time()
        print(json.dumps({'event':'start','stage':name,'command':cmd}),flush=True)
        with (r/'logs'/f'final_{name}.log').open('w',encoding='utf-8') as log:res=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=r)
        d={'stage':name,'status':'complete' if res.returncode==0 and completed.exists() else 'failed','returncode':res.returncode,'seconds':time.time()-start,'gpu':a.gpu};status.append(d)
        (r/'results/final_evaluation_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8');print(json.dumps(d),flush=True)
        if d['status']!='complete':raise SystemExit(res.returncode or 1)
    print('FINAL_EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()

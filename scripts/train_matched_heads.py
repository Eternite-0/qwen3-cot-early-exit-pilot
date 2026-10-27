#!/usr/bin/env python3
"""Same candidate-conditioned head initialization/ordering on each frozen backbone."""
from pathlib import Path
import argparse,json,random,time,torch
import torch.nn.functional as F
from train_readout import ChoiceHead,extract,build_candidate_tensor,metrics,DEPTHS
from matched_common import read,write,load_model,load_tokenizer,sha,HEAD_ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--condition',required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    run=a.run;device=torch.device(f'cuda:{a.gpu}');seed=20281028
    out=run/HEAD_ROOT/a.condition
    if (out/'metrics.json').exists():raise RuntimeError('heads already complete; refusing overwrite')
    out.mkdir(parents=True,exist_ok=True)
    t=time.time();model=load_model(run,a.condition,device);tok=load_tokenizer(run)
    train=read(run/'data/train_teacher_verified.jsonl');dev=read(run/'data/dev.jsonl')
    trf,trr=extract(model,tok,train,device,'train',run);dvf,dvr=extract(model,tok,dev,device,'dev',run)
    embed=model.get_input_embeddings().weight.detach().float().cpu()
    tc,tm,ty=build_candidate_tensor(tok,embed,trr,device);dc,dm,dy=build_candidate_tensor(tok,embed,dvr,device)
    result={'condition':a.condition,'seed':seed,'train_n':len(train),'dev_n':len(dev),'epochs':8,'train_sha256':sha(run/'data/train_teacher_verified.jsonl'),'depths':{},'method':'frozen backbone; candidate-label embeddings; pairwise choice head','inference_weights':'unchanged BF16 base plus active unmerged LoRA projections; matches training dev path'}
    for depth in DEPTHS:
        torch.manual_seed(seed+depth);head=ChoiceHead().to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4);best=None;curves=[]
        for epoch in range(8):
            head.train();order=torch.randperm(len(trr),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            for start in range(0,len(order),32):
                ix=order[start:start+32];h=torch.stack([trf[i][depth].float() for i in ix]).to(device)
                logits=head(h,tc[ix]).masked_fill(~tm[ix],-1e9);loss=F.cross_entropy(logits,ty[ix]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.0);opt.step()
            dmets=metrics(head,dvf,dvr,depth,dc,dm,dy,device);tmets=metrics(head,trf,trr,depth,tc,tm,ty,device)
            curves.append({'epoch':epoch+1,'dev_accuracy':dmets['accuracy'],'train_accuracy':tmets['accuracy'],'dev_cross_entropy':dmets['cross_entropy']})
            if best is None or dmets['accuracy']>best['dev']['accuracy']:
                best={'epoch':epoch+1,'dev':dmets,'train':tmets,'state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}
        torch.save(best.pop('state'),out/f'head_layer{depth}.pt')
        best['dev'].pop('predictions',None);best['train'].pop('predictions',None);best['curve']=curves;result['depths'][str(depth)]=best
        print(json.dumps({'condition':a.condition,'depth':depth,'dev':best['dev']['accuracy'],'epoch':best['epoch']}),flush=True)
    result.update(elapsed_s=time.time()-t,gpu_hours=(time.time()-t)/3600,peak_memory_bytes=torch.cuda.max_memory_allocated(device))
    write(out/'metrics.json',result)
    print(json.dumps({'condition':a.condition,'elapsed_s':result['elapsed_s'],'gpu_hours':result['gpu_hours']}),flush=True)
if __name__=='__main__':main()

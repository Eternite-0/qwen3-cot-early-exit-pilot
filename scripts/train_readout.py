#!/usr/bin/env python3
"""Train candidate-conditioned lightweight heads on frozen Qwen3 hidden states."""
from __future__ import annotations
import argparse,json,time,random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

DEPTHS=[2,4,8,12,16,20,24,28]

def read(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]
def qwen_prompt(tok,prompt):
    ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'input_ids'): ids=ids.input_ids
    if hasattr(ids,'ids'): ids=ids.ids
    return list(ids)

@torch.inference_mode()
def extract(model,tok,rows,device,split,run):
    model.eval(); cands=[]; feats=[]; start=time.time()
    for i,row in enumerate(rows):
        ids=qwen_prompt(tok,row['prompt'])
        x=torch.tensor([ids],device=device)
        hidden=[]
        handles=[]
        for depth in DEPTHS:
            def hook(module,inputs,output,depth=depth): hidden.append((depth,output[0] if isinstance(output,tuple) else output))
            handles.append(model.model.layers[depth-1].register_forward_hook(hook))
        _=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True)
        for h in handles:h.remove()
        # Qwen3 hidden_states[0] is embedding output, [d] is after d decoder blocks.
        feats.append({d:h[0,-1].to('cpu',dtype=torch.float16) for d,h in hidden})
        cands.append({'labels':row['labels'],'answer':row['answer'],'family':row['family'],'id':row['id']})
        if (i+1)%250==0: print(json.dumps({'split':split,'encoded':i+1,'n':len(rows),'elapsed_s':time.time()-start}),flush=True)
    return feats,cands

class ChoiceHead(nn.Module):
    def __init__(self,hid=2048,proj=256):
        super().__init__(); self.hnorm=nn.LayerNorm(hid);self.cnorm=nn.LayerNorm(hid)
        self.hp=nn.Linear(hid,proj);self.cp=nn.Linear(hid,proj)
        self.scorer=nn.Sequential(nn.Linear(proj*3,proj),nn.GELU(),nn.Linear(proj,1))
    def forward(self,h,c):
        # h: [B,H], c: [B,K,H]
        a=F.gelu(self.hp(self.hnorm(h)))[:,None,:].expand(-1,c.shape[1],-1)
        b=F.gelu(self.cp(self.cnorm(c)))
        return self.scorer(torch.cat([a,b,a*b],dim=-1)).squeeze(-1)

def build_candidate_tensor(tok,embedding,rows,device):
    maxk=max(len(r['labels']) for r in rows); dim=embedding.shape[1]
    out=torch.zeros((len(rows),maxk,dim),dtype=torch.float32)
    mask=torch.zeros((len(rows),maxk),dtype=torch.bool); targets=torch.zeros(len(rows),dtype=torch.long)
    for i,r in enumerate(rows):
        for j,label in enumerate(r['labels']):
            ids=tok(str(label),add_special_tokens=False).input_ids
            if not ids: raise ValueError(f'empty candidate label {label!r}')
            out[i,j]=embedding[torch.tensor(ids)].float().mean(0)
            mask[i,j]=True
            if label==r['answer']: targets[i]=j
        if r['answer'] not in r['labels']: raise ValueError('gold label absent from candidates')
    return out.to(device),mask.to(device),targets.to(device)

def metrics(head,feats,rows,depth,cand,mask,target,device,batch=64):
    head.eval(); pred=[]; losses=[]
    with torch.no_grad():
        for s in range(0,len(rows),batch):
            e=min(len(rows),s+batch); h=torch.stack([x[depth].float() for x in feats[s:e]]).to(device); c=cand[s:e]; m=mask[s:e]; t=target[s:e]
            logits=head(h,c).masked_fill(~m,-1e9);losses.append(F.cross_entropy(logits,t,reduction='sum').item());pred.extend(logits.argmax(-1).cpu().tolist())
    ok=[]
    for i,r in enumerate(rows):
        # Rows passed to metrics are candidate metadata in the same order.
        ok.append(int(r['labels'][pred[i]]==r['answer']))
    fam={}
    for i,r in enumerate(rows):
        a=fam.setdefault(r['family'],[0,0]);a[0]+=ok[i];a[1]+=1
    return {'n':len(rows),'correct':sum(ok),'accuracy':sum(ok)/len(ok),'cross_entropy':sum(losses)/len(rows),'by_family':{k:{'correct':v[0],'n':v[1],'accuracy':v[0]/v[1]} for k,v in fam.items()},'predictions':pred}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--seed',type=int,default=20261025);a=ap.parse_args()
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);device=torch.device(f'cuda:{a.gpu}')
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    train=read(a.run/'data/train.jsonl');dev=read(a.run/'data/dev.jsonl')
    t0=time.time();trf,trr=extract(model,tok,train,device,'train',a.run);dvf,dvr=extract(model,tok,dev,device,'dev',a.run)
    embed=model.get_input_embeddings().weight.detach().to('cpu',dtype=torch.float32)
    tc,tm,ty=build_candidate_tensor(tok,embed,trr,device);dc,dm,dy=build_candidate_tensor(tok,embed,dvr,device)
    outdir=a.run/'checkpoints'/'readout_heads';outdir.mkdir(parents=True,exist_ok=True)
    result={'depths':{},'train_n':len(train),'dev_n':len(dev),'seed':a.seed,'elapsed_encode_s':time.time()-t0,'frozen_backbone':True,'candidate_conditioned':True}
    for depth in DEPTHS:
        head=ChoiceHead().to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4);best=None
        for epoch in range(a.epochs):
            head.train(); order=torch.randperm(len(trr),device=device)
            for s in range(0,len(trr),32):
                idx=order[s:s+32];h=torch.stack([trf[int(i)][depth].float() for i in idx.cpu().tolist()]).to(device);c=tc[idx];m=tm[idx];y=ty[idx]
                logits=head(h,c).masked_fill(~m,-1e9);loss=F.cross_entropy(logits,y)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.0);opt.step()
            devm=metrics(head,dvf,dvr,depth,dc,dm,dy,device);trainm=metrics(head,trf,trr,depth,tc,tm,ty,device)
            if best is None or devm['accuracy']>best['dev']['accuracy']: best={'epoch':epoch+1,'dev':devm,'train':trainm,'state':{k:v.detach().cpu() for k,v in head.state_dict().items()}}
        torch.save(best.pop('state'),outdir/f'head_layer{depth}.pt')
        best['dev'].pop('predictions',None);best['train'].pop('predictions',None);result['depths'][str(depth)]=best
        print(json.dumps({'depth':depth,'best_epoch':best['epoch'],'dev_accuracy':best['dev']['accuracy'],'train_accuracy':best['train']['accuracy']}),flush=True)
    (a.run/'results'/'readout_dev_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'elapsed_total_s':time.time()-t0,'depth_accuracy':{k:v['dev']['accuracy'] for k,v in result['depths'].items()}},ensure_ascii=False),flush=True)
if __name__=='__main__': main()

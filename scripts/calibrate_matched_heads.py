#!/usr/bin/env python3
"""Dev-only rule selection, paired quality gate, and actual incremental smoke."""
from pathlib import Path
import argparse,json,time,math,random,torch,numpy as np
from train_readout import qwen_prompt
from decision_paths import load_heads,rule_accept,incremental_head,fixed_head
from evaluate_readout import candidate_embeddings,predict_head
from matched_common import read,write,load_model,load_tokenizer,DEPTHS,POLICY_PREFIX

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--condition',required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    device=torch.device(f'cuda:{a.gpu}');run=a.run;tok=load_tokenizer(run);model=load_model(run,a.condition,device);heads=load_heads(run,a.condition,device);rows=read(run/'data/dev.jsonl');t=time.time();preds=[]
    with torch.inference_mode():
        for row in rows:
            states={};hooks=[]
            for d in DEPTHS:
                def hook(module,inputs,output,d=d):states[d]=(output[0] if isinstance(output,tuple) else output)[0,-1].float()
                hooks.append(model.model.layers[d-1].register_forward_hook(hook))
            x=torch.tensor([qwen_prompt(tok,row['prompt'])],device=device)
            try:model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False)
            finally:
                for hook in hooks:hook.remove()
            cand=candidate_embeddings(tok,model.get_input_embeddings().weight.detach(),row,device)
            rec={'id':row['id'],'family':row['family'],'gold':row['answer'],'depths':{}}
            for d in DEPTHS:
                lab,conf,probs=predict_head(heads[d],states[d],cand,row['labels']);rec['depths'][str(d)]={'prediction':lab,'confidence':conf,'probabilities':probs,'correct':lab==row['answer']}
            preds.append(rec)
    base=np.array([r['depths']['28']['correct'] for r in preds],float);candidates=[];rng=np.random.default_rng(20281030)
    boot=np.array([rng.integers(0,len(rows),len(rows)) for _ in range(5000)])
    grid={'confidence':[.5,.6,.7,.8,.9,.95,.99],'margin':[.1,.2,.3,.4,.5,.7,.9],'entropy':[.1,.2,.3,.5,.7,.9],'agreement':[.5,.6,.7,.8,.9,.95,.99]}
    for kind,thresholds in grid.items():
        for tau in thresholds:
            exits=[];oks=[]
            for r in preds:
                previous=None;d=28
                for depth in (4,8,12,16,20,24,28):
                    z=r['depths'][str(depth)]
                    if depth==28 or (depth>=8 and rule_accept(z['probabilities'],previous,kind,tau)):
                        d=depth;break
                    previous=z['prediction']
                exits.append(d);oks.append(r['depths'][str(d)]['correct'])
            delta=np.array(oks,float)-base;ci=np.quantile(delta[boot].mean(1),[.025,.975]).tolist()
            early=[i for i,d in enumerate(exits) if d<28]
            candidates.append({'kind':kind,'threshold':tau,'accuracy':float(np.mean(oks)),'mean_executed_layers':float(np.mean(exits)),'paired_vs_full_ci95':ci,'coverage':len(early)/len(rows),'early_risk':float(np.mean([not oks[i] for i in early])) if early else None,'exit_distribution':{str(d):exits.count(d) for d in sorted(set(exits))},'quality_eligible':bool(ci[0]>=-.02 and np.mean(exits)<=22.4)})
    eligible=sorted([r for r in candidates if r['quality_eligible']],key=lambda r:(r['mean_executed_layers'],-r['accuracy'],r['kind'],r['threshold']))
    selected=None;timing=None
    if eligible:
        rule=eligible[0];lat_fixed=[];lat_dynamic=[];consistency=True
        with torch.inference_mode():
            for row in rows[:10]:fixed_head(model,heads,tok,row,device,28);incremental_head(model,heads,tok,row,device,rule)
            schedule=[(rep,ix,mode) for rep in range(5) for ix in range(len(rows)) for mode in (0,1)];random.Random(20281030).shuffle(schedule)
            for rep,ix,mode in schedule:
                torch.cuda.synchronize(device);ts=time.perf_counter()
                z=incremental_head(model,heads,tok,rows[ix],device,rule) if mode else fixed_head(model,heads,tok,rows[ix],device,28)
                torch.cuda.synchronize(device);dt=time.perf_counter()-ts;(lat_dynamic if mode else lat_fixed).append(dt)
                if mode:consistency &= (z['prediction']==preds[ix]['depths'][str(z['executed_layers'])]['prediction'])
        timing={'fixed_p50_s':float(np.median(lat_fixed)),'fixed_p95_s':float(np.quantile(lat_fixed,.95)),'dynamic_p50_s':float(np.median(lat_dynamic)),'dynamic_p95_s':float(np.quantile(lat_dynamic,.95)),'incremental_prediction_consistency':consistency,'repeats':5}
        if consistency and timing['dynamic_p50_s']<=.8*timing['fixed_p50_s'] and timing['dynamic_p95_s']<=timing['fixed_p95_s']:selected=rule
    # Even with no eligible threshold, verify the incremental loop against the
    # cached full-forward hidden-state readout on ten dev items only.
    smoke=[]
    with torch.inference_mode():
        for i,row in enumerate(rows[:10]):
            z=incremental_head(model,heads,tok,row,device,{'kind':'confidence','threshold':1.1})
            assert z['executed_layers']==28 and z['prediction']==preds[i]['depths']['28']['prediction']
            smoke.append({'id':row['id'],'same_prediction':True,'executed_layers':28})
    out=run/'configs'/f'{POLICY_PREFIX}{a.condition}.json'
    report={'condition':a.condition,'dev_n':len(rows),'reference_full_head_accuracy':float(base.mean()),'selected':selected,'all_rules':candidates,'timing':timing,'incremental_full_depth_smoke':smoke,'selection_uses_test':False,'elapsed_s':time.time()-t,'gate':'paired 95% lower >= -0.02 vs full head; mean layers <=22.4; actual dev p50<=0.8 reference and p95<=reference; full direct-generation comparison reported separately','sample_size_analysis':{'max_observed_disagreement_variance':max(float(np.mean((np.array([r['depths']['8']['correct'] for r in preds],float)-base)**2)),0),'conservative_n_for_2pp_at_95pct':int(math.ceil(1.96**2*max(float(np.var(np.array([r['depths']['8']['correct'] for r in preds],float)-base)),1e-9)/.02**2)),'test_n_planned':500,'note':'normal approximation from dev paired differences; final inference uses paired item bootstrap'}}
    write(out,report)
    path=run/'results'/f'matched_head_dev_unmerged_{a.condition}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in preds),encoding='utf-8')
    print(json.dumps({'condition':a.condition,'dev_full_accuracy':float(base.mean()),'selected':selected,'elapsed_s':time.time()-t,'sample_size_analysis':report['sample_size_analysis']}),flush=True)
if __name__=='__main__':main()

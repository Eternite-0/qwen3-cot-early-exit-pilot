#!/usr/bin/env python3
"""Analysis of saved matched predictions; no model execution or test-guided tuning."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,csv,hashlib,json,math,numpy as np
POLICY_PREFIX='matched_policy_unmerged_'
def read(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def semantic_key(row):
    p={k:v for k,v in row.get('parameters',{}).items() if k!='case_ref'}
    if 'edges' in p:p['edges']=sorted(p['edges'])
    if 'evidence' in p:p['evidence']=sorted(p['evidence'])
    return hashlib.sha256(json.dumps([row['family'],p],sort_keys=True).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'results';rng=np.random.default_rng(20291102)
    assert (run/'configs/final_evaluation_frozen.json').exists()
    items=read(run/'data/synthetic_test_v6_unopened.jsonl');meta={r['id']:r for r in items}
    ext=read(run/'data/external_regression/evaluation.jsonl');meta.update({r['id']:r for r in ext})
    groups={};sources=[]
    for split in ('v6','external'):
        for i in range(4):
            path=out/f'matched_{split}_generation_group{i}_predictions.jsonl';runtime=path.with_name(path.name.replace('_predictions.jsonl','_runtime.json'))
            if not path.exists() or not runtime.exists():raise RuntimeError(f'incomplete final evaluation: {path.name}')
            sources.append(path)
            for rec in read(path):
                key=f'{split}/generation/{rec["condition"]}'
                if rec['condition']=='direct' and i!=0:key=f'{split}/generation/direct_group{i}'
                groups.setdefault(key,[]).append(rec)
    path=out/'matched_v6_exit_predictions.jsonl'
    if not path.exists() or not path.with_name('matched_v6_exit_runtime.json').exists():raise RuntimeError('fixed-exit evaluation incomplete')
    sources.append(path)
    for rec in read(path):groups.setdefault('v6/exit/'+rec['condition'],[]).append(rec)
    # Semantic clusters remove random case references; shared latent problems
    # are never counted as independent bootstrap units.
    cluster={sid:(semantic_key(r) if 'parameters' in r else sid) for sid,r in meta.items()}
    reference_rows={};leak={}
    test_clusters={semantic_key(r) for r in items}
    for split in ('train','dev','pilot_holdout'):
        rows=read(run/f'data/{split}.jsonl');cs={semantic_key(r) for r in rows}
        leak[split]={'test_semantic_clusters_shared':len(test_clusters&cs),'test_items_with_semantics_seen':sum(semantic_key(r) in cs for r in items),'exact_prompt_overlap':len({r['prompt'] for r in items}&{r['prompt'] for r in rows})}
    write(out/'semantic_overlap_audit.json',{'case_reference_removed':True,'test_n_items':len(items),'test_n_semantic_clusters':len(test_clusters),'overlap':leak,'consequence':'unique case IDs are not proof of disjoint latent task structures; results remain target-informed generator transfer'})

    def bootstrap(values,ids,B=10000,quantiles=(.025,.975)):
        fam=defaultdict(lambda:defaultdict(list))
        for sid in ids:fam[meta[sid]['family']][cluster[sid]].append(values[sid])
        draws=np.zeros(B);N=len(ids)
        for f,cs in fam.items():
            sums=np.array([sum(v) for v in cs.values()]);weights=np.array([len(v) for v in cs.values()]);idx=rng.integers(0,len(cs),(B,len(cs)))
            draws+=sums[idx].sum(1)/weights[idx].sum(1)*sum(weights)/N
        return np.quantile(draws,quantiles).tolist()
    def item_values(rows,field):
        z=defaultdict(list)
        for r in rows:z[r['id']].append(float(r[field]))
        return {sid:float(np.mean(v)) for sid,v in z.items()}
    metrics={};quality={};latency={}
    for name,rows in groups.items():
        by=defaultdict(list)
        for r in rows:by[r['id']].append(r)
        ids=sorted(by);values=item_values(rows,'correct');quality[name]=values;ls=np.array([r['latency_s'] for r in rows]);latency[name]={sid:np.array([r['latency_s'] for r in by[sid]]) for sid in ids}
        per_family={}
        for family in sorted({r['family'] for r in rows}):
            sub=[sid for sid in ids if meta[sid]['family']==family];vs={sid:values[sid] for sid in sub}
            per_family[family]={'n_items':len(sub),'accuracy':float(np.mean(list(vs.values()))),'ci95':bootstrap(vs,sub)}
        valid=item_values(rows,'schema_valid');m={'n_items':len(ids),'n_semantic_clusters':len({cluster[sid] for sid in ids}),'n_item_repeats':len(rows),'repeats_per_item':sorted({len(v) for v in by.values()}),'accuracy':float(np.mean(list(values.values()))),'accuracy_item_variance':float(np.var(list(values.values()),ddof=1)),'accuracy_ci95':bootstrap(values,ids),'schema_valid_rate':float(np.mean(list(valid.values()))),'schema_valid_ci95':bootstrap(valid,ids),'format_failure_rate':1-float(np.mean(list(valid.values()))),'p50_latency_s':float(np.quantile(ls,.5)),'p95_latency_s':float(np.quantile(ls,.95)),'mean_latency_s':float(ls.mean()),'latency_variance_s2':float(ls.var(ddof=1)),'mean_executed_layers':float(np.mean([r['executed_layers'] for r in rows])),'families':per_family,'throughput_requests_s':1/float(ls.mean()),'generated_tokens_mean':float(np.mean([r['generated_tokens'] for r in rows])) if 'generated_tokens' in rows[0] else None,'generated_tokens_variance':float(np.var([r['generated_tokens'] for r in rows],ddof=1)) if 'generated_tokens' in rows[0] else None,'readout':rows[0].get('readout','generation'),'depth':rows[0].get('depth'),'per_repeat_timing':{str(rep):{'p50_s':float(np.median([r['latency_s'] for r in rows if r['repeat']==rep])),'p95_s':float(np.quantile([r['latency_s'] for r in rows if r['repeat']==rep],.95))} for rep in sorted({r['repeat'] for r in rows})}}
        # Cluster bootstrap latency with all observed within-item repetitions.
        cs=defaultdict(list)
        for sid in ids:cs[(meta[sid]['family'],cluster[sid])].extend(latency[name][sid])
        block=[np.array(v) for v in cs.values()];draws=[]
        for _ in range(1000):
            ix=rng.integers(0,len(block),len(block));sample=np.concatenate([block[i] for i in ix]);draws.append(np.quantile(sample,[.5,.95]))
        ci=np.quantile(np.array(draws),[.025,.975],axis=0);m['p50_latency_ci95']=ci[:,0].tolist();m['p95_latency_ci95']=ci[:,1].tolist()
        if 'probabilities' in rows[0]:
            first=[v[0] for v in by.values()];probs=[r['probabilities'] for r in first];brier=[sum((prob-(label==r['gold']))**2 for label,prob in r['probabilities'].items()) for r in first]
            conf=np.array([r['confidence'] for r in first]);correct=np.array([r['correct'] for r in first],float);ece=0.
            for low in np.arange(0,1,.1):
                mask=(conf>=low)&(conf<(low+.1 if low<.9 else 1.0000001))
                if mask.any():ece+=mask.mean()*abs(conf[mask].mean()-correct[mask].mean())
            m.update(multiclass_brier=float(np.mean(brier)),ece_10_bins=float(ece),calibration_definition='project synthetic candidate distribution; not JevBench official calibration')
        # Stratify observed accuracy by fixed character-length bins and latent
        # operation-count proxy; use no outcome-driven bin selection.
        m['length_strata']={}
        for low,high in ((0,500),(500,1000),(1000,100000)):
            sub=[sid for sid in ids if low<=len(meta[sid]['prompt'])<high]
            if sub:m['length_strata'][f'{low}_{high}_chars']={'n':len(sub),'accuracy':float(np.mean([values[sid] for sid in sub]))}
        metrics[name]=m
    paired={}
    def contrast(current,reference,label=None):
        common=sorted(set(quality[current])&set(quality[reference]));diff={sid:quality[current][sid]-quality[reference][sid] for sid in common};ci=bootstrap(diff,common)
        p50cur=np.median(np.concatenate([latency[current][sid] for sid in common]));p50ref=np.median(np.concatenate([latency[reference][sid] for sid in common]));p95cur=np.quantile(np.concatenate([latency[current][sid] for sid in common]),.95);p95ref=np.quantile(np.concatenate([latency[reference][sid] for sid in common]),.95)
        blocks=defaultdict(list)
        for sid in common:blocks[(meta[sid]['family'],cluster[sid])].append(sid)
        cs=list(blocks.values());ratios=[]
        for _ in range(1000):
            ix=rng.integers(0,len(cs),len(cs));draw=[sid for i in ix for sid in cs[i]]
            x=np.concatenate([latency[current][sid] for sid in draw]);y=np.concatenate([latency[reference][sid] for sid in draw])
            ratios.append(np.quantile(x,[.5,.95])/np.quantile(y,[.5,.95]))
        lci=np.quantile(np.array(ratios),[.025,.975],axis=0)
        multiplicity=sum(name.startswith('v6/exit/') for name in metrics) if current.startswith('v6/exit/') else 1
        qs=(.025/multiplicity,1-.025/multiplicity)
        adjusted=bootstrap(diff,common,quantiles=qs) if multiplicity>1 else ci
        adjusted_lci=np.quantile(np.array(ratios),qs,axis=0)
        d={'current':current,'reference':reference,'n_common_items':len(common),'accuracy_difference_pp':100*float(np.mean(list(diff.values()))),'paired_ci95_pp':[100*x for x in ci],'multiplicity_adjusted_ci95_pp':[100*x for x in adjusted],'multiplicity':multiplicity,'p50_ratio':float(p50cur/p50ref),'p95_ratio':float(p95cur/p95ref),'p50_ratio_ci95':lci[:,0].tolist(),'p95_ratio_ci95':lci[:,1].tolist(),'adjusted_p50_ratio_ci95':adjusted_lci[:,0].tolist(),'adjusted_p95_ratio_ci95':adjusted_lci[:,1].tolist(),'quality_noninferiority_point_and_ci':bool(ci[0]>-.02),'observed_latency_gate':bool(p50cur<=.8*p50ref and p95cur<=p95ref),'formal_success':bool(adjusted[0]>-.02 and adjusted_lci[1,0]<=.8 and adjusted_lci[1,1]<=1.),'success_note':'paired quality lower > -2pp and latency upper p50<=0.8/p95<=1; Bonferroni familywise correction across tested exit grid'}
        paired[label or (current+' minus '+reference)]=d
    direct='v6/generation/direct';answer='v6/generation/matched_sft_answer_a1_seed20281028';msd='v6/generation/matched_msd_a1_seed20281028'
    contrast(msd,answer,'primary_msd1_minus_answer_only');contrast(answer,direct,'answer_only_minus_direct');contrast(msd,direct,'msd1_minus_direct')
    reference_group={'matched_sft_trace_a0_seed20281028':1,'matched_sft_trace_a0.5_seed20281028':1,'matched_sft_trace_a1_seed20281028':2,'matched_msd_a0_seed20281028':2,'matched_msd_a0.5_seed20281028':3,'program_answer_sft64_seed20261025':3}
    for name in metrics:
        if name.startswith('v6/exit/'):
            contrast(name,direct)
        elif '/generation/' in name:
            split,_,condition=name.split('/')
            if condition.startswith('direct') or condition=='explicit_cot':continue
            group=reference_group.get(condition,0)
            reference=f'{split}/generation/direct'+(f'_group{group}' if group else '')
            if split=='external' or condition in reference_group:contrast(name,reference)
    if 'v6/generation/explicit_cot' in metrics:
        for reference in (direct,answer,msd):contrast(reference,'v6/generation/explicit_cot','subset72_'+reference.split('/')[-1]+'_minus_explicit_cot')
    primary=paired['primary_msd1_minus_answer_only'];expand=primary['accuracy_difference_pp']>=3 and primary['paired_ci95_pp'][0]>=0
    training={}
    for path in sorted(out.glob('matched_*/train_log.json')):
        d=json.loads(path.read_text());pr=json.loads((path.parent/'training_provenance.json').read_text());training[path.parent.name]={**{k:v for k,v in d.items() if k not in ('steps','dev')},'dev':d['dev'],'tokens':{k:v for k,v in pr.items() if k.startswith('total_')},'train_sha256':pr['train_sha256'],'adapter_sha256':hashlib.sha256((path.parent/'adapter_model.safetensors').read_bytes()).hexdigest()}
    policies={c:json.loads((run/'configs'/f'{POLICY_PREFIX}{c}.json').read_text()) for c in json.loads((run/'configs/matched_models_frozen.json').read_text())['head_conditions']}
    report={'analysis':'frozen one-seed, update/example-matched synthetic pilot; clustered by semantic parameters, stratified quality resampling','bootstrap_seed':20291102,'quality_bootstrap_draws':10000,'latency_bootstrap_draws':1000,'conditions':metrics,'paired_comparisons':paired,'training':training,'dynamic_policies':{c:p['selected'] for c,p in policies.items()},'expansion_gate':{'passed':bool(expand),'required':'>=3pp primary MSD1 vs matched answer-only and paired lower bound >=0','observed':primary},'source_files':[x.relative_to(run).as_posix() for x in sources],'unmeasured':{'generation_intervention':'no reliable labeled trajectories; not measured','steering_memory':'not run; secondary controls deferred','general_ability_retention':'only AG News regression; broad retention not established','multi_seed_robustness':'one matched training seed'},'semantic_overlap_audit':'results/semantic_overlap_audit.json'}
    write(out/'matched_aggregate_metrics.json',report)
    with (out/'matched_aggregate_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['condition','n_items','n_semantic_clusters','n_item_repeats','accuracy','accuracy_ci_low','accuracy_ci_high','schema_valid_rate','p50_latency_s','p95_latency_s','mean_executed_layers','generated_tokens_mean','multiclass_brier','ece_10_bins'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,m in metrics.items():w.writerow({'condition':name,**{k:m.get(k) for k in fields if k not in ('condition','accuracy_ci_low','accuracy_ci_high')},'accuracy_ci_low':m['accuracy_ci95'][0],'accuracy_ci_high':m['accuracy_ci95'][1]})
    with (out/'matched_per_item_predictions.jsonl').open('w',encoding='utf-8') as f:
        for source in sources:
            for r in read(source):f.write(json.dumps({**r,'source_file':source.name},ensure_ascii=False)+'\n')
    print(json.dumps({'conditions':len(metrics),'primary':primary,'expansion_gate_passed':bool(expand),'semantic_clusters':len(test_clusters)},indent=2))
if __name__=='__main__':main()

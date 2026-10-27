#!/usr/bin/env python3
"""CPU-only post-training data audit. Never reads v6 or changes training."""
from pathlib import Path
from collections import Counter
import argparse,copy,hashlib,json
from generate_tasks import solve,render
from analyze_matched import semantic_key

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    rows={s:[json.loads(x) for x in (r/'data'/f'{s}.jsonl').read_text(encoding='utf-8').splitlines() if x] for s in ('train','dev','pilot_holdout')}
    rows['teacher_screened_train']=[json.loads(x) for x in (r/'data/train_teacher_verified.jsonl').read_text(encoding='utf-8').splitlines() if x]
    rows['adapter_exposed_train']=rows['teacher_screened_train'][:256]
    clusters={s:Counter(semantic_key(x) for x in rs) for s,rs in rows.items()}
    overlap={f'{x}__{y}':{'shared_semantic_clusters':len(set(clusters[x])&set(clusters[y])),'right_items_seen_in_left':sum(semantic_key(z) in clusters[x] for z in rows[y])} for x,y in [('train','dev'),('train','pilot_holdout'),('dev','pilot_holdout'),('teacher_screened_train','dev'),('adapter_exposed_train','dev'),('teacher_screened_train','pilot_holdout'),('adapter_exposed_train','pilot_holdout')]}
    counts=Counter();failures=[]
    for row in rows['train']:
        family=row['family']
        if counts[family]>=30:continue
        u=copy.deepcopy(row['parameters']);v=copy.deepcopy(u)
        if family=='rules':
            rule=next(x for x in u['rules'] if x['name']==u['required_rule']);u['facts'].update(rule['if']);u['exception']=None;v=copy.deepcopy(u);k=next(iter(v['facts']));v['exception']={'if':{k:v['facts'][k]}}
        elif family=='routing':
            u['state'].update(u['routes'][0]['requires']);u['load']=0;v=copy.deepcopy(u);v['load']=max(x['capacity'] for x in v['routes'])+1
        elif family=='time_numeric':
            value=u['base_minutes']+u['elapsed_minutes']*u['multiplier']+u['offset_minutes'];u['limit_minutes']=value;v=copy.deepcopy(u);v['limit_minutes']=value-1
        elif family=='multihop':
            u['edges']=[[u['source'],u['target']]];v=copy.deepcopy(u);v['edges']=[]
        elif family=='evidence':
            u['contradictory_evidence']=['COUNTER_EVIDENCE'];u['evidence']=list(u['required_evidence']);v=copy.deepcopy(u);v['evidence'].append('COUNTER_EVIDENCE')
        elif family=='intent':
            alt=next(k for k in u['mapping'] if u['mapping'][k]!=u['mapping'][u['intent']]);v['mapping'][u['intent']],v['mapping'][alt]=v['mapping'][alt],v['mapping'][u['intent']]
        first,_=solve(family,u);second,_=solve(family,v);pu,lu=render(family,u,'template_a');pv,lv=render(family,v,'template_a')
        ok=first!=second and pu!=pv and set(lu)==set(lv) and first in lu and second in lv
        if not ok:failures.append({'family':family,'source_id':row['id'],'answer_changed':first!=second})
        counts[family]+=1
    out={'scope':'train/dev/historical pilot data audit only; no v6 prompts/results accessed','generated_before_training':False,'post_training_diagnostic_not_selection':True,'semantic_rule':'remove random case_ref; canonicalize edge/evidence order; remaining parameters exact, not full graph isomorphism','unique_semantic_counts':{s:len(x) for s,x in clusters.items()},'pairwise_overlap':overlap,'counterfactual_pairs_checked':dict(counts),'counterfactual_failures':failures,'first_audit_correction':'Some intent mappings are many-to-one after label forcing, so swapping arbitrary entries need not change the correct label. Preserved first attempt; corrected only the counterfactual test to swap distinct mapping values. Generator/data/models unchanged.','interpretation':'Symbolic solver/render sensitivity only; no model shortcut robustness was measured. Exact semantic overlap and generator simplifications limit independence/generalization claims. Cluster bootstrap does not remove train/dev overlap bias.'}
    (r/'results/training_semantic_counterfactual_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert not failures
if __name__=='__main__':main()

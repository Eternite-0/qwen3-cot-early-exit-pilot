#!/usr/bin/env python3
"""CPU-only clustered dev design calculation; does not select policies."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,math,numpy as np
from analyze_matched import read,semantic_key

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    meta={x['id']:x for x in read(r/'data/dev.jsonl')};config=json.loads((r/'configs/matched_models_frozen.json').read_text());records=[]
    for condition in config['head_conditions']:
        rs=read(r/'results'/f'matched_head_dev_unmerged_{condition}.jsonl');N=len(rs)
        for depth in (2,4,8,12,16,20,24):
            differences={x['id']:float(x['depths'][str(depth)]['correct'])-float(x['depths']['28']['correct']) for x in rs};fam=defaultdict(list)
            for sid,v in differences.items():fam[meta[sid]['family']].append((sid,v))
            variance_mean=0.;iid_variance_mean=0.
            for family,values in fam.items():
                n=len(values);mean=np.mean([v for _,v in values]);scores=defaultdict(float)
                for sid,v in values:scores[semantic_key(meta[sid])]+=v-mean
                G=len(scores);weight=n/N
                variance_mean+=weight**2*(G/(G-1))*sum(z*z for z in scores.values())/(n*n)
                iid_variance_mean+=weight**2*np.var([v for _,v in values],ddof=1)/n
            diff=float(np.mean(list(differences.values())));se=math.sqrt(variance_mean);design_effect=variance_mean/iid_variance_mean if iid_variance_mean else None
            needed=math.ceil((1.959963984540054+.8416212335729143)**2*variance_mean*N/(.02**2))
            records.append({'condition':condition,'depth':depth,'reference_depth':28,'n_dev':N,'semantic_clusters':len({semantic_key(x) for x in meta.values()}),'observed_difference_pp':100*diff,'cluster_stratified_se_pp':100*se,'normal_approx_ci_low_pp':100*(diff-1.96*se),'normal_approx_ci_high_pp':100*(diff+1.96*se),'design_effect_vs_stratified_iid':design_effect,'approx_n_for_80pct_power_true_delta0_margin2pp':needed,'observed_point_below_minus2pp':diff<-.02})
    report={'scope':'post-registration dev-only design diagnostic, no v6 data/results; no threshold/model changes','method':'family-stratified cluster sandwich variance; exact parameter clusters after case_ref removal; normal approximation','assumed_true_difference_pp':0,'noninferiority_margin_pp':2,'one_sided_alpha':.025,'power':.8,'limitations':'Assumes same family mix and stable variance/design effect as dev. Does not remove train/dev overlap. A systematic loss below margin cannot be repaired by increasing N. This is a planning approximation, not a noninferiority test.','records':records}
    out=r/'results/development_cluster_sample_size.json';out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with out.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps({'depth8_designs':[x for x in records if x['depth']==8]},indent=2))
if __name__=='__main__':main()

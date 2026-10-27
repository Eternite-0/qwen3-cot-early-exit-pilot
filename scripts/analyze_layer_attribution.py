#!/usr/bin/env python3
"""Supplementary controlled contrasts from complete saved fixed-exit records."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,numpy as np
from analyze_matched import read,semantic_key

BACKBONES=['direct','matched_sft_answer_a1_seed20281028','matched_msd_a1_seed20281028']
DEPTHS=[2,4,8,12,16,20,24,28]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
    assert (r/'results/matched_aggregate_metrics.json').exists()
    rows=read(r/'results/matched_v6_exit_predictions.jsonl');meta={x['id']:x for x in read(r/'data/synthetic_test_v6_unopened.jsonl')};groups=defaultdict(lambda:defaultdict(list));rng=np.random.default_rng(20291104)
    for row in rows:groups[row['condition']][row['id']].append(row)
    cases=[]
    for b in BACKBONES:
        for kind in ('native','trained'):
            for depth in DEPTHS[:-1]:cases.append((f'{b}/{kind}/L{depth}',f'{b}/{kind}/L28','layer_only'))
    for kind in ('native','trained'):
        for depth in DEPTHS:cases.append((f'{BACKBONES[2]}/{kind}/L{depth}',f'{BACKBONES[1]}/{kind}/L{depth}','msd_vs_answer_same_readout_depth'))
    contrasts=[]
    for current,reference,purpose in cases:
        ids=sorted(set(groups[current])&set(groups[reference]));blocks=defaultdict(lambda:defaultdict(list))
        for sid in ids:blocks[meta[sid]['family']][semantic_key(meta[sid])].append(sid)
        diff={sid:np.mean([x['correct'] for x in groups[current][sid]])-np.mean([x['correct'] for x in groups[reference][sid]]) for sid in ids};draws=np.zeros(10000)
        for fam,cs in blocks.items():
            sums=np.array([sum(diff[sid] for sid in sids) for sids in cs.values()]);weights=np.array([len(sids) for sids in cs.values()]);ix=rng.integers(0,len(cs),(10000,len(cs)))
            draws+=sums[ix].sum(1)/weights[ix].sum(1)*weights.sum()/len(ids)
        def timing(name,sids):return np.array([x['latency_s'] for sid in sids for x in groups[name][sid]])
        x=timing(current,ids);y=timing(reference,ids);ratio=np.quantile(x,[.5,.95])/np.quantile(y,[.5,.95]);ldraws=[]
        for _ in range(1000):
            selected=[]
            for fam,cs in blocks.items():
                clusters=list(cs.values());ix=rng.integers(0,len(clusters),len(clusters));selected.extend(sid for j in ix for sid in clusters[j])
            ldraws.append(np.quantile(timing(current,selected),[.5,.95])/np.quantile(timing(reference,selected),[.5,.95]))
        ci=np.quantile(draws,[.025,.975]);lci=np.quantile(np.array(ldraws),[.025,.975],axis=0)
        contrasts.append({'current':current,'reference':reference,'purpose':purpose,'n_items':len(ids),'n_semantic_clusters':sum(map(len,blocks.values())),'accuracy_difference_pp':100*float(np.mean(list(diff.values()))),'paired_ci95_pp':(100*ci).tolist(),'p50_ratio':float(ratio[0]),'p95_ratio':float(ratio[1]),'p50_ratio_ci95':lci[:,0].tolist(),'p95_ratio_ci95':lci[:,1].tolist()})
    report={'status':'supplementary_fixed_contrasts_not_policy_selection','seed':20291104,'quality_draws':10000,'latency_draws':1000,'method':'paired family-stratified exact-parameter cluster resampling, retaining all five observed timing repeats','inference_or_primary_analysis_changed':False,'interpretation':'Layer-only contrasts share backbone and readout; method contrasts share depth and readout. Marginal95 intervals, exploratory across58 contrasts; do not apply as new formal deployment or expansion gate. Original direct-generation-reference analysis remains unchanged.','contrasts':contrasts}
    (r/'results/layer_attribution_metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (r/'results/layer_attribution_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(contrasts[0]));w.writeheader();w.writerows(contrasts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained',sharey=True)
    for ax,kind in zip(axs,('native','trained')):
        for b,label,color,marker in zip(BACKBONES,['Original','Answer-only SFT','MSD alpha=1'],['#0072B2','#D55E00','#009E73'],['o','s','^']):
            zs=[next(z for z in contrasts if z['current']==f'{b}/{kind}/L{d}' and z['purpose']=='layer_only') for d in DEPTHS[:-1]]
            ys=np.array([z['accuracy_difference_pp'] for z in zs]+[0]);cis=np.array([z['paired_ci95_pp'] for z in zs]+[[0,0]])
            ax.errorbar(DEPTHS,ys,yerr=np.maximum(0,np.vstack([ys-cis[:,0],cis[:,1]-ys])),color=color,marker=marker,ms=3,capsize=2,lw=1,elinewidth=.6,label=label)
        ax.axhline(-2,color='#666666',ls='--',lw=.8);ax.axhline(0,color='#666666',lw=.6);ax.set_xticks(DEPTHS);ax.set_xlabel('Executed decoder blocks');ax.set_title('Native LM readout' if kind=='native' else 'Trained readout');ax.grid(alpha=.18,lw=.45)
    axs[0].set_ylabel('Paired quality difference vs own L28 (pp)');fig.suptitle('Layer contribution with backbone/readout held fixed; marginal95% intervals',fontsize=8);fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    for ext in ('svg','pdf','png'):fig.savefig(r/'figures'/f'layer_only_paired_difference.{ext}',dpi=300 if ext=='png' else None)
    plt.close(fig)
    p=r/'figures/matched_figure_manifest.json';fm=json.loads(p.read_text(encoding='utf-8'));fm['figures'].append('layer_only_paired_difference');fm['additional_source']='results/layer_attribution_metrics.json';p.write_text(json.dumps(fm,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'supplementary_fixed_contrasts':len(contrasts),'figure':'layer_only_paired_difference'}))
if __name__=='__main__':main()

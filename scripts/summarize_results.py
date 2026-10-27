#!/usr/bin/env python3
"""Aggregate saved predictions and render publication-ready, editable figures."""
from __future__ import annotations
import csv, json, random, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / 'results'
FIG = RUN / 'figures'
SEED = 20261025
B = 10000

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def ci_cluster(values_by_id, rng, n_boot=B):
    vals=np.asarray(list(values_by_id.values()),dtype=float)
    sampled=rng.integers(0,len(vals),size=(n_boot,len(vals)))
    draws=vals[sampled].mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]

def paired_diff_ci(left, right, rng, n_boot=B):
    ids = sorted(set(left) & set(right))
    diffs = np.array([left[i] - right[i] for i in ids], dtype=float)
    sampled=rng.integers(0,len(diffs),size=(n_boot,len(diffs)))
    draws=diffs[sampled].mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]

def grouped_predictions(rows, condition=None, depth=None):
    out = defaultdict(list)
    for r in rows:
        if condition is not None and r.get('condition') != condition: continue
        if depth is not None and r.get('depth') != depth: continue
        out[r['id']].append(r)
    return out

def acc_map(items):
    return {i: float(np.mean([bool(x['correct']) for x in xs])) for i, xs in items.items()}

def latency_ci(items, rng, stat='p50', n_boot=3000):
    ids = list(items)
    matrix=np.asarray([[x['latency_s'] for x in items[i]] for i in ids],dtype=float)
    draws=[]
    for start in range(0,n_boot,100):
        count=min(100,n_boot-start)
        sampled=rng.integers(0,len(ids),size=(count,len(ids)))
        values=matrix[sampled].reshape(count,-1)
        draws.extend(np.quantile(values,.5 if stat=='p50' else .95,axis=1).tolist())
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]

def schema_valid(raw, labels):
    """Accept exactly one allowed label plus an optional terminal EOS marker."""
    if not isinstance(raw, str):
        return False
    value=raw.strip()
    for eos in ('<|im_end|>','<|endoftext|>'):
        if value.endswith(eos):
            value=value[:-len(eos)].strip()
            break
    if '<|im_end|>' in value or '<|endoftext|>' in value:
        return False
    return value in labels

def main():
    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    flat = []
    results = {'run_id': RUN.name, 'synthetic_only': True, 'primary_synthetic_check': 'synthetic_balanced_template_transfer_check_v3', 'bootstrap_replicates': B, 'conditions': {}, 'paired_comparisons': {}, 'not_run': {}}
    plot_rows = []

    gen_v3 = OUT / 'synthetic_test_v3_direct_sft64_predictions.jsonl'
    read_v3 = OUT / 'synthetic_test_v3_readout_valid_predictions.jsonl'
    gen_v2 = OUT / 'synthetic_test_v2_direct_sft64_predictions.jsonl'
    read_v2 = OUT / 'synthetic_test_v2_readout_valid_predictions.jsonl'
    gen_path = gen_v3 if gen_v3.exists() else gen_v2
    read_path = read_v3 if read_v3.exists() else read_v2
    gen_rows = read_jsonl(gen_v3) if gen_v3.exists() else (read_jsonl(gen_v2) if gen_v2.exists() else [])
    read_rows = read_jsonl(read_v3) if read_v3.exists() else (read_jsonl(read_v2) if read_v2.exists() else [])
    split_label = 'synthetic_balanced_template_transfer_check_v3' if gen_v3.exists() else 'synthetic_template_transfer_test_v2'
    data_file = RUN / 'data' / ('synthetic_test_v3_balanced.jsonl' if gen_v3.exists() else 'synthetic_test_v2.jsonl')
    candidate_labels = {x['id']:x.get('labels',[]) for x in read_jsonl(data_file)} if data_file.exists() else {}
    flat.extend(gen_rows)
    flat.extend(read_rows)
    teacher_path = OUT / 'teacher_dev.jsonl'
    teacher_rows = read_jsonl(teacher_path) if teacher_path.exists() else []
    # Normalize teacher dev records to the same per-item record in final output.
    for r in teacher_rows:
        for mode, key in [('teacher_direct', 'direct_correct'), ('teacher_explicit_cot', 'cot_correct')]:
            flat.append({'id':r['id'], 'family':r['family'], 'gold':r['gold'], 'prediction':r.get('direct' if key=='direct_correct' else 'cot'), 'correct':bool(r.get(key,False)), 'condition':mode, 'split':'dev'})

    if gen_rows:
        for cond in ('direct','sft_answer'):
            rows = [x for x in gen_rows if x['condition'] == cond]
            grouped = grouped_predictions(gen_rows, condition=cond)
            by_id = acc_map(grouped)
            lat_by_id = defaultdict(list)
            for x in rows: lat_by_id[x['id']].append(x)
            lat = [x['latency_s'] for x in rows]
            tokens = [x['generated_tokens'] for x in rows]
            fam = {}
            for name in sorted({x['family'] for x in rows}):
                fam_rows = [x for x in rows if x['family'] == name]
                fam[name] = {'n_items': len({x['id'] for x in fam_rows}), 'n_generations': len(fam_rows), 'accuracy': float(np.mean([x['correct'] for x in fam_rows]))}
            valid_by_id={}
            for x in rows:
                valid_by_id.setdefault(x['id'],[]).append(float(schema_valid(x.get('prediction_raw'),candidate_labels.get(x['id'],[]))))
            valid_item_means={i:float(np.mean(v)) for i,v in valid_by_id.items()}
            valid_rate=float(np.mean([v for values in valid_by_id.values() for v in values]))
            valid_ci=ci_cluster(valid_item_means,rng)
            results['conditions'][cond] = {'split':split_label, 'n_items':len(by_id), 'n_item_repeats':len(rows), 'n_correct_item_equivalents':float(sum(by_id.values())), 'accuracy':float(np.mean(list(by_id.values()))), 'accuracy_item_bootstrap_95':ci_cluster(by_id,rng), 'schema_valid_generations':int(sum(sum(v) for v in valid_by_id.values())), 'schema_valid_rate':valid_rate, 'schema_valid_item_cluster_bootstrap_95':valid_ci, 'format_failure_rate':float(1-valid_rate), 'mean_generated_tokens':float(np.mean(tokens)), 'p50_latency_s':float(np.quantile(lat,.5)), 'p50_latency_cluster_bootstrap_95':latency_ci(lat_by_id,rng,'p50'), 'p95_latency_s':float(np.quantile(lat,.95)), 'p95_latency_cluster_bootstrap_95':latency_ci(lat_by_id,rng,'p95'), 'family':fam}
            plot_rows.append({'name':cond.replace('_',' '), 'accuracy':results['conditions'][cond]['accuracy'], 'accuracy_ci':results['conditions'][cond]['accuracy_item_bootstrap_95'], 'p50':results['conditions'][cond]['p50_latency_s'], 'p50_ci':results['conditions'][cond]['p50_latency_cluster_bootstrap_95'], 'kind':'generation'})
        direct = acc_map(grouped_predictions(gen_rows,condition='direct'))
        sft = acc_map(grouped_predictions(gen_rows,condition='sft_answer'))
        results['paired_comparisons']['sft_answer_minus_direct'] = {'difference_pp':100*(np.mean(list(sft.values()))-np.mean(list(direct.values()))), 'paired_item_cluster_bootstrap_95_pp':[100*x for x in paired_diff_ci(sft,direct,rng)], 'n_items':len(set(sft)&set(direct))}

    if read_rows:
        for depth in (2,4,8,12,16,20,24,28):
            rows = [x for x in read_rows if x.get('condition') == f'fixed_{depth}']
            grouped = grouped_predictions(rows)
            # Unique deterministic prediction, plus per-item repeated timings.
            by_id = {i:float(xs[0]['correct']) for i,xs in grouped.items()}
            lat_by_id = defaultdict(list)
            for x in rows: lat_by_id[x['id']].append(x)
            lat = [x['latency_s'] for x in rows]
            fam = {}
            for name in sorted({x['family'] for x in rows}):
                vals = [x for x in rows if x['family']==name and x['repeat']==0]
                fam[name]={'n':len(vals),'accuracy':float(np.mean([x['correct'] for x in vals]))}
            key=f'readout_depth_{depth}'
            results['conditions'][key]={'split':split_label,'n_items':len(by_id),'n_latency_repeats':len(rows),'accuracy':float(np.mean(list(by_id.values()))),'accuracy_item_bootstrap_95':ci_cluster(by_id,rng),'p50_latency_s':float(np.quantile(lat,.5)),'p50_latency_cluster_bootstrap_95':latency_ci(lat_by_id,rng,'p50'),'p95_latency_s':float(np.quantile(lat,.95)),'p95_latency_cluster_bootstrap_95':latency_ci(lat_by_id,rng,'p95'),'mean_executed_layers':depth,'family':fam}
            plot_rows.append({'name':f'{depth}-layer head','accuracy':results['conditions'][key]['accuracy'],'accuracy_ci':results['conditions'][key]['accuracy_item_bootstrap_95'],'p50':results['conditions'][key]['p50_latency_s'],'p50_ci':results['conditions'][key]['p50_latency_cluster_bootstrap_95'],'kind':'readout','depth':depth})
        base=acc_map(grouped_predictions([x for x in read_rows if x.get('condition')=='fixed_28']))
        for depth in (2,4,8,12,16,20,24):
            cur=acc_map(grouped_predictions([x for x in read_rows if x.get('condition')==f'fixed_{depth}']))
            results['paired_comparisons'][f'fixed_{depth}_minus_fixed_28']={'difference_pp':100*(np.mean(list(cur.values()))-np.mean(list(base.values()))),'paired_item_cluster_bootstrap_95_pp':[100*x for x in paired_diff_ci(cur,base,rng)],'n_items':len(set(cur)&set(base))}

    # Same checked fixed-depth prediction probabilities permit proper multiclass
    # Brier/ECE only for this candidate-conditioned choice head.
    calibration = {}
    for depth in (2,4,8,12,16,20,24,28):
        rows=[x for x in read_rows if x.get('condition')==f'fixed_{depth}' and x.get('repeat')==0]
        if not rows: continue
        brier=[]; confs=[]; correct=[]
        for x in rows:
            p=x.get('probabilities',{}); labels=list(p)
            if not labels: continue
            brier.append(sum((float(p[k])-(1.0 if k==x['gold'] else 0.0))**2 for k in labels))
            confs.append(float(x.get('confidence',max(p.values()))));correct.append(float(x['correct']))
        bins=np.linspace(0,1,11);ece=0.0
        for lo,hi in zip(bins[:-1],bins[1:]):
            idx=[i for i,c in enumerate(confs) if lo<=c<(hi if hi<1 else hi+1e-9)]
            if idx:ece+=len(idx)/len(confs)*abs(float(np.mean([confs[i] for i in idx]))-float(np.mean([correct[i] for i in idx])))
        calibration[str(depth)]={'multiclass_brier_mean':float(np.mean(brier)),'top_label_ece_10bin':float(ece),'n_items':len(brier)}
    if calibration: results['readout_calibration'] = calibration

    if teacher_rows:
        for mode,key in [('teacher_direct','direct_correct'),('teacher_explicit_cot','cot_correct')]:
            values={r['id']:float(bool(r.get(key,False))) for r in teacher_rows}
            results['conditions'][mode]={'split':'dev','n_items':len(values),'accuracy':float(np.mean(list(values.values()))),'accuracy_item_bootstrap_95':ci_cluster(values,rng)}
    results['not_run']={'paper_masked_self_distillation':'Not executed: teacher quality gate failed (25/300 explicit-CoT traces passed); this is not an MSD method result.', 'teacher_trace_sft':'Not executed: no teacher traces qualified; matched-data teacher-trace SFT not run.', 'dynamic_stopper':'Not evaluated on the synthetic template-transfer test: no dev threshold met both <=2 pp accuracy loss and >=20% fewer executed layers.', 'general_capability_regression':'Not measured: no independent capability suite run.', 'generation_intervention':'Not measured: no reliable labeled generation trajectories.'}
    results['prediction_sources']={'generation_predictions':gen_path.name if gen_rows else None,'readout_predictions':read_path.name if read_rows else None,'combined_export':'results/per_item_predictions.jsonl'}
    # The final long-form file contains only real observed predictions and records
    # source IDs so the v2 exploratory rows can never be mistaken for v3.
    if gen_rows or read_rows:
        with (OUT/'per_item_predictions.jsonl').open('w',encoding='utf-8') as f:
            for row in gen_rows:
                x={**row,'source_file':gen_path.name,'analysis_split':split_label}
                f.write(json.dumps(x,ensure_ascii=False)+'\n')
            for row in read_rows:
                x={**row,'source_file':read_path.name,'analysis_split':split_label}
                f.write(json.dumps(x,ensure_ascii=False)+'\n')
    (OUT/'aggregate_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    # Long-form aggregate CSV with actual observed conditions only.
    with (OUT/'aggregate_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['condition','split','n_items','n_repeats','accuracy','accuracy_ci_low','accuracy_ci_high','schema_valid_rate','schema_valid_ci_low','schema_valid_ci_high','format_failure_rate','p50_latency_s','p50_latency_ci_low','p50_latency_ci_high','p95_latency_s','mean_tokens','mean_layers'])
        w.writeheader()
        for name,m in results['conditions'].items():
            w.writerow({'condition':name,'split':m.get('split'),'n_items':m.get('n_items'),'n_repeats':m.get('n_item_repeats',m.get('n_latency_repeats')),'accuracy':m.get('accuracy'),'accuracy_ci_low':(m.get('accuracy_item_bootstrap_95') or [None,None])[0],'accuracy_ci_high':(m.get('accuracy_item_bootstrap_95') or [None,None])[1],'schema_valid_rate':m.get('schema_valid_rate'),'schema_valid_ci_low':(m.get('schema_valid_item_cluster_bootstrap_95') or [None,None])[0],'schema_valid_ci_high':(m.get('schema_valid_item_cluster_bootstrap_95') or [None,None])[1],'format_failure_rate':m.get('format_failure_rate'),'p50_latency_s':m.get('p50_latency_s'),'p50_latency_ci_low':(m.get('p50_latency_cluster_bootstrap_95') or [None,None])[0],'p50_latency_ci_high':(m.get('p50_latency_cluster_bootstrap_95') or [None,None])[1],'p95_latency_s':m.get('p95_latency_s'),'mean_tokens':m.get('mean_generated_tokens'),'mean_layers':m.get('mean_executed_layers')})

    # Plot defaults: vector text, 8 pt labels, restrained ink.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65,'lines.linewidth':1.2})
    colors={'direct':'#3274A1','sft answer':'#E1812C','readout':'#3A923A'}
    markers={'generation':'o','readout':'^'}
    if plot_rows:
        fig,ax=plt.subplots(figsize=(7.1,3.7),constrained_layout=True)
        label_offsets={
            'direct':(8,9), 'sft answer':(8,9),
            '2-layer head':(-3,-18), '4-layer head':(14,-19),
            '8-layer head':(-16,9), '12-layer head':(-17,24),
            '16-layer head':(-15,-20), '20-layer head':(13,-8),
            '24-layer head':(11,11), '28-layer head':(10,-17),
        }
        for p in plot_rows:
            c=colors.get(p['name'].split(' head')[0],colors.get(p['kind'],'#777777'))
            ylo=p['accuracy']-p['accuracy_ci'][0];yhi=p['accuracy_ci'][1]-p['accuracy']
            xlo=max(0,p['p50']-p['p50_ci'][0]);xhi=max(0,p['p50_ci'][1]-p['p50'])
            ax.errorbar(p['p50']*1000,p['accuracy'],xerr=[[xlo*1000],[xhi*1000]],yerr=[[ylo],[yhi]],fmt=markers[p['kind']],ms=4,capsize=2,color=c,ecolor=c,alpha=.9)
            label=(f"L{p['depth']}" if p['kind']=='readout' else p['name'])
            ax.annotate(label,(p['p50']*1000,p['accuracy']),xytext=label_offsets.get(p['name'],(5,5)),textcoords='offset points',fontsize=6.5,
                        bbox={'boxstyle':'round,pad=0.12','facecolor':'white','edgecolor':'none','alpha':.8})
        frontier=[p for p in plot_rows if not any(
            (q['p50'] <= p['p50'] and q['accuracy'] >= p['accuracy']) and
            (q['p50'] < p['p50'] or q['accuracy'] > p['accuracy'])
            for q in plot_rows if q is not p)]
        frontier.sort(key=lambda p:p['p50'])
        ax.plot([p['p50']*1000 for p in frontier],[p['accuracy'] for p in frontier],
                linestyle='--',linewidth=.75,color='#333333',alpha=.8)
        ax.set(xlabel='Warm serial p50 latency (ms)',ylabel='Accuracy',ylim=(0.25,1.0));ax.grid(axis='both',alpha=.2,linewidth=.5);ax.set_title('Balanced synthetic-generator check v3')
        ax.text(.01,.02,'Triangles: trained heads; dashed line: observed Pareto frontier; L = executed decoder blocks',transform=ax.transAxes,ha='left',va='bottom',fontsize=6.5,color='#555555')
        savefig(fig,'accuracy_latency_pareto')

    if read_rows:
        fams=sorted({x['family'] for x in read_rows if x.get('condition','').startswith('fixed_')})
        fig,axs=plt.subplots(2,3,figsize=(7.1,4.5),sharex=True,sharey=True,constrained_layout=True)
        for ax,fam in zip(axs.ravel(),fams):
            xx=[];yy=[];lo=[];hi=[]
            for d in (2,4,8,12,16,20,24,28):
                items={x['id']:float(x['correct']) for x in read_rows if x.get('condition')==f'fixed_{d}' and x['family']==fam and x.get('repeat')==0}
                if not items:continue
                ci=ci_cluster(items,rng,5000);acc=float(np.mean(list(items.values())))
                xx.append(d);yy.append(acc);lo.append(acc-ci[0]);hi.append(ci[1]-acc)
            ax.errorbar(xx,yy,yerr=[lo,hi],marker='o',ms=3,capsize=2,color='#3274A1')
            ax.set_title(fam.replace('_',' '));ax.grid(alpha=.2,linewidth=.5);ax.set_xticks([2,4,8,12,16,20,24,28])
        for ax in axs[-1,:]:ax.set_xlabel('Executed decoder blocks')
        for ax in axs[:,0]:ax.set_ylabel('Accuracy')
        fig.suptitle('Trained candidate-conditioned readout by layer depth and task family')
        savefig(fig,'layer_depth_results')

    if gen_rows:
        fig,axs=plt.subplots(1,3,figsize=(7.1,2.8),constrained_layout=True)
        names=['direct','sft_answer'];labels=['Direct\n(no-think)','Program-label\nanswer-only SFT\n(64 steps)']
        acc=[];cis=[];p50=[];tokens=[]
        for n in names:
            m=results['conditions'][n];acc.append(m['accuracy']);cis.append(m['accuracy_item_bootstrap_95']);p50.append(m['p50_latency_s']*1000);tokens.append(m['mean_generated_tokens'])
        for ax,vals,axislabel in zip(axs,[acc,p50,tokens],['Accuracy','Warm serial p50 latency (ms)','Mean generated tokens']):
            ax.bar(range(2),vals,color=['#3274A1','#E1812C'],width=.6)
            if axislabel=='Accuracy':
                ax.errorbar(range(2),vals,yerr=[[a-c[0] for a,c in zip(acc,cis)],[c[1]-a for a,c in zip(acc,cis)]],fmt='none',ecolor='black',capsize=3,lw=.8)
                ax.set_ylim(0,1)
            ax.set_ylabel(axislabel);ax.set_xticks(range(2),labels);ax.tick_params(axis='x',labelsize=6.2);ax.grid(axis='y',alpha=.2,linewidth=.5)
        axs[1].set_ylim(0,max(p50)*1.2);axs[2].set_ylim(0,max(tokens)*1.25)
        axs[2].set_yticks([0,1,2,3,4])
        fig.suptitle('Available controls: balanced synthetic-generator check v3',fontsize=8)
        fig.text(.5,-.05,'No teacher-trace SFT or masked self-distillation was run.',ha='center',fontsize=7)
        savefig(fig,'cot_internalization_ablation')

    if gen_rows and read_rows:
        fig,axs=plt.subplots(1,2,figsize=(7.1,3.0),constrained_layout=True)
        gen_names=['direct','sft_answer'];gen_labels=['Direct','Answer-only\nSFT (64 steps)']
        gen_acc=[results['conditions'][k]['accuracy'] for k in gen_names]
        gen_ci=[results['conditions'][k]['accuracy_item_bootstrap_95'] for k in gen_names]
        axs[0].bar(range(2),gen_acc,color=['#3274A1','#E1812C'],width=.62)
        axs[0].errorbar(range(2),gen_acc,yerr=[[a-c[0] for a,c in zip(gen_acc,gen_ci)],[c[1]-a for a,c in zip(gen_acc,gen_ci)]],fmt='none',ecolor='black',capsize=3,lw=.8)
        axs[0].set_xticks(range(2),gen_labels);axs[0].set_ylabel('Accuracy');axs[0].set_ylim(0,1);axs[0].grid(axis='y',alpha=.2,linewidth=.5)
        depths=[2,4,8,12,16,20,24,28]
        ref=results['conditions']['readout_depth_28']['accuracy']
        ref_ci=results['conditions']['readout_depth_28']['accuracy_item_bootstrap_95']
        diffs=[];cis=[]
        for d in depths:
            key=f'fixed_{d}_minus_fixed_28'
            v=results['paired_comparisons'].get(key)
            if v is None: diffs.append(0.0);cis.append([0.0,0.0])
            else: diffs.append(v['difference_pp']/100);cis.append([x/100 for x in v['paired_item_cluster_bootstrap_95_pp']])
        axs[1].axhline(0,color='#555555',linewidth=.8)
        axs[1].axhline(-.02,color='#B44B4B',linewidth=.9,linestyle='--',label='−2 pp margin')
        axs[1].errorbar(depths,diffs,yerr=[[a-c[0] for a,c in zip(diffs,cis)],[c[1]-a for a,c in zip(diffs,cis)]],fmt='o',capsize=2,color='#3A923A',ecolor='#3A923A',ms=4)
        axs[1].set_xticks(depths);axs[1].set_xlabel('Readout depth (blocks)');axs[1].set_ylabel('Paired accuracy difference vs 28 blocks')
        axs[1].grid(axis='y',alpha=.2,linewidth=.5);axs[1].legend(frameon=False,fontsize=6.5,loc='lower right')
        axs[1].set_title(f'28-block reference: {ref:.1%} [{ref_ci[0]:.1%}, {ref_ci[1]:.1%}]',fontsize=7)
        fig.suptitle('Measured quality contrasts; no dynamic stopper selected',fontsize=8)
        savefig(fig,'paired_quality_difference')

    # Preserve the unbalanced v2 exploration in an explicitly separate summary
    # for transparency; never pool it with the corrected balanced v3 check.
    if gen_v3.exists() and read_v3.exists() and gen_v2.exists() and read_v2.exists():
        v2_gen=read_jsonl(gen_v2);v2_read=read_jsonl(read_v2)
        v2={'label':'archived exploratory v2; answer labels were imbalanced within family','n_items':len({x['id'] for x in v2_gen}), 'conditions':{},'readout':{}}
        for cond in ('direct','sft_answer'):
            xs=[x for x in v2_gen if x['condition']==cond]
            v2['conditions'][cond]={'accuracy':float(np.mean([x['correct'] for x in xs])),'mean_generated_tokens':float(np.mean([x['generated_tokens'] for x in xs])),'p50_latency_s':float(np.quantile([x['latency_s'] for x in xs],.5)),'p95_latency_s':float(np.quantile([x['latency_s'] for x in xs],.95))}
        for d in (2,4,8,12,16,20,24,28):
            xs=[x for x in v2_read if x['condition']==f'fixed_{d}' and x['repeat']==0]
            v2['readout'][str(d)]={'accuracy':float(np.mean([x['correct'] for x in xs])),'n_items':len(xs)}
        (OUT/'v2_exploratory_metrics.json').write_text(json.dumps(v2,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    print(json.dumps({'metrics':str(OUT/'aggregate_metrics.json'),'figure_files':[str(x) for x in sorted(FIG.glob('*'))]},ensure_ascii=False,indent=2))

def savefig(fig, stem):
    fig.savefig(FIG/f'{stem}.svg',format='svg',bbox_inches='tight')
    fig.savefig(FIG/f'{stem}.pdf',format='pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{stem}.png',format='png',dpi=300,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Editable scientific figures from saved final metrics and dev policies only."""
from pathlib import Path
import argparse,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';PURPLE='#7B4DA1';GRAY='#666666'
BACKBONES=['direct','matched_sft_answer_a1_seed20281028','matched_msd_a1_seed20281028']
NAMES=['Original','Answer-only SFT','MSD alpha=1'];COLORS=[BLUE,ORANGE,GREEN];MARKERS=['o','s','^']
def error(ax,x,y,ci,**kwargs):
    y=np.array(y);ci=np.array(ci);ax.errorbar(x,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),capsize=2,ms=3.8,elinewidth=.6,**kwargs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;figdir=r/'figures';figdir.mkdir(exist_ok=True)
    report=json.loads((r/'results/matched_aggregate_metrics.json').read_text());m=report['conditions'];paired=report['paired_comparisons'];created=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.65,'lines.linewidth':1.1,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,stem):
        for ext in ('svg','pdf','png'):fig.savefig(figdir/f'{stem}.{ext}',dpi=300 if ext=='png' else None)
        created.append(stem);plt.close(fig)
    def metric(c):return m['v6/generation/'+c]
    def clean(ax):ax.grid(alpha=.18,lw=.45);ax.set_axisbelow(True)
    depths=[2,4,8,12,16,20,24,28]

    # Main internalization comparison: identical examples/updates; measured
    # token counts differ and are not called a matched-token budget.
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.85),layout='constrained')
    for method,color,marker,label in [('msd',GREEN,'^','MSD Eq.(1) pilot'),('sft_trace',PURPLE,'s','Teacher-suffix SFT proxy')]:
        cs=[metric(f'matched_{method}_a{x}_seed20281028') for x in ('0','0.5','1')]
        error(axs[0],[0,.5,1],[100*z['accuracy'] for z in cs],[[100*t for t in z['accuracy_ci95']] for z in cs],color=color,marker=marker,label=label)
        error(axs[1],[0,.5,1],[1000*z['p50_latency_s'] for z in cs],[[1000*t for t in z['p50_latency_ci95']] for z in cs],color=color,marker=marker)
        axs[2].plot([0,.5,1],[z['generated_tokens_mean'] for z in cs],marker=marker,color=color,label=label)
    for condition,color,style,label in [('direct',BLUE,':','Original direct'),(BACKBONES[1],ORANGE,'--','Matched answer-only')]:
        z=metric(condition)
        for ax,field,scale in [(axs[0],'accuracy',100),(axs[1],'p50_latency_s',1000),(axs[2],'generated_tokens_mean',1)]:ax.axhline(z[field]*scale,color=color,ls=style,lw=1,label=label if ax is axs[0] else None)
    for ax in axs:ax.set_xlabel('Masked fraction alpha');ax.set_xticks([0,.5,1]);clean(ax)
    axs[0].set_ylabel('Strict accuracy (%)');axs[1].set_ylabel('Warm p50 latency (ms)');axs[2].set_ylabel('Mean generated tokens')
    axs[0].set_ylim(0,100);fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=6.8)
    fig.suptitle('CoT internalization pilot: matched examples and 64 updates; one seed',fontsize=8)
    save(fig,'cot_internalization_ablation')

    fig,axs=plt.subplots(1,2,figsize=(7.1,3.6),layout='constrained')
    for ax,kind,title in zip(axs,['native','trained'],['Untrained native LM readout','Trained candidate-conditioned readout']):
        points=[]
        for backbone,name,color,marker in zip(BACKBONES,NAMES,COLORS,MARKERS):
            for d in depths:
                key=f'v6/exit/{backbone}/{kind}/L{d}';z=m[key];x=1000*z['p50_latency_s'];y=100*z['accuracy'];points.append((x,y))
                ax.errorbar(x,y,xerr=[[max(0,x-1000*z['p50_latency_ci95'][0])],[max(0,1000*z['p50_latency_ci95'][1]-x)]],yerr=[[max(0,y-100*z['accuracy_ci95'][0])],[max(0,100*z['accuracy_ci95'][1]-y)]],fmt=marker,color=color,ms=3,capsize=1.5,elinewidth=.5,alpha=.78)
                if d in (2,8,28) and backbone==BACKBONES[2]:ax.annotate('L'+str(d),(x,y),xytext=(3,5 if d!=8 else -10),textcoords='offset points',fontsize=6.5,color=color)
            z=metric(backbone);x=1000*z['p50_latency_s'];y=100*z['accuracy'];points.append((x,y));ax.scatter(x,y,c=color,marker='*',s=55,zorder=5)
        frontier=sorted([p for p in points if not any(q[0]<=p[0] and q[1]>=p[1] and q!=p for q in points)])
        if frontier:ax.plot(*zip(*frontier),ls='--',color=GRAY,lw=.8)
        ax.set_xscale('log');ax.set_xlabel('Warm serial p50 latency (ms)');ax.set_title(title);clean(ax);ax.set_ylim(0,100)
    axs[0].set_ylabel('Strict accuracy (%)')
    handles=[Line2D([],[],color=c,marker=z,ls='none',label=n) for n,c,z in zip(NAMES,COLORS,MARKERS)]+[Line2D([],[],color=GRAY,marker='*',ls='none',label='Full generation'),Line2D([],[],color=GRAY,ls='--',label='Observed frontier')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=6.7);fig.suptitle('Accuracy–latency trade-offs: 500 synthetic items, clustered 95% intervals',fontsize=8)
    save(fig,'accuracy_latency_pareto')

    fig,axs=plt.subplots(1,2,figsize=(7.1,3.1),sharey=True,layout='constrained')
    for ax,kind in zip(axs,['native','trained']):
        for backbone,name,color,marker in zip(BACKBONES,NAMES,COLORS,MARKERS):
            zs=[m[f'v6/exit/{backbone}/{kind}/L{d}'] for d in depths]
            error(ax,depths,[100*z['accuracy'] for z in zs],[[100*t for t in z['accuracy_ci95']] for z in zs],color=color,marker=marker,label=name)
        ax.set_title('Native LM readout' if kind=='native' else 'Trained readout');ax.set_xlabel('Executed decoder blocks');ax.set_xticks(depths);ax.set_ylim(0,100);clean(ax)
    axs[0].set_ylabel('Accuracy (%)');fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False);fig.suptitle('Fixed exits; five actual serial timing repeats, one trained seed',fontsize=8)
    save(fig,'layer_depth_results')

    fig,axs=plt.subplots(1,2,figsize=(7.1,3.2),sharey=True,layout='constrained')
    for ax,kind in zip(axs,['native','trained']):
        for backbone,name,color,marker in zip(BACKBONES,NAMES,COLORS,MARKERS):
            zs=[paired[f'v6/exit/{backbone}/{kind}/L{d} minus v6/generation/direct'] for d in depths]
            error(ax,depths,[z['accuracy_difference_pp'] for z in zs],[z['paired_ci95_pp'] for z in zs],color=color,marker=marker,label=name)
        ax.axhline(-2,color=GRAY,ls='--',lw=.9);ax.axhline(0,color=GRAY,lw=.6);ax.set_title('Native LM readout' if kind=='native' else 'Trained readout');ax.set_xlabel('Executed decoder blocks');ax.set_xticks(depths);clean(ax)
    axs[0].set_ylabel('Paired accuracy difference vs direct (pp)');fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False);fig.suptitle('Quality noninferiority: dashed line is the -2 pp margin',fontsize=8)
    save(fig,'paired_quality_difference')

    fig,axs=plt.subplots(1,3,figsize=(7.1,2.9),sharex=True,sharey=True,layout='constrained')
    rc={'confidence':(BLUE,'o'),'margin':(ORANGE,'s'),'entropy':(PURPLE,'D'),'agreement':(GREEN,'^')}
    for ax,c,name in zip(axs,BACKBONES,NAMES):
        policy=json.loads((r/'configs'/f'matched_policy_unmerged_{c}.json').read_text())
        for kind,(color,marker) in rc.items():
            zs=sorted([z for z in policy['all_rules'] if z['kind']==kind and z['early_risk'] is not None],key=lambda z:z['coverage'])
            ax.plot([100*z['coverage'] for z in zs],[100*z['early_risk'] for z in zs],color=color,marker=marker,ms=3,label=kind)
        ax.set_title(name);ax.set_xlabel('Early-exit coverage (%)');ax.set_xlim(0,100);ax.set_ylim(0,100);clean(ax)
        ax.text(.03,.97,'Selected: '+('yes' if policy['selected'] else 'none'),transform=ax.transAxes,va='top',fontsize=7)
    axs[0].set_ylabel('Error among early exits (%)');fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False,fontsize=6.6);fig.suptitle('Dynamic rules: dev-only risk–coverage; test only if gates passed',fontsize=8)
    save(fig,'fixed_dynamic_exit_results')

    keys=['v6/generation/'+x for x in BACKBONES]+[f'v6/exit/{BACKBONES[2]}/trained/L{d}' for d in (2,8,16,24,28)]
    labels=['Original gen','Answer-only gen','MSD1 gen','MSD1 head L2','MSD1 head L8','MSD1 head L16','MSD1 head L24','MSD1 head L28'];fams=sorted(next(iter(m.values()))['families'])
    vals=np.array([[100*(1-m[k]['families'][f]['accuracy']) for f in fams] for k in keys])
    fig,ax=plt.subplots(figsize=(7.1,3.3),layout='constrained');im=ax.imshow(vals,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto')
    for i in range(len(keys)):
        for j in range(len(fams)):ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white' if vals[i,j]>65 else 'black')
    ax.set_xticks(range(len(fams)),[f.replace('_',' ') for f in fams],rotation=20,ha='right');ax.set_yticks(range(len(labels)),labels);ax.set_title('Error rates (%) by task family; each cell uses 83–84 items');fig.colorbar(im,ax=ax,label='Error (%)',shrink=.9)
    save(fig,'stratified_errors')

    ext=[(name,z) for name,z in paired.items() if z['current'].startswith('external/')];display=[]
    for name,z in ext:
        c=z['current'].split('/')[-1]
        short=c.replace('matched_','').replace('_seed20281028','').replace('sft_trace','SFT proxy').replace('sft_answer_a1','Answer-only').replace('msd_a','MSD alpha=').replace('_a',' alpha=')
        if c.startswith('program_'):short='Prior unmatched SFT'
        display.append((short,z))
    fig,ax=plt.subplots(figsize=(7.1,3.4),layout='constrained')
    for i,(label,z) in enumerate(display):
        lo,hi=z['paired_ci95_pp'];v=z['accuracy_difference_pp'];ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',ms=4,capsize=2,color=GREEN if 'MSD' in label else PURPLE)
    ax.set_yticks(range(len(display)),[x[0] for x in display]);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=.7);ax.set_xlabel('Paired accuracy difference vs original (pp)');ax.set_title('Independent regression check: AG News, 120 items, five draws/item');clean(ax)
    save(fig,'general_capability_regression')

    fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    for method,ax in [('sft',axs[0]),('msd',axs[1])]:
        for path in sorted((r/'results').glob('matched_*/train_log.json')):
            c=path.parent.name
            if (method=='msd')!=c.startswith('matched_msd'):continue
            d=json.loads(path.read_text());alpha=d['alpha'];label='answer-only' if d['method_label']=='sft_answer' else f'alpha={alpha:g}'
            ax.plot([z['step'] for z in d['steps']],[z['loss'] for z in d['steps']],label=label,lw=.8)
        ax.set_xlabel('Optimizer update');ax.set_ylabel('Mean microbatch loss');ax.set_title('Token-mean SFT cross entropy' if method=='sft' else 'MSD summed reverse KL');ax.set_yscale('symlog',linthresh=.001);ax.legend(frameon=False,fontsize=6.5);clean(ax)
    fig.suptitle('Actual training curves; loss scales differ by objective',fontsize=8);save(fig,'training_curves')
    meta={'source':'results/matched_aggregate_metrics.json','figures':created,'formats':['editable SVG text/vector','PDF vector','PNG 300 dpi'],'width_inches':7.1,'font':'DejaVu Sans','text_size_pt':'7–8','unmeasured_intervention_figure_created':False}
    (figdir/'matched_figure_manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Render the same training logs with legends outside the data panels."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


def render(run):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'axes.linewidth': .65,
                         'lines.linewidth': 1.1, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), layout='constrained')
    for method, axis in zip(('sft', 'msd'), axes):
        for path in sorted((run / 'results').glob('matched_*/train_log.json')):
            if (method == 'msd') != path.parent.name.startswith('matched_msd'):
                continue
            record = json.loads(path.read_text())
            alpha = record['alpha']
            label = 'answer-only' if record['method_label'] == 'sft_answer' else f'alpha={alpha:g}'
            marker = 'o' if record['method_label'] == 'sft_answer' else {0.: 's', .5: '^', 1.: 'D'}[alpha]
            axis.plot([row['step'] for row in record['steps']],
                      [row['loss'] for row in record['steps']],
                      label=label, lw=.8, marker=marker, markevery=8, ms=2.5)
        axis.set_xlabel('Optimizer update')
        axis.set_ylabel('Mean microbatch loss')
        axis.set_title('Token-mean SFT cross entropy' if method == 'sft' else 'MSD summed reverse KL')
        axis.set_yscale('symlog', linthresh=.001)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:.0e}' if value and abs(value) < .01 else f'{value:g}'))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2,
                    frameon=False, fontsize=7, borderaxespad=0.)
        axis.grid(alpha=.18, lw=.45)
        axis.set_axisbelow(True)
    fig.suptitle('Actual training curves; loss scales differ by objective', fontsize=8)
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(run / 'figures' / ('training_curves.' + extension),
                    dpi=300 if extension == 'png' else None)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    render(parser.parse_args().run.resolve())

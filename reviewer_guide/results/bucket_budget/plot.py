"""Render the paired-budget pilot; pip install matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DEST=Path(__file__).resolve().parent
s=json.loads((DEST/'summary.json').read_text())

plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharey=True)
colors = {'full': '#222222', 'top4': '#D55E00', 'top8': '#0072B2', 'top16': '#009E73'}
markers = {'full': 's', 'top4': 'v', 'top8': 'o', 'top16': '^'}
for ax, (length, cohort) in zip(axes, s['by_length'].items()):
    for arm, row in cohort['arms'].items():
        label = 'Full' if arm == 'full' else f"B={arm[3:]}"
        x, y = row['graph_module_median_case_ms'], row['quality']
        ax.scatter(x, y, s=45, color=colors[arm], marker=markers[arm], label=label, zorder=3)
        dy = -13 if arm == 'top16' else 7
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(0, dy), ha='center', fontsize=8)
    ax.set_title(f'{int(length)//1024}K context; 13 RULER cases')
    ax.set_xlabel('Complete Attention module (ms)')
    ax.set_ylim(62, 100)
    ax.grid(axis='y', alpha=.18)
axes[0].set_ylabel('Official score (0-100)')
fig.tight_layout(pad=1.0)
for suffix in ('png', 'pdf', 'svg'):
    fig.savefig(DEST / ('quality_latency.' + suffix), bbox_inches='tight')
plt.close(fig)

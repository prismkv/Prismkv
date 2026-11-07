"""Plot the complete 195-case budget follow-up; requires matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DEST=Path(__file__).resolve().parent
s=json.loads((DEST/'summary.json').read_text())
arms=('full','top4','top8','top16')
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
fig, axes = plt.subplots(2,3,figsize=(10,5.9),sharey=True)
colors={'full':'#222222','top4':'#D55E00','top8':'#0072B2','top16':'#009E73'}
markers={'full':'s','top4':'v','top8':'o','top16':'^'}
for col,(length,cohort) in enumerate(s['by_length'].items()):
    ax=axes[0,col]
    ax.axhline(cohort['arms']['full']['quality'],color=colors['full'],ls='--',lw=1,label='Full quality')
    for arm in arms[1:]:
        row=cohort['arms'][arm]
        x,y=row['old_read_routed_only']*100,row['quality']
        ax.scatter(x,y,color=colors[arm],marker=markers[arm],s=38)
        ax.annotate('B='+arm[3:],(x,y),xytext=(0,-14 if arm=='top16' else 6),textcoords='offset points',ha='center',fontsize=8)
    ax.set_title(str(int(length)//1024)+'K; 65 cases')
    ax.set_xlabel('Routed old-history reads (%)')
    ax.set_ylim(62,101); ax.grid(axis='y',alpha=.18)
    ax=axes[1,col]
    for arm in arms:
        row=cohort['arms'][arm]
        x,y=row['graph_module_median_case_ms'],row['quality']
        ax.scatter(x,y,color=colors[arm],marker=markers[arm],s=38)
        dy=10 if arm=='top16' and col==0 else 6
        ax.annotate('Full' if arm=='full' else 'B='+arm[3:],(x,y),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xlabel('Graph complete Attention (ms)');ax.grid(axis='y',alpha=.18)
axes[0,0].set_ylabel('Official score (0-100)');axes[1,0].set_ylabel('Official score (0-100)')
axes[0,0].legend(frameon=False,loc='lower right',fontsize=8)
fig.tight_layout(pad=1.1)
for ext in ('png','pdf','svg'):
    fig.savefig(DEST/('budget_quality_latency.'+ext),bbox_inches='tight')
plt.close(fig)

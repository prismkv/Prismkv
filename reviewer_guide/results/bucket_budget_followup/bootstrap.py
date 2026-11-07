"""Descriptive paired, cell-stratified intervals; requires numpy."""
import json
from pathlib import Path
import numpy as np

s=json.loads((Path(__file__).resolve().parent/'summary.json').read_text())
arms=('full','top4','top8','top16')
groups={}
for row in s['cases']:
    groups.setdefault((row['band'],row['task']),[]).append(row)
assert len(groups)==39 and all(len(rows)==5 for rows in groups.values())
values=np.array([[[r['quality'][arm]['score'] for arm in arms] for r in rows]
                 for key,rows in sorted(groups.items())])
rng=np.random.default_rng(20261107)
means=np.empty((10000,4))
for repeat in range(len(means)):
    ix=rng.integers(0,5,size=(39,5))
    means[repeat]=values[np.arange(39)[:,None],ix].mean((0,1))
result={}
for label,i,j in [('top16_minus_full',3,0),('top8_minus_top16',2,3),('top4_minus_top16',1,3)]:
    result[label]=dict(delta=float((values[:,:,i]-values[:,:,j]).mean()),
                       percentile_95=np.quantile(means[:,i]-means[:,j],[.025,.975]).tolist())
print(json.dumps(result,indent=2))

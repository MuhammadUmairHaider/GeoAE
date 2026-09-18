import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source=Path('results/range_number_dpc_control.json')
d=json.loads(source.read_text()); arms=d['arms']; examples=d['test_examples']
assert d['status']=='complete' and len(arms)==144
fit=d['fit_examples']; labels=np.array([x['subject_number'] for x in examples])
lemmas=np.array([x['subject_lemma'] for x in examples]); nouns=sorted(set(lemmas))
for field in ['subject_lemma','template','prompt','id']:
 assert set(x[field] for x in fit).isdisjoint(x[field] for x in examples)
for key,a in arms.items():
 source_num=int('suppress_plural' in key)
 corr=np.array(a['rows']['correct']); ref=np.array(a['rows']['ref_correct'])
 diff=ref.astype(float)-corr.astype(float)
 assert np.isclose(a['all']['target_drop'],diff[labels==source_num].mean())
 assert np.isclose(a['all']['complement_drop'],diff[labels!=source_num].mean())
 assert a['all']==a['joint_correct']

def aggregate(space,mode,alpha):
 rows=[arms[f'{space}:suppress_{direction}:{mode}:a{alpha:.1f}'] for direction in ['singular','plural']]
 target_probs = np.concatenate([np.asarray(r['rows']['gold_pair_probability'])[labels == source] for source, r in enumerate(rows)])
 return {'strict_target_flip': float(np.mean(target_probs < .5)),
 'target_tie_rate': float(np.mean(target_probs == .5)),
 **{k:float(np.mean([r['all'][k] for r in rows])) for k in
 ['target_drop','complement_drop','selectivity','target_counterpart_top1']},
 'neutral_kl':float(np.mean([r['neutral']['kl'] for r in rows])),
 'neutral_top1_change':float(np.mean([r['neutral']['top1_change'] for r in rows])),
 'matching_flip':float(np.mean([r['matching_attractor']['target_drop'] for r in rows])),
 'opposite_flip':float(np.mean([r['opposite_attractor']['target_drop'] for r in rows]))}
spaces=['h','z','z_rot0','z_rot1','z_rot2','z_shuffled']
summary={f'{space}:{mode}:a{alpha:.1f}':aggregate(space,mode,alpha)
 for space in spaces for mode in ['global','salient','range','transport'] for alpha in [.5,1.,2.]}

def noun_flip(space,mode):
 flipped=np.zeros(len(labels))
 for source_num,direction in enumerate(['singular','plural']):
  row=arms[f'{space}:suppress_{direction}:{mode}:a1.0']['rows']
  flip=(np.array(row['gold_pair_probability']) < .5).astype(float)
  mask=labels==source_num;flipped[mask]=flip[mask]
 return np.array([flipped[lemmas==noun].mean() for noun in nouns])
rng=np.random.RandomState(20260917)
indices=rng.randint(0,len(nouns),size=(20000,len(nouns)))
comparisons={}
for mode in ['range','transport']:
 z=noun_flip('z',mode)
 for control in ['h','rotations_mean']:
  baseline=noun_flip('h',mode) if control=='h' else np.mean([noun_flip(f'z_rot{i}',mode) for i in range(3)],axis=0)
  delta=z-baseline;ci=np.quantile(delta[indices].mean(1),[.025,.975])
  comparisons[f'{mode}:z_minus_{control}']={'metric': 'strict_target_flip', 'difference':float(delta.mean()),'noun_bootstrap_95':ci.tolist(),'n_subject_lemmas':len(nouns),'noun_differences':delta.tolist()}

report={'source':str(source),'checked_arms':len(arms),'holdouts_disjoint':True,
 'drop_summaries_recomputed_from_rows':True, 'flip_definition': 'gold_pair_probability < 0.5; exact ties reported separately','n_test':len(examples),'n_subject_lemmas':len(nouns),
 'averages_over_suppression_directions':summary,'paired_alpha1_comparisons':comparisons,
 'caveat':'Bootstrap resamples the 12 held-out subject nouns; it does not capture training-seed or template-family uncertainty.'}
Path('eval_out/number_control_review.json').write_text(json.dumps(report,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1,1.2]})
x=np.arange(len(spaces));width=.35
for source_num,direction,color in [(0,'singular','#4263eb'),(1,'plural','#0b9b83')]:
 vals=[100*np.mean(np.asarray(arms[f'{s}:suppress_{direction}:range:a1.0']['rows']['gold_pair_probability'])[labels == source_num] < .5) for s in spaces]
 bars=ax.bar(x+(source_num-.5)*width,vals,width,label=f'Suppress {direction}',color=color)
 ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
ax.set_xticks(x,['Residual','GeoAE','Rotation 0','Rotation 1','Rotation 2','Shuffled'],rotation=25,ha='right')
ax.set_ylim(0,112);ax.set_ylabel('Target is/are preference flipped (%)')
ax.set_title('Range shift at alpha = 1')
ax.legend(loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
styles=[('h','global','Residual global','#777777','--'),('z','global','GeoAE global','#5f3dc4','--'),
 ('z','salient','GeoAE salient','#e67700','-.'),('z','range','GeoAE range','#4263eb','-'),
 ('z','transport','GeoAE transport','#0b9b83','-')]
for space,mode,label,color,ls in styles:
 vals=[aggregate(space,mode,a) for a in [.5,1.,2.]]
 bx.plot([v['neutral_kl'] for v in vals],[100*v['strict_target_flip'] for v in vals],marker='o',color=color,ls=ls,label=label)
for i in range(3):
 vals=[aggregate(f'z_rot{i}','transport',a) for a in [.5,1.,2.]]
 bx.plot([v['neutral_kl'] for v in vals],[100*v['strict_target_flip'] for v in vals],color='#b6b6b6',marker='x',alpha=.8,label='Rotated transport (3 seeds)' if i==0 else None)
bx.set_xscale('log');bx.set_ylim(-3,108);bx.set_xlabel('Neutral-prompt KL from own baseline (log scale; lower is better)')
bx.set_ylabel('Target is/are preference flipped (%)');bx.set_title('Effect versus additional distribution change')
bx.grid(alpha=.2);bx.legend(fontsize=8,loc='lower right')
fig.suptitle('Number control: learned coordinates retain stronger coordinate-wise edits',fontsize=13)
fig.text(.5,.01,'144 held-out prompts; 12 subject nouns; 16 neutral prompts. Points: alpha 0.5, 1, 2. Strict forced-choice flips exclude ties and are not full-vocabulary generation hits.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.045,1,.95]);out=Path('figures/number_control');out.mkdir(exist_ok=True)
fig.savefig(out/'review.png',dpi=180);fig.savefig(out/'review.pdf');plt.close(fig)
print(json.dumps(comparisons,indent=2))
print('Audit: all 144 arm summaries match saved per-example outcomes; held-out nouns/templates/prompts are disjoint.')
print('Saved eval_out/number_control_review.json and figures/number_control/review.{png,pdf}')

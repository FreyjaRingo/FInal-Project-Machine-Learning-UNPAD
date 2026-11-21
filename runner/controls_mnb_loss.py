"""Follow-up controls for the paper revision (supervisor feedback).
1. Multinomial naive Bayes (MNB) vs CNB: BR x {none, weight, mlsmote} x {fixed, thr, tuned}.
2. MLP with imbalance-aware losses: focal loss (gamma=2) and asymmetric loss (gamma+=0, gamma-=4, clip 0.05),
   plus the BCE baseline retrained on the same machine; BR, tuned mode.
The script loads the functions of the Structured notebook unchanged and patches only make_estimator,
train_binary_model (MNB) and the loss of train_mlp. Results: results/experiments_controls.csv,
out-of-fold scores: results/preds_controls/<config>__f<fold>.npz.
Run from the repository root:  python runner/controls_mnb_loss.py [mnb|mlp]
"""
from pathlib import Path
import os,sys,time,json
os.chdir(Path(__file__).resolve().parents[1])
import json, sys, types
from pathlib import Path
NB=str(Path(__file__).resolve().parents[1]/'Indonesian_Multilabel_Hate_Speech_Detection_Structured.ipynb')
n=json.load(open(NB,encoding='utf-8'))
sys.modules['IPython.display']=types.SimpleNamespace(display=print)
from pathlib import Path
g={'__name__':'nbh'}
cells=[4,7,11]+list(range(14,200))
for i in cells:
    c=n['cells'][i]
    if c['cell_type']!='code': continue
    if i==11: g['ROOT']=Path.cwd(); g['RES']=g['ROOT']/'results'
    exec(compile(''.join(c['source']),f'cell{i}','exec'),g)

import numpy as np, torch
from torch import nn
from sklearn.naive_bayes import MultinomialNB
g['BASE_PARAMS']['mnb']={'alpha':1.0}; g['HYPERPARAMETER_GRID']['mnb']=[{'alpha':a} for a in (0.1,0.3,1.0)]
# patch make_estimator and train_binary_model for MNB
src=''.join(n['cells'][104]['source']).replace('    if algo == "cnb":\n','    if algo == "mnb":\n        return MultinomialNB(**params)\n    if algo == "cnb":\n')
g['MultinomialNB']=MultinomialNB; exec(src,g)
src=''.join(n['cells'][106]['source']).replace('algo == "cnb"','algo in ("cnb", "mnb")'); exec(src,g)
# loss-based MLP
class FocalLoss(nn.Module):
    def __init__(s,gamma=2.0): super().__init__(); s.gamma=gamma
    def forward(s,logits,y):
        p=torch.sigmoid(logits); ce=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
        pt=p*y+(1-p)*(1-y); return ((1-pt)**s.gamma*ce).mean()
class AsymmetricLoss(nn.Module):
    def __init__(s,gp=0.0,gn=4.0,clip=0.05): super().__init__(); s.gp,s.gn,s.clip=gp,gn,clip
    def forward(s,logits,y):
        p=torch.sigmoid(logits); pm=(p-s.clip).clamp(min=0)
        lp=y*((1-p)**s.gp)*torch.log(p.clamp(min=1e-8))
        ln=(1-y)*(pm**s.gn)*torch.log((1-pm).clamp(min=1e-8))
        return -(lp+ln).mean()
g['FocalLoss'],g['AsymmetricLoss']=FocalLoss,AsymmetricLoss; g['LOSS_MODE']='bce'
src=''.join(n['cells'][100]['source']).replace('    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)\n',
 '    loss_fn = FocalLoss() if LOSS_MODE == "focal" else AsymmetricLoss() if LOSS_MODE == "asl" else nn.BCEWithLogitsLoss(pos_weight=pos_weight)\n')
assert 'FocalLoss()' in src; exec(src,g)
data,folds=g['load_inputs'](g['ROOT'])
out=g['ROOT']/'results'/os.environ.get('OUTCSV','experiments_controls.csv')
done=g['completed_runs'](out)
grid=[(a,'br',c,t) for a in ['mnb'] for c in ['none','weight','mlsmote'] for t in ['fixed','thr','tuned']]
grid+=[('mlp','br',c,t) for t in ['tuned'] for c in ['focal','asl','none']]
only=sys.argv[1] if len(sys.argv)>1 else None
for algo,st,cond,tun in grid:
    if only and algo!=only: continue
    g['LOSS_MODE']=cond if cond in ('focal','asl') else 'bce'
    for f in range(5):
        cfg=g['config_id'](algo,st,cond,tun)
        if (cfg,f) in done: continue
        t=time.time(); (g['ROOT']/'results'/'preds_controls').mkdir(exist_ok=True); r=g['run_experiment'](g['ROOT'],algo,st,cond,tun,'word',f,data,folds,preds_subdir='preds_controls')
        g['append_rows'](out,[r]); print(cfg,f,round(r['macro_f1'],4),round(r['minority_macro_f1'],4),r['override'],f'{time.time()-t:.0f}s',flush=True)

"""Descriptive cap sensitivity; never replaces original study outcomes."""
import argparse,pathlib,numpy as np
from core import read,put
from audit_results import verify
from vendor.scoring import score

def analyze(prepared,outputs,destination):
 rows=verify(prepared,outputs,'cap');baseline=read(prepared/'BASELINE.json');lookup={(r['reader'],r['item_id']):r for r in baseline};results={};scored=[];rng=np.random.default_rng(2026092106)
 for r in rows:
  old=read(prepared/r['path']/'ORIGINAL.json');base=lookup[r['reader'],r['case_id']]
  new=score(r['prediction'],base['reference'],'locomo',category=base['category'])['canonical']['f1']
  scored.append(dict(r,cluster=base['cluster'],old_f1=base['score']['canonical']['f1'],new_f1=new,prefix96_matches=r['generated_token_ids'][:96]==old['generated_token_ids']))
 for reader in ['Qwen2B','Qwen9B']:
  rr=[r for r in scored if r['reader']==reader];new={(r['reader'],r['case_id']):r['new_f1'] for r in rr};bb=[b for b in baseline if b['reader']==reader];clusters=sorted({b['cluster'] for b in bb});oldmeans=[];newmeans=[]
  for c in clusters:
   group=[b for b in bb if b['cluster']==c];oldmeans.append(np.mean([b['score']['canonical']['f1'] for b in group]));newmeans.append(np.mean([new.get((reader,b['item_id']),b['score']['canonical']['f1']) for b in group]))
  d=np.array(newmeans)-oldmeans;boot=d[rng.integers(0,10,size=(50000,10))].mean(axis=1)
  results[reader]={'fixed_capped_n':len(rr),'prefix96_mismatches':sum(not r['prefix96_matches'] for r in rr),'still_capped_256':sum(r['cap_hit'] for r in rr),'capped_subset_question_mean_old':float(np.mean([r['old_f1'] for r in rr])),'capped_subset_question_mean_new':float(np.mean([r['new_f1'] for r in rr])),'full_500_conversation_macro_old':float(np.mean(oldmeans)),'full_500_conversation_macro_sensitivity':float(np.mean(newmeans)),'macro_change':float(d.mean()),'paired_conversation_bootstrap_ci95':list(map(float,np.quantile(boot,[.025,.975]))),'absolute_change_at_least_3_points':bool(abs(d.mean())>=.03),'conversation_changes':dict(zip(clusters,map(float,d)))}
 destination.mkdir(parents=True,exist_ok=False);put(destination/'cap_analysis.json',{'role':'conditional descriptive cap sensitivity, not replacement primary outcomes','readers':results,'interpretation':'Any prefix mismatch requires execution-plus-cap interpretation; no selective exclusions. Continuing caps make small changes inconclusive. No further cap increase.'});put(destination/'cap_rows.json',scored)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=pathlib.Path,required=True);p.add_argument('--outputs',type=pathlib.Path,nargs=2,required=True);p.add_argument('--destination',type=pathlib.Path,required=True);a=p.parse_args();analyze(a.prepared,a.outputs,a.destination)

"""Verify every record before development selection or held-out scoring."""
import argparse,collections,hashlib,json,pathlib,sys
import numpy as np
from core import D,read,put,sha,verify_files
from tokenizers import Tokenizer
from vendor.scoring import score
def canonical(x):
 x=x.strip();left='<|begin_of_box|>';right='<|end_of_box|>'
 if x.startswith(left) and x.endswith(right) and x.count(left)==x.count(right)==1:x=x[len(left):-len(right)].strip()
 return ' '.join(x.split())
def verify(prepared,outputs,stage):
 freeze=read(prepared/'INPUT_FREEZE.json');protocol=read(D/'PROTOCOL.json');verify_files(D,read(D/'DEV_FREEZE.json')['files']);verify_files(prepared,freeze['files'])
 if freeze['stage']!=stage or freeze['dev_freeze_sha256']!=sha(D/'DEV_FREEZE.json'):raise RuntimeError('Source or stage mismatch')
 expected=freeze['rows'];all_rows=[]
 for output in outputs:
  start=read(output/'RUN_START.json');complete=read(output/'RUN_COMPLETE.json');reader=start['reader'];want=[r for r in expected if r['reader']==reader]
  if start['stage']!=stage or start['dev_freeze_sha256']!=sha(D/'DEV_FREEZE.json') or start['protocol_sha256']!=sha(D/'PROTOCOL.json') or start['input_freeze_sha256']!=sha(prepared/'INPUT_FREEZE.json') or complete.get('status')!='complete-unverified' or complete['calls']!=len(want):raise RuntimeError('Incomplete or mismatched run')
  env=read(output/'ENVIRONMENT.json');snap=read(D/f'environment/{reader}_SNAPSHOT.json')
  if env['torch']!='2.10.0+cu128' or env['transformers']!='5.3.0' or 'A100' not in env['gpu'] or env['revision']!=snap['revision'] or env['generation']['max_new_tokens']!=(256 if stage=='cap' else 64) or env['generation']['do_sample'] or env['generation']['num_beams']!=1:raise RuntimeError('Output environment mismatch')
  if {x.name for x in output.iterdir() if x.is_dir()}!={r['case_id']+'--'+r['arm'] for r in want}:raise RuntimeError('Unexpected or missing output folders')
  tok=Tokenizer.from_file(str(D/'assets'/('glm_tokenizer.json' if reader=='GLM' else 'reference_tokenizer.json')))
  for row in want:
   folder=output/(row['case_id']+'--'+row['arm']);s=read(folder/'START.json');done=read(folder/'COMPLETE.json');raw=read(folder/'RAW.json')
   if s['row']!=row or done['raw_sha256']!=sha(folder/'RAW.json') or done['journal_sha256']!=sha(folder/'TOKENS.jsonl'):raise RuntimeError('Row journal/hash failure')
   journal=[json.loads(x) for x in (folder/'TOKENS.jsonl').read_text().splitlines()];m=read(prepared/row['path']/'MEASUREMENT.json')
   if journal[0]!=[m['input_token_ids']] or [v for line in journal[1:] for v in line]!=raw['generated_token_ids']:raise RuntimeError('Token journal differs')
   for skip,key in [(True,'prediction'),(False,'decoded_with_special_tokens')]:
    if tok.decode(raw['generated_token_ids'],skip_special_tokens=skip)!=raw[key]:raise RuntimeError('Decode mismatch')
   if len(raw['generated_token_ids'])>(256 if stage=='cap' else 64) or done['generated_tokens']!=len(raw['generated_token_ids']):raise RuntimeError('Generation cap violation')
   if done['cap_hit']!=(len(raw['generated_token_ids'])==(256 if stage=='cap' else 64)) or s['max_new_tokens']!=(256 if stage=='cap' else 64):raise RuntimeError('Cap metadata mismatch')
   all_rows.append(dict(row,prediction=raw['prediction'],generated_token_ids=raw['generated_token_ids'],generated_token_count=len(raw['generated_token_ids']),cap_hit=done['cap_hit']))
 if {(r['reader'],r['case_id'],r['arm']) for r in all_rows}!={(r['reader'],r['case_id'],r['arm']) for r in expected} or len(all_rows)!=len(expected):raise RuntimeError('Missing/duplicated run or row; do not score partial evaluation')
 return all_rows
def select(prepared,outputs,destination):
 rows=verify(prepared,outputs,'dev');scores=[]
 for layout in read(D/'PROTOCOL.json')['layouts']:
  rr=[r for r in rows if r['arm']==layout];hits=[];glyph=[];width=[]
  for r in rr:
   c=read(D/'cases/dev'/(r['case_id']+'.json'));hits.append(canonical(r['prediction'])==c['reference']);g=read(prepared/r['path']/'REQUEST.json')['geometry'];glyph.append(min(g['effective_font_em_pixels']));width.append(g['width_utilization'])
  scores.append({'layout':layout,'n':len(rr),'correct':sum(hits),'em':float(np.mean(hits)),'median_min_page_effective_font_em':float(np.median(glyph)),'median_max_width_fraction':float(np.median(width))})
 chosen=sorted(scores,key=lambda s:(-s['correct'],-s['median_min_page_effective_font_em'],-s['median_max_width_fraction'],s['layout']))[0]['layout']
 put(destination,{'status':'development-verified','selected_layout':chosen,'scores':scores,'dev_freeze_sha256':sha(D/'DEV_FREEZE.json'),'prepared_input_freeze_sha256':sha(prepared/'INPUT_FREEZE.json'),'result_files':[{'path':str(o/'RUN_COMPLETE.json'),'sha256':sha(o/'RUN_COMPLETE.json')} for o in outputs],'selection_rule':'Frozen EM, glyph, width, lexical order; no additional development round','verified_rows':36})
 print('Development verified and selected layout recorded; prepare all evaluation inputs before inference.')
def analyze(prepared,outputs,destination):
 from scipy.stats import binomtest
 rows=verify(prepared,outputs,'eval');p=read(D/'PROTOCOL.json');rng=np.random.default_rng(p['statistics_seed']);results={};tests=[]
 for reader in p['models']:
  rr=[r for r in rows if r['reader']==reader]
  for r in rr:r['scores']=score(r['prediction'],read(D/'cases/eval'/(r['case_id']+'.json'))['reference'],'controlled')
  ids=sorted({r['case_id'] for r in rr});lookup={(r['case_id'],r['arm']):r for r in rr};vectors={}
  for arm in p['eval_arms']:vectors[arm]=np.array([canonical(lookup[i,arm]['prediction'])==read(D/'cases/eval'/(i+'.json'))['reference'] for i in ids],dtype=int)
  raw=vectors['original_c2_p4'];eff=vectors['efficient_c2_p4'];gains=int(sum((raw==0)&(eff==1)));losses=int(sum((raw==1)&(eff==0)));d=eff-raw
  # The source schedule lists two replicates consecutively within each joint cell.
  w=rng.binomial(2,.5,size=(50000,24));boot=(w*d[0::2]+(2-w)*d[1::2]).sum(axis=1)/48
  test={'reader':reader,'effect':float(d.mean()),'paired_ci95':list(map(float,np.quantile(boot,[.025,.975]))),'gains':gains,'losses':losses,'p':float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.}
  tests.append(test);results[reader]={'arms':{a:{'n':48,'em':float(v.mean()),'correct':int(v.sum()),'cer':float(np.mean([lookup[i,a]['scores']['canonical']['cer'] for i in ids])),'literal_raw_em':float(np.mean([lookup[i,a]['scores']['native']['literal_exact_match'] for i in ids])),'caps':sum(lookup[i,a]['cap_hit'] for i in ids)} for a,v in vectors.items()},'readable_meets_prespecified_75_percent':bool(vectors['readable_p4'].mean()>=.75),'raw_meets_prespecified_87p5_percent':bool(vectors['full_raw'].mean()>=.875),'primary':test}
 last=0
 for j,t in enumerate(sorted(tests,key=lambda x:x['p'])):last=max(last,min(1,(3-j)*t['p']));t['holm_p']=last;t['reject_005']=last<=.05
 destination.mkdir(parents=True,exist_ok=False);put(destination/'analysis.json',{'role':'new prospectively specified follow-up; separate from original six tests','readers':results,'verified_rows':576,'warning':'Two replicates per joint cell; boundary bootstrap intervals do not establish certainty. No interaction significance claims.'});put(destination/'audit.json',{'status':'passed','rows':576,'input_freeze_sha256':sha(prepared/'INPUT_FREEZE.json')})
 with (destination/'rows.jsonl').open('x',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','analyze']);p.add_argument('--prepared',type=pathlib.Path,required=True);p.add_argument('--outputs',type=pathlib.Path,nargs='+',required=True);p.add_argument('--destination',type=pathlib.Path,required=True);a=p.parse_args();(select if a.mode=='select' else analyze)(a.prepared,a.outputs,a.destination)

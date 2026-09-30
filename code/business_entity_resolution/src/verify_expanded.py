"""Independently verify saved experiment splits, predictions, and paired scores."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def ids(text):return set(filter(None,text.split(',')))
def main():
    sample=read(ROOT/'reports/expanded/validation_sample.tsv')
    byid={r['source1_entity_id']:r for r in sample}
    assert len(byid)==len(sample)==8000
    counts={s:sum(r['split']==s for r in sample) for s in ('training','threshold_tuning','evaluation')}
    assert counts==dict(training=6000,threshold_tuning=1000,evaluation=1000)
    excluded=set()
    for p in ['reports/validation_sample.tsv','reports/v2-fresh/validation_sample.tsv','reports/parallel-smoke/candidate_pairs.tsv']:
        excluded.update(r['source1_entity_id'] for r in read(ROOT/p))
    assert not excluded & byid.keys()
    rows=read(ROOT/'reports/expanded/sample_candidate_pairs.tsv')
    assert len(rows)==8000 and {r['source1_entity_id'] for r in rows}==byid.keys()
    candidates={}
    for r in rows:
        values=list(filter(None,r['candidate_entity_ids'].split(',')))
        assert len(values)==len(set(values)) and len(values)<=100
        candidates[r['source1_entity_id']]=set(values)
    evaluation={k:r for k,r in byid.items() if r['split']=='evaluation'}
    report=json.loads((ROOT/'models/expanded/expanded_report.json').read_text())
    frozen=json.loads((ROOT/'models/expanded/frozen_model.json').read_text())
    assert hashlib.sha256((ROOT/'models/expanded/matcher.joblib').read_bytes()).hexdigest()==frozen['model_sha256']
    assert hashlib.sha256((ROOT/'models/baseline/matcher.joblib').read_bytes()).hexdigest()==frozen['baseline_sha256']
    all_scores=[];recomputed={}
    for name,file in [('expanded','evaluation_matching_results.tsv'),('baseline','baseline_evaluation_matching_results.tsv')]:
        output=read(ROOT/'models/expanded'/file)
        assert len(output)==1000 and {r['source1_entity_id'] for r in output}==evaluation.keys()
        predictions={r['source1_entity_id']:r['matched_entity_ids'] for r in output}
        scores=[];tp=fp=fn=0
        for eid,r in evaluation.items():
            values=list(filter(None,predictions[eid].split(',')));pred=set(values);truth=ids(r['matched_entity_ids'])
            assert len(values)==len(pred) and pred<=candidates[eid]
            a=len(pred & truth);b=len(pred-truth);c=len(truth-pred)
            tp+=a;fp+=b;fn+=c
            scores.append(1.25*a/(1.25*a+b+.25*c) if truth or pred else 1.)
        computed=dict(tp=tp,fp=fp,fn=fn,macro_f05=float(np.mean(scores)),precision=tp/(tp+fp),recall=tp/(tp+fn))
        expected=report[name+'_evaluation']['all']
        for key,value in computed.items():assert abs(value-expected[key])<1e-12,(key,value,expected[key])
        recomputed[name]=computed;all_scores.append(np.asarray(scores))
    delta=all_scores[0]-all_scores[1];rng=np.random.default_rng(20270927)
    boot=np.mean(delta[rng.integers(0,1000,size=(5000,1000))],axis=1)
    result=dict(status='PASS',split_counts=counts,excluded_overlap=0,candidate_rows=8000,prediction_rows_per_model=1000,
                recomputed=recomputed,paired_macro_f05_difference=float(delta.mean()),
                paired_bootstrap_95_percent_interval=np.quantile(boot,[.025,.975]).tolist(),
                bootstrap_note='Seeded paired S1 bootstrap, 5000 resamples; sample uncertainty only, not France/leaderboard generalization.')
    (ROOT/'reports/expanded/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

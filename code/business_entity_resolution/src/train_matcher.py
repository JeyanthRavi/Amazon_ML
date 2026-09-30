"""Train a pair classifier with S1-disjoint tuning and evaluation groups."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from difflib import SequenceMatcher

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from improve_blocking import compact_name, jaccard, prepare, score
from audit_blocking import rows

FEATURE_NAMES = ['name_ratio','name_token_jaccard','name_token_containment','name_compact_equal',
                 'name_3gram_jaccard','name_4gram_jaccard','name_length_ratio',
                 'address_token_jaccard','address_token_containment','address_ratio',
                 'number_jaccard','number_equal','number_conflict','postcode_jaccard',
                 'address_missing_query','address_missing_target','country_equal','name_digit_jaccard',
                 'name_ascii_query','name_ascii_target','ranking_score','name_tokens_query','name_tokens_target']


def grams(text,n):
    return {text[i:i+n] for i in range(max(0,len(text)-n+1))}


def containment(a,b):
    a,b=set(a),set(b)
    return len(a&b)/min(len(a),len(b)) if a and b else 0


def features(a,b):
    x,y=compact_name(a),compact_name(b)
    an,bn={t for t in a[3] if t.isdigit()},{t for t in b[3] if t.isdigit()}
    ap,bp={t for t in an if len(t) in (5,6)},{t for t in bn if len(t) in (5,6)}
    return [SequenceMatcher(None,x,y,autojunk=False).ratio() if x and y else 0,
            jaccard(a[2],b[2]),containment(a[2],b[2]),float(bool(x) and x==y),
            jaccard(grams(x,3),grams(y,3)),jaccard(grams(x,4),grams(y,4)),
            min(len(x),len(y))/max(len(x),len(y)) if x or y else 0,
            jaccard(a[3],b[3]),containment(a[3],b[3]),
            SequenceMatcher(None,' '.join(sorted(a[3])),' '.join(sorted(b[3])),autojunk=False).ratio() if a[3] and b[3] else 0,
            jaccard(an,bn),float(bool(an) and an==bn),float(bool(an and bn) and not an&bn),jaccard(ap,bp),
            float(not a[3]),float(not b[3]),float(a[4]==b[4]),
            jaccard({t for t in a[2] if t.isdigit()},{t for t in b[2] if t.isdigit()}),
            float(x.isascii()),float(y.isascii()),score(a,b),len(a[2]),len(b[2])]


def load_sample(directory):
    with (directory/'validation_sample.tsv').open() as f:
        sample=list(csv.DictReader(f,delimiter='\t'))
    with (directory/'sample_candidate_pairs.tsv').open() as f:
        candidates={r['source1_entity_id']: r['candidate_entity_ids'].split(',') if r['candidate_entity_ids'] else []
                    for r in csv.DictReader(f,delimiter='\t')}
    return sample,candidates


def evaluate(sample,candidates,predictions):
    groups={}
    for row in sample:
        eid=row['source1_entity_id'];truth=set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
        pred=set(predictions.get(eid,[]));tp=len(truth&pred);fp=len(pred-truth);fn=len(truth-pred)
        entity_score=1.0 if not truth and not pred else (1.25*tp/(1.25*tp+fp+.25*fn) if tp else 0)
        for country in ('all',row['country']):
            g=groups.setdefault(country,dict(entities=0,tp=0,fp=0,fn=0,macro_f05=0,singletons=0,singleton_false_positive=0,truth_pairs=0,candidate_hits=0))
            g['entities']+=1;g['tp']+=tp;g['fp']+=fp;g['fn']+=fn;g['macro_f05']+=entity_score
            g['singletons']+=not truth;g['singleton_false_positive']+=bool(not truth and pred)
            g['truth_pairs']+=len(truth);g['candidate_hits']+=len(truth&set(candidates[eid]))
    for g in groups.values():
        g['macro_f05']/=g['entities'];g['precision']=g['tp']/(g['tp']+g['fp']) if g['tp']+g['fp'] else 0
        g['recall']=g['tp']/(g['tp']+g['fn']) if g['tp']+g['fn'] else 0
        g['blocking_recall']=g['candidate_hits']/g['truth_pairs'] if g['truth_pairs'] else None
    return groups


def decide(probabilities,threshold,pair_ids):
    predictions={}
    for probability,(eid,tid) in zip(probabilities,pair_ids):
        if probability>=threshold:
            predictions.setdefault(eid,[]).append(tid)
    return predictions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--development',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);started=time.time()
    development,dc=load_sample(args.development);evaluation,ec=load_sample(args.evaluation)
    dids={r['source1_entity_id'] for r in development};eids={r['source1_entity_id'] for r in evaluation}
    if dids&eids: raise ValueError('Classifier evaluation overlaps development')
    sample=development+evaluation;candidates={**dc,**ec};samples={r['source1_entity_id']:r for r in sample}
    truth={eid:set(r['matched_entity_ids'].split(',')) if r['matched_entity_ids'] else set() for eid,r in samples.items()}
    owners={}
    for eid,targets in truth.items():
        for tid in targets:
            if tid in owners and owners[tid]!=eid: raise ValueError('Shared positive target: use group-based splitting')
            owners[tid]=eid
    partitions={eid:('evaluation' if eid in eids else ('threshold_tuning' if int(hashlib.sha256(('matcher:2026:'+eid).encode()).hexdigest()[:8],16)%5==0 else 'training')) for eid in samples}
    wanted=set(t for targets in candidates.values() for t in targets);targets={}
    for source in (2,3):
        for row in rows(args.data/f'train/train_source{source}.tsv'):
            if row[0] in wanted: targets[row[0]]=prepare(row)
        print(f'Collected candidate records from S{source}: {len(targets):,}',flush=True)
    if set(targets)!=wanted: raise ValueError('Candidate IDs missing from target sources')
    matrix,labels,pair_ids,parts=[],[],[],[]
    for n,(eid,row) in enumerate(samples.items(),1):
        query=prepare([eid,row['business_name'],row['business_address'],row['country']])
        for tid in candidates[eid]:
            matrix.append(features(query,targets[tid]));labels.append(tid in truth[eid]);pair_ids.append((eid,tid));parts.append(partitions[eid])
        if n%200==0: print(f'Computed features for {n} S1 businesses',flush=True)
    x=np.asarray(matrix,dtype=np.float32);y=np.asarray(labels,dtype=np.uint8);parts=np.asarray(parts)
    train=parts=='training';tune=parts=='threshold_tuning';test=parts=='evaluation'
    model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1,
                                        early_stopping=False,random_state=2026)
    model.fit(x[train],y[train]);probabilities=model.predict_proba(x)[:,1]
    tune_sample=[r for r in sample if partitions[r['source1_entity_id']]=='threshold_tuning']
    tune_ids=[ids for ids,keep in zip(pair_ids,tune) if keep]
    thresholds=list(np.arange(.05,.951,.025))+[.975,.99,.995,.999]
    search=[]
    for t in thresholds:
        metrics=evaluate(tune_sample,candidates,decide(probabilities[tune],float(t),tune_ids))['all']
        search.append({'threshold':float(t),'macro_f05':metrics['macro_f05'],'precision':metrics['precision'],'recall':metrics['recall']})
    best=max(search,key=lambda r:(r['macro_f05'],r['threshold']))
    baseline_scores=x[:,FEATURE_NAMES.index('ranking_score')]
    baseline_search=[]
    for t in np.arange(1,4.401,.05):
        metric=evaluate(tune_sample,candidates,decide(baseline_scores[tune],float(t),tune_ids))['all']['macro_f05']
        baseline_search.append((metric,float(t)))
    baseline_threshold=max(baseline_search)[1]
    artifact={'model':model,'feature_names':FEATURE_NAMES,'threshold':best['threshold'],'license':'MIT',
              'training_s1_ids':sorted(eid for eid in samples if partitions[eid]=='training'),
              'tuning_s1_ids':sorted(eid for eid in samples if partitions[eid]=='threshold_tuning')}
    joblib.dump(artifact,args.output/'matcher.joblib')
    # Freeze model and thresholds before evaluating the classifier holdout.
    frozen={'threshold':best['threshold'],'baseline_threshold':baseline_threshold,'model_sha256':hashlib.sha256((args.output/'matcher.joblib').read_bytes()).hexdigest(),
            'training_entities':sum(p=='training' for p in partitions.values()),'tuning_entities':len(tune_sample),
            'evaluation_entities':len(evaluation),'training_pairs':int(train.sum()),'positive_training_pairs':int(y[train].sum()),
            'model_license':'MIT','library_license':'scikit-learn BSD-3-Clause','feature_names':FEATURE_NAMES,
            'split_seed':2026,'shared_positive_targets_across_sampled_S1':0}
    (args.output/'frozen_model.json').write_text(json.dumps(frozen,indent=2)+'\n')
    test_ids=[ids for ids,keep in zip(pair_ids,test) if keep]
    predicted=decide(probabilities[test],best['threshold'],test_ids)
    baseline_predicted=decide(baseline_scores[test],baseline_threshold,test_ids)
    report={'frozen':frozen,'threshold_tuning':best,'classifier_evaluation':evaluate(evaluation,candidates,predicted),
            'heuristic_evaluation':evaluate(evaluation,candidates,baseline_predicted),'threshold_search':search,
            'elapsed_seconds':time.time()-started,'limitations':['Classifier holdout is the prior blocker evaluation sample, excluded from classifier training/tuning.',
            'Small training sample; not a final competition model.', 'No labeled France evaluation.', 'Production index and full test inference still need benchmarking.']}
    (args.output/'matcher_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'evaluation_matching_results.tsv').open('w',newline='') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['source1_entity_id','matched_entity_ids'])
        for row in evaluation: writer.writerow([row['source1_entity_id'],','.join(predicted.get(row['source1_entity_id'],[]))])
    with (args.output/'split_assignments.tsv').open('w',newline='') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['source1_entity_id','split'])
        writer.writerows(sorted(partitions.items()))
    print(json.dumps({k:v for k,v in report.items() if k!='threshold_search'},indent=2),flush=True)


if __name__=='__main__': main()

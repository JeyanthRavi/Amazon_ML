"""Larger S1-disjoint training experiment with indexed, resumable features."""
import argparse
import csv
import hashlib
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from audit_blocking import rows
from fast_retrieval import CachedTargetIndex
from improve_blocking import prepare
from train_matcher import FEATURE_NAMES, decide, evaluate, features

INDEX=None
THREADS=None
CONFIGS=[dict(max_iter=150,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1),
         dict(max_iter=300,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=2),
         dict(max_iter=250,max_leaf_nodes=31,min_samples_leaf=80,l2_regularization=5)]


def initialize(index):
    global INDEX,THREADS
    THREADS=threadpool_limits(limits=1)
    INDEX=CachedTargetIndex(index)


def feature_batch(batch):
    result=[]
    for row in batch:
        query=prepare(row);targets=INDEX.candidates(row,return_records=True)
        matrix=np.asarray([features(query,t) for t in targets],dtype=np.float32).reshape(-1,len(FEATURE_NAMES))
        result.append((row[0],[t[0] for t in targets],matrix))
    return result


def create_sample(args):
    sample_path=args.output/'validation_sample.tsv';plan_path=args.output/'plan.json'
    if sample_path.exists():
        with sample_path.open() as f:return list(csv.DictReader(f,delimiter='\t'))
    excluded=set()
    for path in args.exclude:
        with path.open() as f:excluded.update(r['source1_entity_id'] for r in csv.DictReader(f,delimiter='\t'))
    total=args.train_size+args.tune_size+args.eval_size;rng=random.Random(args.seed);sample=[];eligible=0
    for row in rows(args.data/'train/train_source1.tsv'):
        if row[0] in excluded:continue
        eligible+=1
        if len(sample)<total:sample.append(row)
        else:
            position=rng.randrange(eligible)
            if position<total:sample[position]=row
    if len(sample)!=total:raise ValueError('Insufficient eligible S1 records')
    rng.shuffle(sample);selected={r[0] for r in sample};truth={}
    for eid,text in rows(args.data/'train/train_ground_truth.tsv'):
        if eid in selected:truth[eid]=text
    if set(truth)!=selected:raise ValueError('Missing selected labels')
    plan={'seed':args.seed,'training_entities':args.train_size,'tuning_entities':args.tune_size,'evaluation_entities':args.eval_size,
          'excluded_entities':len(excluded),'sampled_old_id_overlap':len(selected&excluded),'candidate_limit':100,
          'feature_names':FEATURE_NAMES,'model_configs':CONFIGS,
          'selection_rule':'Maximize tuning macro F0.5 subject to precision >= max(0.96, baseline tuning precision). Freeze before evaluation.',
          'evaluation_policy':'Compare selected expanded model and frozen baseline once on the same fresh evaluation S1 groups.'}
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    with sample_path.open('w',newline='') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['source1_entity_id','split','country','business_name','business_address','matched_entity_ids'])
        for i,row in enumerate(sample):
            split='training' if i<args.train_size else 'threshold_tuning' if i<args.train_size+args.tune_size else 'evaluation'
            writer.writerow([row[0],split,row[3],row[1],row[2],truth[row[0]]])
    with sample_path.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def generate_features(args,sample):
    checkpoint=args.output/'features_checkpoint.json';capacity=len(sample)*100
    signature=hashlib.sha256((args.output/'validation_sample.tsv').read_bytes()+Path(__file__).read_bytes()+
        Path(__file__).with_name('fast_retrieval.py').read_bytes()+Path(__file__).with_name('train_matcher.py').read_bytes()).hexdigest()
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(signature=signature,queries=0,pairs=0,complete=False,elapsed_seconds=0)
    if state['signature']!=signature:raise ValueError('Feature inputs/code changed; choose a new experiment directory')
    mode='r+' if checkpoint.exists() else 'w+'
    x=np.memmap(args.output/'features.f32',dtype=np.float32,mode=mode,shape=(capacity,len(FEATURE_NAMES)))
    y=np.memmap(args.output/'labels.u8',dtype=np.uint8,mode=mode,shape=(capacity,))
    queries=np.memmap(args.output/'pair_queries.i32',dtype=np.int32,mode=mode,shape=(capacity,))
    targets=np.memmap(args.output/'pair_targets.s32',dtype='S32',mode=mode,shape=(capacity,))
    if not state['complete']:
        positions={r['source1_entity_id']:i for i,r in enumerate(sample)}
        truths={r['source1_entity_id']:set(filter(None,r['matched_entity_ids'].split(','))) for r in sample}
        candidate_path=args.output/'sample_candidate_pairs.tsv'
        with candidate_path.open('r+b' if checkpoint.exists() else 'w+b') as output:
            if checkpoint.exists():output.truncate(state['candidate_bytes']);output.seek(state['candidate_bytes'])
            else:output.write(b'source1_entity_id\tcandidate_entity_ids\n')
            start=state['queries'];batches=[]
            for i in range(start,len(sample),100):
                batches.append([[r['source1_entity_id'],r['business_name'],r['business_address'],r['country']] for r in sample[i:i+100]])
            before=time.time()
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(args.index,)) as pool:
                for result in pool.map(feature_batch,batches):
                    for eid,ids,matrix in result:
                        if any(len(t.encode('ascii'))>32 for t in ids):raise ValueError('Target ID exceeds feature-store capacity')
                        start=state['pairs'];end=start+len(ids)
                        x[start:end]=matrix;y[start:end]=[t in truths[eid] for t in ids];queries[start:end]=positions[eid];targets[start:end]=ids
                        state['pairs']=end;state['queries']+=1;output.write((eid+'\t'+','.join(ids)+'\n').encode())
                    for array in (x,y,queries,targets):array.flush()
                    output.flush();state['candidate_bytes']=output.tell();state['elapsed_seconds']+=time.time()-before;before=time.time()
                    temporary=checkpoint.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(checkpoint)
                    if state['queries']%500==0:print(f"Features: {state['queries']:,}/{len(sample):,} S1; {state['pairs']:,} pairs",flush=True)
            state['complete']=True;checkpoint.write_text(json.dumps(state,indent=2)+'\n')
    n=state['pairs'];return x[:n],y[:n],queries[:n],targets[:n]


def predict(probabilities,threshold,qids,tids,sample):
    output={}
    for i in np.flatnonzero(probabilities>=threshold):
        eid=sample[int(qids[i])]['source1_entity_id'];output.setdefault(eid,[]).append(tids[i].decode())
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-output',type=Path,required=True);parser.add_argument('--exclude',type=Path,nargs='+',required=True)
    parser.add_argument('--train-size',type=int,default=6000);parser.add_argument('--tune-size',type=int,default=1000)
    parser.add_argument('--eval-size',type=int,default=1000);parser.add_argument('--seed',type=int,default=20270927)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);args.model_output.mkdir(parents=True,exist_ok=True);started=time.time()
    sample=create_sample(args);print(f'Prepared {len(sample):,} fresh S1 records',flush=True)
    x,y,qids,tids=generate_features(args,sample)
    splits=np.asarray([r['split'] for r in sample]);pair_splits=splits[qids];train=pair_splits=='training';tune=pair_splits=='threshold_tuning';test=pair_splits=='evaluation'
    tune_sample=[r for r in sample if r['split']=='threshold_tuning'];evaluation=[r for r in sample if r['split']=='evaluation']
    with (args.output/'sample_candidate_pairs.tsv').open() as f:candidates={r['source1_entity_id']:list(filter(None,r['candidate_entity_ids'].split(','))) for r in csv.DictReader(f,delimiter='\t')}
    baseline=joblib.load(args.baseline)
    with threadpool_limits(limits=4):base_tune=baseline['model'].predict_proba(x[tune])[:,1]
    baseline_tuning=evaluate(tune_sample,candidates,predict(base_tune,baseline['threshold'],qids[tune],tids[tune],sample))
    precision_target=max(.96,baseline_tuning['all']['precision']);search=[];best=None;best_model=None
    thresholds=list(np.arange(.05,.951,.025))+[.975,.99,.995,.999]
    for trial,config in enumerate(CONFIGS):
        print(f'Training trial {trial+1}/{len(CONFIGS)} on {int(train.sum()):,} pairs',flush=True)
        model=HistGradientBoostingClassifier(**config,early_stopping=False,random_state=args.seed)
        with threadpool_limits(limits=4):model.fit(x[train],y[train]);probabilities=model.predict_proba(x[tune])[:,1]
        for threshold in thresholds:
            predictions=predict(probabilities,float(threshold),qids[tune],tids[tune],sample)
            metric=evaluate(tune_sample,candidates,predictions)['all']
            result=dict(trial=trial,config=config,threshold=float(threshold),**metric);search.append(result)
            if metric['precision']>=precision_target and (best is None or (metric['macro_f05'],float(threshold))>(best['macro_f05'],best['threshold'])):
                best=result;best_model=model
        print(f"Best tuning macro F0.5 so far: {best['macro_f05'] if best else 'no precision-qualified threshold'}",flush=True)
    if best is None:raise RuntimeError('No threshold meets the predeclared precision constraint')
    artifact={'model':best_model,'feature_names':FEATURE_NAMES,'threshold':best['threshold'],'license':'MIT',
              'training_s1_ids':[r['source1_entity_id'] for r in sample if r['split']=='training'],
              'tuning_s1_ids':[r['source1_entity_id'] for r in sample if r['split']=='threshold_tuning']}
    joblib.dump(artifact,args.model_output/'matcher.joblib')
    frozen={'selected':best,'precision_target':precision_target,'training_pairs':int(train.sum()),'positive_training_pairs':int(y[train].sum()),
            'training_entities':args.train_size,'tuning_entities':args.tune_size,'evaluation_entities':args.eval_size,
            'model_sha256':hashlib.sha256((args.model_output/'matcher.joblib').read_bytes()).hexdigest(),
            'baseline_sha256':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),'evaluation_policy':'One fresh paired evaluation after model/threshold freeze.'}
    (args.model_output/'frozen_model.json').write_text(json.dumps(frozen,indent=2)+'\n')
    # Evaluation is reached only after model configuration and threshold are frozen.
    with threadpool_limits(limits=4):
        new_probabilities=best_model.predict_proba(x[test])[:,1];old_probabilities=baseline['model'].predict_proba(x[test])[:,1]
    new_pred=predict(new_probabilities,best['threshold'],qids[test],tids[test],sample)
    old_pred=predict(old_probabilities,baseline['threshold'],qids[test],tids[test],sample)
    report={'frozen':frozen,'baseline_tuning':baseline_tuning,'expanded_evaluation':evaluate(evaluation,candidates,new_pred),
            'baseline_evaluation':evaluate(evaluation,candidates,old_pred),'tuning_search':search,'elapsed_seconds':time.time()-started,
            'limitations':['Fresh labeled evaluation covers US/India only; France accuracy is unknown.',
                           'Training remains sampled rather than all training businesses.', 'Full test predictions have not been generated.']}
    (args.model_output/'expanded_report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,predictions in [('evaluation_matching_results.tsv',new_pred),('baseline_evaluation_matching_results.tsv',old_pred)]:
        with (args.model_output/name).open('w',newline='') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['source1_entity_id','matched_entity_ids'])
            for r in evaluation:writer.writerow([r['source1_entity_id'],','.join(predictions.get(r['source1_entity_id'],[]))])
    print(json.dumps({k:v for k,v in report.items() if k!='tuning_search'},indent=2),flush=True)


if __name__=='__main__':main()

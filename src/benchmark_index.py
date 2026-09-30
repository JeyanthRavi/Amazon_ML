"""Check persistent retrieval against saved lists and measure end-to-end queries."""
import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import joblib
import numpy as np
from improve_blocking import prepare
from persistent_index import TargetIndex, disk_bytes
from fast_retrieval import CachedTargetIndex
from train_matcher import FEATURE_NAMES, features, load_sample


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--samples',type=Path,nargs='+',required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cached',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sample,expected=[],{}
    for directory in args.samples:
        records,candidates=load_sample(directory);sample.extend(records);expected.update(candidates)
    if len(sample)!=len({r['source1_entity_id'] for r in sample}):raise ValueError('Duplicate benchmark S1 IDs')
    artifact=joblib.load(args.model)
    if FEATURE_NAMES!=artifact['feature_names']:raise ValueError('Feature order differs from saved model')
    index=CachedTargetIndex(args.index) if args.cached else TargetIndex(args.index)
    retrieval_times=[];feature_times=[];ids=[];matrix=[]
    candidate_counts=[];mismatches=[];started=time.time()
    for n,row in enumerate(sample,1):
        eid=row['source1_entity_id'];raw=[eid,row['business_name'],row['business_address'],row['country']]
        before=time.time();candidates=index.candidates(raw,return_records=True);retrieval_times.append(time.time()-before)
        actual=[r[0] for r in candidates]
        if actual!=expected[eid]:mismatches.append(eid)
        before=time.time();query=prepare(raw)
        for target in candidates:
            matrix.append(features(query,target));ids.append((eid,target[0]))
        feature_times.append(time.time()-before);candidate_counts.append(len(candidates))
        if n%100==0: print(f'Benchmarked {n} queries; {time.time()-started:.1f}s',flush=True)
    before=time.time();probabilities=artifact['model'].predict_proba(np.asarray(matrix,dtype=np.float32))[:,1]
    model_seconds=time.time()-before;predictions={}
    for probability,(eid,tid) in zip(probabilities,ids):
        if probability>=artifact['threshold']:predictions.setdefault(eid,[]).append(tid)
    elapsed=time.time()-started
    report={'sample_size':len(sample),'cached':args.cached,'target_rows':index.state['count'],'index_bytes':disk_bytes(args.index),
            'index_build_seconds':index.state['build_seconds'],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'candidate_equivalence_mismatch_count':len(mismatches),'candidate_equivalence_mismatches':mismatches,
            'retrieval_seconds':sum(retrieval_times),'retrieval_mean_ms':1000*np.mean(retrieval_times),
            'retrieval_p95_ms':1000*np.quantile(retrieval_times,.95),'feature_seconds':sum(feature_times),
            'model_seconds':model_seconds,'end_to_end_seconds':elapsed,'end_to_end_queries_per_second':len(sample)/elapsed,
            'average_candidates':np.mean(candidate_counts),'p95_candidates':float(np.quantile(candidate_counts,.95)),
            'max_candidates':max(candidate_counts),'projected_1732544_queries_hours':elapsed/len(sample)*1732544/3600,
            'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'limitations':['Projection from sampled training queries; test and France workload may differ.',
                           'Benchmark reuses a process and cache; full inference must use bounded feature/prediction batches.']}
    with (args.output/'benchmark_matching_results.tsv').open('w',newline='') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['source1_entity_id','matched_entity_ids'])
        for row in sample:writer.writerow([row['source1_entity_id'],','.join(predictions.get(row['source1_entity_id'],[]))])
    (args.output/'index_benchmark.json').write_text(json.dumps(report,indent=2)+'\n');index.close()
    print(json.dumps(report,indent=2),flush=True)
    if mismatches:raise ValueError('Persistent retrieval differs from measured blocker')


if __name__=='__main__':main()

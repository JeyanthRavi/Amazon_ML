"""Full-pool retrieval pilot with GPU scoring and CPU probability parity checks."""
import argparse,csv,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from audit_blocking import rows
from expanded_training import initialize,feature_batch
from gpu_matcher import GPUMatcher


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--trees',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--max-entities',type=int,default=1000);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);artifact=joblib.load(args.model);gpu=GPUMatcher(args.trees)
    # Compile and warm CUDA before timing the production path.
    gpu.predict_proba(np.zeros((1,23),dtype=np.float32))
    selected=[]
    for row in rows(args.data/'test/test_source1.tsv'):
        selected.append(row)
        if len(selected)==args.max_entities:break
    stats=dict(rows=0,pairs=0,feature_and_retrieval_seconds=0.,gpu_scoring_seconds=0.,cpu_parity_check_seconds=0.,max_probability_error=0.)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(args.index,)) as pool,(args.output/'candidate_pairs.tsv').open('w') as cf,(args.output/'matching_results.tsv').open('w') as mf:
        cw=csv.writer(cf,delimiter='\t',lineterminator='\n');mw=csv.writer(mf,delimiter='\t',lineterminator='\n')
        cw.writerow(['source1_entity_id','candidate_entity_ids']);mw.writerow(['source1_entity_id','matched_entity_ids'])
        for offset in range(0,len(selected),args.workers*100):
            frame=selected[offset:offset+args.workers*100];batches=[frame[i:i+100] for i in range(0,len(frame),100)]
            before=time.perf_counter();result=[r for batch in pool.map(feature_batch,batches) for r in batch]
            stats['feature_and_retrieval_seconds']+=time.perf_counter()-before
            matrix=np.concatenate([r[2] for r in result]);before=time.perf_counter()
            probabilities=gpu.predict_proba(matrix)[:,1];stats['gpu_scoring_seconds']+=time.perf_counter()-before
            before=time.perf_counter()
            with threadpool_limits(limits=2):reference=artifact['model'].predict_proba(matrix)[:,1] if len(matrix) else np.array([])
            stats['cpu_parity_check_seconds']+=time.perf_counter()-before
            error=float(np.max(abs(probabilities-reference))) if len(matrix) else 0.
            assert error<1e-12 and np.array_equal(probabilities>=artifact['threshold'],reference>=artifact['threshold'])
            stats['max_probability_error']=max(stats['max_probability_error'],error)
            cursor=0
            for eid,ids,features in result:
                scores=probabilities[cursor:cursor+len(ids)];cursor+=len(ids)
                cw.writerow([eid,','.join(ids)]);mw.writerow([eid,','.join(t for t,s in zip(ids,scores) if s>=artifact['threshold'])])
                stats['rows']+=1;stats['pairs']+=len(ids)
            print(f"GPU pilot: {stats['rows']} test businesses",flush=True)
    stats['pipeline_seconds_excluding_cpu_check']=stats['feature_and_retrieval_seconds']+stats['gpu_scoring_seconds']
    stats['queries_per_second']=stats['rows']/stats['pipeline_seconds_excluding_cpu_check']
    stats['projected_full_test_hours']=1732544/stats['queries_per_second']/3600
    stats.update(status='PASS',cpu_gpu_decision_equivalence=True,partial_test_output=True,workers=args.workers)
    (args.output/'benchmark.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()

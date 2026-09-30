"""Batched, resumable inference; export exactly the pairs passed to the matcher."""
import argparse
import hashlib
import json
import os
import shutil
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import joblib
import numpy as np
from persistent_index import parse
from fast_retrieval import CachedTargetIndex
from improve_blocking import prepare
from train_matcher import FEATURE_NAMES, features

WORKER_INDEX=None
WORKER_ARTIFACT=None
WORKER_THREADS=None


def process_batch(index,artifact,batch):
    matrix,pairs,candidate_lists=[],[],{}
    for row in batch:
        eid=row[0];query=prepare(row);targets=index.candidates(row,return_records=True)
        candidate_lists[eid]=[t[0] for t in targets]
        for target in targets:matrix.append(features(query,target));pairs.append((eid,target[0]))
    probabilities=artifact['model'].predict_proba(np.asarray(matrix,dtype=np.float32))[:,1] if matrix else []
    predictions={}
    for probability,(eid,tid) in zip(probabilities,pairs):
        if probability>=artifact['threshold']:predictions.setdefault(eid,[]).append(tid)
    return [(row[0],candidate_lists[row[0]],predictions.get(row[0],[])) for row in batch]


def initialize_worker(index_path,model_path):
    global WORKER_INDEX,WORKER_ARTIFACT,WORKER_THREADS
    from threadpoolctl import threadpool_limits
    WORKER_THREADS=threadpool_limits(limits=1)
    WORKER_INDEX=CachedTargetIndex(index_path)
    WORKER_ARTIFACT=joblib.load(model_path)


def worker_batch(batch):
    return process_batch(WORKER_INDEX,WORKER_ARTIFACT,batch)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.output/'inference_checkpoint.json'
    artifact=joblib.load(args.model)
    if artifact['feature_names']!=FEATURE_NAMES:raise ValueError('Model feature order changed')
    source=args.data/f'{args.split}/{args.split}_source1.tsv'
    identity={'source_path':str(source.resolve()),'source_bytes':source.stat().st_size,'source_mtime_ns':source.stat().st_mtime_ns,
              'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'index_path':str(args.index.resolve()),
              'threshold':artifact['threshold'],
              'pipeline_sha256':hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in
                  ['run_inference.py','fast_retrieval.py','persistent_index.py','improve_blocking.py','train_matcher.py','audit_blocking.py'])).hexdigest()}
    if checkpoint.exists():
        if not args.resume:raise ValueError('Existing output: use --resume or a new directory')
        state=json.loads(checkpoint.read_text())
        if state['identity']!=identity:raise ValueError('Inference inputs changed since checkpoint')
        if state['complete']:return state
    else:
        if args.resume:raise ValueError('No checkpoint to resume')
        if any((args.output/name).exists() for name in ['candidate_pairs.tsv','matching_results.tsv']):
            raise ValueError('Output exists without checkpoint; choose a new directory')
        state={'identity':identity,'rows':0,'input_position':0,'candidate_position':0,'matching_position':0,'complete':False,'elapsed_seconds':0}
    index=CachedTargetIndex(args.index)
    if any(f'/{args.split}/' not in s['path'] for s in index.state['sources']):raise ValueError('Index split differs from S1 split')
    paths=[args.output/'candidate_pairs.tsv',args.output/'matching_results.tsv'];started=time.time()
    workers=getattr(args,'workers',1)
    stack=ExitStack();stack.callback(index.close)
    executor=stack.enter_context(ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                   initializer=initialize_worker,initargs=(args.index,args.model))) if workers>1 else None
    with stack, source.open('rb') as handle, paths[0].open('r+b' if paths[0].exists() else 'w+b') as candidate_file, paths[1].open('r+b' if paths[1].exists() else 'w+b') as matching_file:
        header=parse(handle.readline())
        if header!=['entity_id','business_name','business_address','country']:raise ValueError('Unexpected S1 header')
        if state['input_position']:
            handle.seek(state['input_position'])
            for f,key in [(candidate_file,'candidate_position'),(matching_file,'matching_position')]:
                if f.seek(0,os.SEEK_END)<state[key]:raise ValueError('Output shorter than checkpoint')
                f.truncate(state[key]);f.seek(state[key])
        else:
            candidate_file.write(b'source1_entity_id\tcandidate_entity_ids\n')
            matching_file.write(b'source1_entity_id\tmatched_entity_ids\n')
        while True:
            if shutil.disk_usage(args.output).free<args.min_free_gib*1024**3:raise RuntimeError('Free-space floor reached; committed batches remain resumable')
            batch=[]
            frame_size=args.batch_size*workers
            count=min(frame_size,args.max_entities-state['rows']) if args.max_entities else frame_size
            if count<=0:break
            for _ in range(count):
                raw=handle.readline()
                if not raw:break
                batch.append(parse(raw))
            chunks=[batch[i:i+args.batch_size] for i in range(0,len(batch),args.batch_size)]
            results=executor.map(worker_batch,chunks) if executor else [process_batch(index,artifact,batch)]
            for result in results:
                for eid,candidate_ids,matching_ids in result:
                    candidate_file.write((eid+'\t'+','.join(candidate_ids)+'\n').encode())
                    matching_file.write((eid+'\t'+','.join(matching_ids)+'\n').encode())
            for f in (candidate_file,matching_file):f.flush();os.fsync(f.fileno())
            state['rows']+=len(batch);state['input_position']=handle.tell()
            state['candidate_position']=candidate_file.tell();state['matching_position']=matching_file.tell()
            state['complete']=len(batch)<count
            state['elapsed_seconds']+=time.time()-started;started=time.time()
            temporary=checkpoint.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,checkpoint)
            if state['rows']%1000==0:print(f"Committed {state['rows']:,} S1 rows",flush=True)
            if state['complete']:break
    state['workers']=workers
    print(json.dumps(state,indent=2),flush=True);return state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--split',choices=['train','test'],default='test');parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--batch-size',type=int,default=100)
    parser.add_argument('--max-entities',type=int,default=0);parser.add_argument('--min-free-gib',type=float,default=1.5)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.batch_size<1 or args.max_entities<0 or args.workers<1:parser.error('Invalid batch size/entity limit/worker count')
    run(args)


if __name__=='__main__':main()

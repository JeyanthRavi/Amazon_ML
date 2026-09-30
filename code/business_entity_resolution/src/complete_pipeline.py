"""Run remaining stages consecutively; write durable progress and failure status."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/final-run'
STATUS=REPORT/'status.json'

def status(stage,**values):
    REPORT.mkdir(parents=True,exist_ok=True)
    payload=dict(stage=stage,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),**values)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(STATUS)
    print(json.dumps(payload),flush=True)

def run(name,argv):
    status(name,log=str(REPORT/(name+'.log')))
    with (REPORT/(name+'.log')).open('a') as log:
        subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--validator',type=Path,required=True)
    parser.add_argument('--wait-index-pid',type=int,default=0);args=parser.parse_args()
    index=ROOT/'cache/test-index';model=ROOT/'models/expanded/matcher.joblib';output=ROOT/'output'
    try:
        if args.wait_index_pid:
            status('building_test_index',index_pid=args.wait_index_pid,log=str(ROOT/'reports/test-index-build.log'))
            while True:
                try:os.kill(args.wait_index_pid,0)
                except ProcessLookupError:break
                time.sleep(15)
        # Idempotent build resumes compatible checkpoints or returns if complete.
        run('test_index',[ROOT/'src/persistent_index.py','--data',args.data,'--split','test','--index',index,'--budget-gib',3])
        build=json.loads((index/'build_report.json').read_text())
        assert build['complete'] and build['count']==9969589
        # Generated training cache is reproducible. Release only if projections threaten the free-space floor.
        if shutil.disk_usage(ROOT).free<4.5*1024**3 and (ROOT/'cache/train-index').exists():
            shutil.copy2(ROOT/'cache/train-index/build_report.json',REPORT/'training_index_build_report.json')
            shutil.rmtree(ROOT/'cache/train-index')
            print('Released generated training index cache to reserve space for outputs and ZIP',flush=True)
        timings={}
        for workers in (4,8):
            pilot=ROOT/f'reports/test-pilot-{workers}'
            command=[ROOT/'src/run_inference.py','--data',args.data,'--split','test','--index',index,'--model',model,
                     '--output',pilot,'--workers',workers,'--batch-size',100,'--max-entities',1000]
            if (pilot/'inference_checkpoint.json').exists():command.append('--resume')
            run(f'pilot_{workers}_workers',command)
            cp=json.loads((pilot/'inference_checkpoint.json').read_text());assert cp['rows']==1000
            timings[workers]=cp['elapsed_seconds']
        # Both worker counts must reproduce exactly the same actual model inputs and outputs.
        for name in ('matching_results.tsv','candidate_pairs.tsv'):
            assert (ROOT/'reports/test-pilot-4'/name).read_bytes()==(ROOT/'reports/test-pilot-8'/name).read_bytes()
        workers=min(timings,key=timings.get)
        (REPORT/'test_throughput.json').write_text(json.dumps(dict(pilot_rows=1000,seconds_by_workers=timings,chosen_workers=workers,
            exact_pilot_output_equivalence=True,projected_inference_hours=timings[workers]/1000*1732544/3600),indent=2)+'\n')
        command=[ROOT/'src/run_inference.py','--data',args.data,'--split','test','--index',index,'--model',model,'--output',output,
                 '--workers',workers,'--batch-size',100]
        if (output/'inference_checkpoint.json').exists():command.append('--resume')
        run('full_test_inference',command)
        run('validation_and_package',[ROOT/'src/finalize_submission.py','--data',args.data,'--output',output,
                                     '--validator',args.validator,'--report',REPORT])
        checklist=ROOT/'CHECKLIST.md';s=checklist.read_text()
        for text in ['Build the separate test S2/S3 index after final model selection and budget check.',
                     'Generate candidates for every test S1 ID, including France and empty candidate lists.',
                     'Run the matcher on exactly the exported candidates.',
                     'Write `output/candidate_pairs.tsv` and `output/matching_results.tsv` with exact headers.',
                     'Verify one row per test S1 ID, valid S2/S3 IDs, no duplicates, and matches contained in candidates.',
                     'Run the supplied `utils/validate_submission.py` and obtain PASS.',
                     'Fill `Documentation_template.md` with measured results and limitations.',
                     'Include all source under `code/business_entity_resolution/src/`, exact run instructions, and pinned dependencies.',
                     'Prepare the leaderboard file and complete submission ZIP for user review/upload.',
                     'Reproduce outputs from the packaged pipeline before building the final ZIP.']:
            s=s.replace('- [ ] '+text,'- [x] '+text)
        checklist.write_text(s)
        status('complete',**json.loads((REPORT/'package_manifest.json').read_text()))
    except Exception as exc:
        status('failed',error=str(exc),recovery='Inspect the stage log; fix the cause and rerun this driver. Inference resumes committed outputs automatically.')
        raise
if __name__=='__main__':main()

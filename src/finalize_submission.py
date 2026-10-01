"""Validate all output rows in bounded chunks, then package the frozen pipeline."""
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(condition, message):
    """Submission gates must also run under python -O."""
    if not condition:
        raise ValueError(message)


def validate(data,output,validator,report):
    spec=importlib.util.spec_from_file_location('official_validator',validator)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    targets=set()
    for n in (2,3):
        with (data/f'test/test_source{n}.tsv').open() as f:
            next(f)
            for line in f:targets.add(line.split('\t',1)[0])
    stats=Counter();countries=Counter();warnings=set()
    with tempfile.TemporaryDirectory(prefix='amazon-validate-') as temporary:
        tmp=Path(temporary)
        with (data/'test/test_source1.tsv').open() as source,(output/'candidate_pairs.tsv').open() as candidates,(output/'matching_results.tsv').open() as matches,(report/'official_validation.log').open('w') as log:
            readers=[csv.reader(f,delimiter='\t') for f in (source,candidates,matches)]
            require(next(readers[0])==['entity_id','business_name','business_address','country'], 'Invalid source header')
            require(next(readers[1])==official.CANDIDATE_HEADER, 'Invalid candidate header')
            require(next(readers[2])==official.MATCHING_HEADER, 'Invalid matching header')
            exhausted=False
            while not exhausted:
                with (tmp/'test_source1.tsv').open('w') as sf,(tmp/'candidate_pairs.tsv').open('w') as cf,(tmp/'matching_results.tsv').open('w') as mf:
                    writers=[csv.writer(f,delimiter='\t',lineterminator='\n') for f in (sf,cf,mf)]
                    for w,h in zip(writers,[['entity_id'],official.CANDIDATE_HEADER,official.MATCHING_HEADER]):w.writerow(h)
                    chunk=0
                    for _ in range(5000):
                        s=next(readers[0],None)
                        if s is None:exhausted=True;break
                        c=next(readers[1],None);m=next(readers[2],None)
                        require(len(s)==4 and c is not None and m is not None and len(c)==len(m)==2, 'Invalid or missing output row')
                        require(s[0]==c[0]==m[0], 'Coverage/order mismatch')
                        ci=c[1].split(',') if c[1] else [];mi=m[1].split(',') if m[1] else []
                        require(len(ci)==len(set(ci))<=100 and len(mi)==len(set(mi)), 'Duplicate IDs or candidate cap exceeded')
                        require(all(i.startswith(('S2-','S3-')) and i in targets for i in ci), 'Invalid target ID')
                        require(set(mi)<=set(ci),'Match outside candidate list')
                        writers[0].writerow([s[0]]);writers[1].writerow(c);writers[2].writerow(m)
                        chunk+=1;stats['rows']+=1;stats['candidate_pairs']+=len(ci);stats['matched_pairs']+=len(mi)
                        stats['empty_candidates']+=not ci;stats['empty_matches']+=not mi;countries[s[3]]+=1
                if chunk:
                    with redirect_stdout(log):errors,ws=official.validate(str(tmp/'matching_results.tsv'),str(tmp/'candidate_pairs.tsv'),str(tmp))
                    require(not errors, str(errors))
                    # ID existence is checked against every complete test target above.
                    unexpected=[w for w in ws if not w.startswith('ID-existence check is OFF')]
                    require(not unexpected, str(unexpected))
                    warnings.update(ws);stats['official_chunks']+=1
                    print(f"Validated {stats['rows']:,} output rows",flush=True)
            require(next(readers[1],None) is None and next(readers[2],None) is None,'Extra output rows')
    result=dict(status='PASS',**stats,countries=dict(countries),valid_target_ids=len(targets),
                official_validator_mode='Unmodified supplied validate() on all rows in 5000-row chunks; avoids retaining all candidate sets in RAM.',
                strict_checks='Complete S1 source-order coverage; actual target ID existence; duplicate-free lists; candidate cap; matches subset candidates.',
                matching_sha256=digest(output/'matching_results.tsv'),candidate_sha256=digest(output/'candidate_pairs.tsv'))
    (report/'final_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def package(root,data,output,report,stats):
    submission=root/'submission';submission.mkdir(exist_ok=True)
    team=json.loads((submission/'team.json').read_text())
    team_name=team['team_name'].strip();members=team['team_members']
    require(bool(team_name) and '/' not in team_name and '\\' not in team_name and team_name not in ('.','..'), 'Invalid team name')
    require(bool(members) and all(isinstance(m,str) and m.strip() for m in members), 'Team members missing')
    document=f'''# ML Challenge 2026: Business Entity Resolution

**Team Name:** {team_name}

**Team Members:** {' & '.join(members)}

**Package preparation date:** {date.today().isoformat()}

## 1. Executive Summary
A same-country blocker retrieves at most 100 targets per reference business. A boosted-tree classifier scores 23 name/address features and returns zero, one, or multiple matches. Only the supplied challenge data is used.

## 2. Methodology
Training has 2,206,821 reference businesses and 10,320,219 targets; test has 1,732,544 reference businesses including 259,452 in France. Names and addresses contain typos, abbreviations, reordered tokens and missing components. Country is an open string label. Accent normalization preserves non-Latin script marks; address aliases and common legal suffix removal improve retrieval. No external business lookup, geocoding or augmentation is used.

Approach: blocking plus classifier. Contentless SQLite FTS postings and packed source-file byte locators avoid duplicating target text and support a bounded-memory, resumable pipeline.

## 3. Candidate Generation
Keys cover normalized names, legal-stripped compact names, informative name tokens and pairs, name character grams, address token pairs, numeric/name and numeric/address combinations, and exact address signatures. All keys include country. Buckets exceeding 100 records are dropped. Candidates are deduplicated, cheaply reduced to 500, and ranked to 100 using name/address character and token similarity with numeric overlap. Exported candidates are exactly the records scored by the classifier.

Full test candidate pairs: **{stats['candidate_pairs']:,}**. Fresh labeled evaluation blocking recall: **95.63%**. Retrieval does lose true links; no claim of perfect recall is made.

## 4. Matching Model
The 23 features cover name character similarity, token overlap/containment, compact equality, character grams and lengths; address token/character similarity, numbers and postcode-like tokens; missing addresses, country agreement and script indicators. Raw IDs and categorical countries are not inputs.

HistGradientBoostingClassifier: 300 boosting iterations, at most 31 leaves per tree, minimum leaf size 40, L2 regularization 2. Training uses 6,000 S1 groups, 494,572 retrieved pairs and 19,961 positives. Original trained artifact and code are MIT licensed; dependencies retain upstream licenses. This tree model is far below the 8-billion-parameter ceiling.

A seeded, earlier-sample-excluded split reserves 1,000 tuning and 1,000 evaluation groups. Three predeclared configurations and a threshold grid are selected using tuning macro F0.5 subject to precision at least max(96%, original baseline tuning precision). Selected threshold: **0.775**. Selection is frozen before evaluation; no refit uses evaluation labels.

## 5. Results and Error Analysis
Fresh paired evaluation macro F0.5: **0.8766** versus baseline **0.8528**. Precision: **96.18%**; recall: **77.02%**. India macro F0.5: **0.8288**; US: **0.9105**. Five of 54 singleton businesses receive false matches. Common-name/address similarity can produce false merges; truncated names, script/format variation and capped/common buckets cause missed links. Of 3,429 true links, blocking misses 150 and the matcher rejects another 638 retrieved true links.

France has no supplied training labels; France accuracy is unknown. These validation results are not leaderboard scores. Full test output contains **{stats['rows']:,}** rows, **{stats['matched_pairs']:,}** predicted links, and **{stats['empty_matches']:,}** empty match lists. Full output validation: **PASS**, including target-ID existence and match/candidate consistency.

## 6. Conclusion
Expanded training improves both precision and recall on a fresh paired sample. The pipeline preserves singleton and multi-match behavior and covers France through country-generic retrieval. Remaining accuracy limitations include lost blocking links and unmeasured France generalization.

## Appendix A: Code Artefacts
All Python source is under `code/business_entity_resolution/src/`, with pinned requirements, MIT license, a frozen model and exact reproduction instructions. Build a test index with `persistent_index.py`, then regenerate both output files with `run_inference.py`. `finalize_submission.py` validates and packages results. Supplied data is not redistributed.

## Appendix B: Validation
The unmodified supplied validator's `validate()` function checks both files in 5,000-row chunks to keep RAM bounded. An additional streaming pass verifies complete source-order coverage, every candidate target ID against the full test target pool, list uniqueness and strict match-subset constraints. Validation reports and SHA256 hashes are included under `code/business_entity_resolution/reports/`.
'''
    (submission/'Documentation_template.md').write_text(document)
    readme='''# Reproduce the submission

Requires Python 3.9+ on macOS or Linux with SQLite FTS5. The run was prepared on macOS arm64 with Python 3.9.6. Supply the original challenge dataset; do not move source files while an index or inference run is active.

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Set this to the original supplied dataset directory (contains train/ and test/).
DATA=/absolute/path/to/student_resource/dataset
.venv/bin/python src/persistent_index.py --data "$DATA" --split test --index cache/test-index --budget-gib 3
.venv/bin/python src/run_inference.py --data "$DATA" --split test --index cache/test-index --model models/expanded/matcher.joblib --output ../../reproduced-output --workers 4 --batch-size 100
```

The frozen model is included, so retraining is unnecessary to reproduce outputs. The new output directory avoids overwriting the packaged originals. Compare regenerated TSV SHA256 hashes with reports/final_validation.json. Checkpoints support restarting the same inference command with `--resume`; code, model, input paths/metadata and index must remain unchanged. Index construction automatically resumes compatible checkpoints. Allow approximately 3 GiB for the index, 2 GiB for outputs and at least 1.5 GiB spare disk. Inference may take many hours and varies with hardware and workload.

Source also includes audits, blocker/model experiments, and verification. Original experiment samples and all supplied data are excluded from the archive; exact experiment metrics and model selection hashes are included in reports. To retrain, generate samples with the audit/blocking scripts and reproduce the documented exclusions before running expanded_training.py. The included trained artifact is the authoritative inference input.

MIT license covers original source and trained model. Third-party packages and supplied challenge data retain their own licenses. Team identity is filled in the root methodology document.
'''
    archive=submission/f'{team_name}_submission.zip';temporary=archive.with_suffix('.zip.tmp')
    prefix='code/business_entity_resolution/'
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name in ('matching_results.tsv','candidate_pairs.tsv'):z.write(output/name,'output/'+name)
        z.write(submission/'Documentation_template.md','Documentation_template.md')
        z.writestr(prefix+'README.md',readme)
        z.write(submission/'team.json',prefix+'submission/team.json')
        for name in ('LICENSE','requirements.txt'):z.write(root/name,prefix+name)
        for path in sorted((root/'src').glob('*.py')):z.write(path,prefix+'src/'+path.name)
        z.write(root/'models/expanded/matcher.joblib',prefix+'models/expanded/matcher.joblib')
        for path in (root/'models/expanded').glob('*.json'):z.write(path,prefix+'models/expanded/'+path.name)
        for path in [report/'final_validation.json',root/'reports/expanded/verification.json',root/'reports/EXPANDED_TRAINING.md']:
            z.write(path,prefix+'reports/'+path.name)
        for name in ('FINISHED_STATUS.md','clean_environment_verification.json'):
            path=root/'reports'/name
            if path.exists():z.write(path,prefix+'reports/'+name)
    with zipfile.ZipFile(temporary) as z:require(z.testzip() is None,'ZIP integrity failure')
    # Reproduce a bounded output prefix from the actual archived code and model.
    with tempfile.TemporaryDirectory(prefix='amazon-package-reproduce-') as work:
        work=Path(work)
        with zipfile.ZipFile(temporary) as z:
            for member in z.namelist():
                if member.startswith(prefix):z.extract(member,work)
        packaged=work/prefix;smoke=work/'smoke'
        subprocess.run([sys.executable,str(packaged/'src/run_inference.py'),'--data',str(data),
            '--split','test','--index',str(root/'cache/test-index'),'--model',str(packaged/'models/expanded/matcher.joblib'),
            '--output',str(smoke),'--workers','2','--batch-size','10','--max-entities','30','--min-free-gib','0.1'],check=True,stdout=subprocess.DEVNULL)
        for name in ('candidate_pairs.tsv','matching_results.tsv'):
            with (output/name).open('rb') as f:expected=b''.join(f.readline() for _ in range(31))
            require((smoke/name).read_bytes()==expected,'Packaged pipeline reproduction mismatch')
    temporary.replace(archive)
    result=dict(status='READY',archive=str(archive.resolve()),archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,
                output_validation='PASS',packaged_reproduction_rows=30,packaged_reproduction='PASS',team_identity_fields='Complete',team_name=team_name,team_members=members,uploaded=False,zip_integrity='PASS')
    (report/'package_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validator',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--package-only',action='store_true',help='Reuse prior full validation only after checking both output SHA256 hashes')
    args=parser.parse_args();args.report.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    checkpoint=json.loads((args.output/'inference_checkpoint.json').read_text())
    require(checkpoint['complete'] and checkpoint['rows']==1732544,'Full test inference is incomplete')
    if args.package_only:
        stats=json.loads((args.report/'final_validation.json').read_text())
        require(stats['status']=='PASS','Previous validation did not pass')
        for name,key in [('matching_results.tsv','matching_sha256'),('candidate_pairs.tsv','candidate_sha256')]:
            require(digest(args.output/name)==stats[key], 'Output changed since full validation: '+name)
    else:
        stats=validate(args.data,args.output,args.validator,args.report)
    require(stats['rows']==1732544 and stats['countries']['France']==259452, 'Incomplete test coverage')
    package(root,args.data,args.output,args.report,stats)
if __name__=='__main__':main()

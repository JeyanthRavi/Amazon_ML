"""Full supplied ID/reference audit; bounded to one dataset split at a time."""
import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path
from audit_blocking import rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.time();report={'issues':[],'source_counts':{},'shared_positive_targets':0,'country_disagreement_links':0}
    for split in ('train','test'):
        reference,targets={},{}
        for source in (1,2,3):
            seen=reference if source==1 else targets;count=0
            for count,row in enumerate(rows(args.data/f'{split}/{split}_source{source}.tsv'),1):
                eid=row[0]
                if not eid.startswith(f'S{source}-'):
                    if len(report['issues'])<20:report['issues'].append(f'{split}: wrong source prefix {eid}')
                if eid in seen:
                    if len(report['issues'])<20:report['issues'].append(f'{split}: duplicate ID {eid}')
                seen[eid]=sys.intern(row[3])
            report['source_counts'][f'{split}_source{source}']=count
            print(f'Checked {split} S{source}: {count:,}',flush=True)
        if split=='train':
            remaining=set(reference);owners={};label_count=link_count=0
            for label_count,(eid,text) in enumerate(rows(args.data/'train/train_ground_truth.tsv'),1):
                if eid not in remaining:
                    if len(report['issues'])<20:report['issues'].append(f'Unknown or duplicate ground-truth S1 ID {eid}')
                else:remaining.remove(eid)
                matched=text.split(',') if text else []
                if len(set(matched))!=len(matched):
                    if len(report['issues'])<20:report['issues'].append(f'Duplicate labeled links for {eid}')
                for target in matched:
                    link_count+=1
                    if target not in targets:
                        if len(report['issues'])<20:report['issues'].append(f'Absent labeled target {target}')
                        continue
                    if target in owners and owners[target]!=eid:report['shared_positive_targets']+=1
                    else:owners[target]=eid
                    if reference.get(eid)!=targets[target]:report['country_disagreement_links']+=1
            if remaining:report['issues'].append(f'{len(remaining)} training S1 IDs missing ground truth')
            report['ground_truth_rows']=label_count;report['ground_truth_links']=link_count
            del remaining,owners
        del reference,targets,seen;gc.collect()
    report['status']='PASS' if not report['issues'] else 'FAIL'
    report['elapsed_seconds']=time.time()-started;report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if report['issues']:raise SystemExit(1)


if __name__=='__main__':main()

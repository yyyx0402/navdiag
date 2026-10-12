"""Small curated-source retrieval regression; no model API calls."""
import argparse
import hashlib
from importlib.resources import files
import json
from pathlib import Path

from navdiag.literature import reading_list


def evaluate(output):
    resource=files('navdiag').joinpath('resources')
    cases=json.loads(resource.joinpath('reading_cases.json').read_text(encoding='utf-8'))
    report={'scope':'Developer-authored source lookup smoke/regression cases, not blind user questions or LLM answer quality.',
            'model_api_requests':0, 'library_sha256':hashlib.sha256(resource.joinpath('literature.json').read_bytes()).hexdigest(),
            'cases_sha256':hashlib.sha256(resource.joinpath('reading_cases.json').read_bytes()).hexdigest(), 'modes':{}}
    for mode in ['sparse','hybrid']:
        results=[]
        for case in cases:
            result=reading_list(case['query'],mode=mode)
            ids=[paper['id'] for paper in result['papers']]
            hit=not ids if case.get('expect_empty') else case['expected'] in ids
            reciprocal=1/(ids.index(case['expected'])+1) if case.get('expected') in ids else 0
            results.append({**case,'returned':ids,'passed':hit,'reciprocal_rank':reciprocal,
                            'actual_method':result['retrieval_method'],'inferred_topics':result['inferred_topics']})
        positive=[row for row in results if not row.get('expect_empty')]
        negative=[row for row in results if row.get('expect_empty')]
        report['modes'][mode]={'positive_hit_at_5':sum(row['passed'] for row in positive), 'positive_cases':len(positive),
            'mrr_at_5':sum(row['reciprocal_rank'] for row in positive)/len(positive),
            'negative_rejections':sum(row['passed'] for row in negative),'negative_cases':len(negative),'results':results}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({mode:{key:value for key,value in row.items() if key!='results'}
                      for mode,row in report['modes'].items()},ensure_ascii=False))
    for mode,row in report['modes'].items():
        print(mode+' failures: '+', '.join(case['id'] for case in row['results'] if not case['passed']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('docs/experiments/reading_retrieval_report.json'))
    evaluate(parser.parse_args().out)

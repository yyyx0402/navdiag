"""Repeatable tool/citation/retrieval checks, distinct from semantic LLM grading."""
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from navdiag.agent import LLMConfig, ask


def evaluate_agent(artifacts: Path, output: Path, config: LLMConfig | None = None,
                   cases_path: Path | None = None) -> dict:
    config = config or LLMConfig()
    source = cases_path or files('navdiag').joinpath('resources/agent_cases.json')
    cases = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases or len(cases) > 100:
        raise ValueError('Evaluation needs 1-100 cases')
    results = []
    for case in cases:
        if not {'id', 'question', 'required_tool'} <= set(case):
            raise ValueError('Every case needs id, question and required_tool')
        try:
            answer = ask(case['question'], artifacts, config)
            evidence = answer['evidence']
            tools = {item['tool'] for item in evidence}
            notes = {note['id'] for item in evidence if item['tool'] == 'search_knowledge'
                     for note in item['payload']['notes']}
            checks = {
                'required_tool_called': case['required_tool'] in tools,
                'has_evidence_citations': answer['citation_check']['has_evidence_citations'],
                'no_unknown_citations': not answer['citation_check']['unknown_citations'],
                'all_evidence_has_provenance': all(item['data_kind'] in {'real', 'synthetic'} for item in evidence),
            }
            if case.get('knowledge_id'):
                checks['expected_note_in_top_3'] = case['knowledge_id'] in notes
            results.append({'id': case['id'], 'question': case['question'], 'checks': checks,
                            'passed': all(checks.values()), 'result': answer})
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            results.append({'id': case['id'], 'question': case['question'], 'passed': False, 'error': str(error)})
    latencies = [row['result']['elapsed_s'] for row in results if 'result' in row]
    passed = sum(row['passed'] for row in results)
    report = {
        'backend': config.provider, 'model': config.model or 'deterministic_tool_baseline',
        'cases': len(results), 'passed': passed, 'pass_rate': passed / len(results),
        'latency_p50_s': float(np.median(latencies)) if latencies else None,
        'latency_p95_s': float(np.percentile(latencies, 95)) if latencies else None,
        'scope': 'Tool routing, citation IDs, evidence provenance and optional note retrieval only.',
        'limitations': ('This is an offline deterministic baseline evaluation, not an LLM evaluation. '
                        if config.provider == 'offline' else '') +
                       'These checks do not grade answer correctness, numerical claims, root-cause reasoning or user usefulness.',
        'results': results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report

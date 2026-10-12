"""Repeatable tool/citation/retrieval checks, distinct from semantic LLM grading."""
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from navdiag.agent import LLMConfig, ask


def evaluate_agent(artifacts: Path, output: Path, config: LLMConfig | None = None,
                   cases_path: Path | None = None, nlos_artifacts: Path | None = None,
                   generalization_artifacts: Path | None = None) -> dict:
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
            answer = ask(case['question'], artifacts, config, nlos_artifacts=nlos_artifacts,
                         generalization_artifacts=generalization_artifacts, task_context=case.get('task_context'))
            evidence = answer['evidence']
            tools = {item['tool'] for item in evidence}
            notes = {note['id'] for item in evidence if item['tool'] == 'search_knowledge'
                     for note in item['payload']['notes']}
            checks = {
                'required_tool_called': case['required_tool'] in tools,
                'has_evidence_citations': answer['citation_check']['has_evidence_citations'],
                'no_unknown_citations': not answer['citation_check']['unknown_citations'],
                'all_evidence_has_provenance': all(
                    item['data_kind'] in {'real', 'synthetic'} or
                    (item['data_kind'] == 'knowledge' and item['tool'] == 'search_knowledge' and
                     all({'id', 'source_url', 'source_type'} <= set(note) for note in item['payload']['notes']))
                    for item in evidence),
            }
            if config.provider != 'offline' and 'answer_check' in answer:
                checks['explicit_recorded_values_match'] = answer['answer_check']['needs_review'] == 0
                checks['bounded_comparisons_consistent'] = not answer['answer_check']['comparison_warnings']
            if case.get('knowledge_id'):
                checks['expected_note_in_top_5'] = case['knowledge_id'] in notes
            results.append({'id': case['id'], 'question': case['question'], 'checks': checks,
                            'passed': all(checks.values()), 'result': answer})
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            results.append({'id': case['id'], 'question': case['question'], 'passed': False, 'error': str(error)})
    latencies = [row['result']['elapsed_s'] for row in results if 'result' in row]
    passed = sum(row['passed'] for row in results)
    report = {
        'backend': config.provider, 'answer_mode': config.answer_mode,
        'model': config.model or 'deterministic_tool_baseline',
        'cases': len(results), 'passed': passed, 'pass_rate': passed / len(results),
        'latency_p50_s': float(np.median(latencies)) if latencies else None,
        'latency_p95_s': float(np.percentile(latencies, 95)) if latencies else None,
        'scope': 'Tool routing, citation IDs, evidence provenance, optional retrieval and bounded explicit metric checks.',
        'limitations': ('This is an offline deterministic baseline evaluation, not an LLM evaluation. '
                        if config.provider == 'offline' else '') +
                       'These checks do not grade full answer correctness, every numerical claim, root-cause reasoning or user usefulness. Unparsed prose is not verified.',
        'results': results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report

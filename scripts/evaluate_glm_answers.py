"""Prepare/execute a bounded GLM answer audit; semantic scoring remains separate."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import time

from navdiag import agent
from navdiag.agent_context import evidence_context
from navdiag.llm_connections import ConnectionStore, connection_path


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def context_hash(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(artifacts, nlos, generalization, case_ids=None):
    cases = json.loads(files('navdiag').joinpath('resources/glm_quality_cases.json').read_text(encoding='utf-8'))
    if case_ids:
        unknown = set(case_ids) - {case['id'] for case in cases}
        if unknown:
            raise ValueError(f'Unknown quality cases: {sorted(unknown)}')
        cases = [case for case in cases if case['id'] in case_ids]
    prepared = []
    for case in cases:
        tools = agent.EvidenceTools(artifacts, nlos, generalization)
        agent.prepare_evidence(case['question'], tools, task_context=case['task_context'])
        context, stats = evidence_context(case['question'], tools.evidence, case['task_context'])
        prepared.append({**case, 'context': context, 'context_stats': stats,
                         'context_sha256': context_hash(context)})
    return {'status': 'prepared_not_executed', 'created_at': datetime.now(timezone.utc).isoformat(),
            'maximum_model_requests': min(8, sum(not case.get('expect_local_guard') for case in cases)),
            'system_prompt_sha256': hashlib.sha256(agent.SYSTEM_PROMPT.encode()).hexdigest(), 'cases': prepared,
            'evaluation_scope': 'Actual provider answers reviewed against frozen local evidence; local guard scored separately.',
            'limitations': ['Small developer-authored regression set, not a representative or blind benchmark.',
                            'Automated citation/format checks do not establish semantic or numerical correctness.',
                            'Semantic review requires per-case judgments and evidence pointers; do not label assistant review as human review.']}


def execute(plan, config, artifacts, nlos, generalization, output):
    if 'glm' not in config.model.lower():
        raise ValueError('The selected connection must identify the GLM model being evaluated.')
    if config.answer_mode != 'evidence':
        raise ValueError('This bounded evaluation requires single-request evidence mode.')
    report = {key: value for key, value in plan.items() if key != 'cases'}
    report.update(status='running', model=config.model, provider=config.provider,
                  answer_mode=config.answer_mode, results=[], model_requests_attempted=0,
                  settings={'output_limit': config.max_output_tokens, 'timeout_s': config.timeout_s,
                            'reasoning_effort': config.reasoning_effort})
    real_post = agent._post
    active_case, sent_context_sha256 = None, None
    def tracked_post(*args, **kwargs):
        nonlocal sent_context_sha256
        if active_case.get('expect_local_guard'):
            raise RuntimeError('Local guard unexpectedly attempted a model request')
        if report['model_requests_attempted'] >= plan['maximum_model_requests']:
            raise RuntimeError('Evaluation request budget exhausted')
        payload = args[2] if len(args) > 2 else kwargs['payload']
        messages = payload.get('messages', payload.get('input', []))
        actual_context = json.loads(next(message['content'] for message in reversed(messages)
                                         if message['role'] == 'user'))
        sent_context_sha256 = context_hash(actual_context)
        if sent_context_sha256 != active_case['context_sha256']:
            raise RuntimeError('Evidence changed after preparation; no request was sent')
        report['model_requests_attempted'] += 1
        return real_post(*args, **kwargs)
    agent._post = tracked_post
    try:
        for case in plan['cases']:
            active_case, sent_context_sha256 = case, None
            requests_before = report['model_requests_attempted']
            started = time.monotonic()
            try:
                result = agent.ask(case['question'], artifacts, config, nlos_artifacts=nlos,
                                   generalization_artifacts=generalization, task_context=case['task_context'])
                report['results'].append({'id': case['id'], 'question': case['question'], 'criteria': case['criteria'],
                    'result': result, 'model_requests_attempted': report['model_requests_attempted'] - requests_before,
                    'sent_context_sha256': sent_context_sha256,
                    'checks': {'answer_complete': result['answer_complete'],
                        'known_citation_ids': not result['citation_check']['unknown_citations'],
                        'has_citations': result['citation_check']['has_evidence_citations'],
                        'expected_local_guard': (result['generation_skipped'] and report['model_requests_attempted'] == requests_before)
                            if case.get('expect_local_guard') else not result['generation_skipped'],
                        'frozen_context_sent': sent_context_sha256 == case['context_sha256'] if sent_context_sha256 else None},
                    'semantic_review': {'status': 'pending', 'reviewer_kind': None, 'judgments': []}})
                print(json.dumps({'id': case['id'], 'status': 'local_guard' if result['generation_skipped'] else 'answered',
                                  'elapsed_s': round(result['elapsed_s'], 2), 'usage': result['usage']}, ensure_ascii=False), flush=True)
            except (RuntimeError, ValueError) as error:
                report['results'].append({'id': case['id'], 'question': case['question'], 'status': 'failed',
                    'elapsed_s': time.monotonic() - started, 'error': str(error),
                    'error_code': getattr(error, 'code', None), 'model_diagnostics': getattr(error, 'diagnostics', None)})
                report.update(status='stopped_after_failure', finished_at=datetime.now(timezone.utc).isoformat())
                print(json.dumps({'id': case['id'], 'status': 'failed', 'error': str(error)}, ensure_ascii=False), flush=True)
                write_report(output, report)
                return report
            write_report(output, report)
        report.update(status='responses_collected_semantic_review_pending', finished_at=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
        return report
    finally:
        agent._post = real_post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connection')
    parser.add_argument('--execute', action='store_true', help='Explicitly make up to eight real model requests')
    parser.add_argument('--cases', nargs='+', help='Run a subset, for example G03 G05 G07; no automatic retries')
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/navigation_demo'))
    parser.add_argument('--nlos-artifacts', type=Path, default=Path('artifacts/klt_protocols'))
    parser.add_argument('--generalization-artifacts', type=Path, default=Path('artifacts/urbannav_generalization'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/glm_quality_report.json'))
    args = parser.parse_args()
    if args.execute and args.out.exists():
        parser.error('Output already exists; use a new --out path to preserve previous answers.')
    plan = prepare(args.artifacts, args.nlos_artifacts, args.generalization_artifacts, args.cases)
    write_report(args.out.with_name(args.out.stem + '_plan.json'), plan)
    if not args.execute:
        print(f"Prepared {len(plan['cases'])} cases. No model API request was made.")
        return 0
    if not args.connection:
        parser.error('--execute requires a locally saved --connection ID')
    config = replace(ConnectionStore(connection_path()).config(args.connection), answer_mode='evidence',
                     max_output_tokens=8192, timeout_s=120, reasoning_effort='low')
    report = execute(plan, config, args.artifacts, args.nlos_artifacts, args.generalization_artifacts, args.out)
    return 1 if report['status'] == 'stopped_after_failure' else 0


if __name__ == '__main__':
    raise SystemExit(main())

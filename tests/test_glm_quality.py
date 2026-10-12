"""No real vendor calls: check the audit's budget, evidence identity and failure handling."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from navdiag import agent


spec = importlib.util.spec_from_file_location('glm_quality', Path(__file__).parents[1] / 'scripts/evaluate_glm_answers.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def make_plan(guard=False):
    context = {'question': 'HDOP?', 'evidence': []}
    case = {'id': 'fixture', 'question': 'HDOP?', 'criteria': [], 'task_context': 'satellite_nlos',
            'context_sha256': audit.context_hash(context), 'expect_local_guard': guard}
    return {'cases': [case], 'maximum_model_requests': 0 if guard else 1}, context


def response(guard=False):
    return {'answer': '依据 [E1]', 'evidence': [], 'answer_complete': True,
            'citation_check': {'unknown_citations': [], 'has_evidence_citations': True},
            'generation_skipped': guard, 'elapsed_s': 0.1, 'usage': []}


def test_budget_and_post_restoration(monkeypatch, tmp_path):
    plan, context = make_plan()
    plan['cases'] *= 2
    calls = []
    def post(*args):
        calls.append(args)
    def ask(question, artifacts, config, **kwargs):
        agent._post(config, '/chat/completions', {'messages': [{'role': 'user', 'content': json.dumps(context)}]})
        return response()
    monkeypatch.setattr(agent, '_post', post)
    monkeypatch.setattr(agent, 'ask', ask)
    report = audit.execute(plan, agent.LLMConfig(model='glm-fixture', answer_mode='evidence'), None, None, None, tmp_path / 'report.json')
    assert len(calls) == report['model_requests_attempted'] == 1
    assert report['status'] == 'stopped_after_failure'
    assert report['results'][0]['checks']['frozen_context_sent'] is True
    assert agent._post is post


def test_changed_evidence_is_rejected_before_vendor_call(monkeypatch, tmp_path):
    plan, context = make_plan()
    context['question'] = 'changed'
    def forbidden(*args):
        raise AssertionError('Evidence mismatch must stop before contacting a provider')
    def ask(question, artifacts, config, **kwargs):
        agent._post(config, '/chat/completions', {'messages': [{'role': 'user', 'content': json.dumps(context)}]})
    monkeypatch.setattr(agent, '_post', forbidden)
    monkeypatch.setattr(agent, 'ask', ask)
    report = audit.execute(plan, agent.LLMConfig(model='glm-fixture', answer_mode='evidence'), None, None, None, tmp_path / 'report.json')
    assert report['status'] == 'stopped_after_failure'
    assert report['model_requests_attempted'] == 0
    assert 'Evidence changed' in report['results'][0]['error']
    assert agent._post is forbidden


def test_provider_failure_stops_without_retry(monkeypatch, tmp_path):
    plan, context = make_plan()
    plan['cases'] *= 2
    plan['maximum_model_requests'] = 2
    calls = []
    def post(*args):
        calls.append(args)
        raise RuntimeError('fixture timeout')
    def ask(question, artifacts, config, **kwargs):
        return agent._post(config, '/chat/completions', {'messages': [{'role': 'user', 'content': json.dumps(context)}]})
    monkeypatch.setattr(agent, '_post', post)
    monkeypatch.setattr(agent, 'ask', ask)
    report = audit.execute(plan, agent.LLMConfig(model='glm-fixture', answer_mode='evidence'), None, None, None, tmp_path / 'report.json')
    assert len(calls) == 1 and len(report['results']) == 1
    assert report['status'] == 'stopped_after_failure'
    assert json.loads((tmp_path / 'report.json').read_text(encoding='utf-8'))['status'] == 'stopped_after_failure'


def test_local_guard_makes_zero_requests(monkeypatch, tmp_path):
    plan, _ = make_plan(guard=True)
    monkeypatch.setattr(agent, 'ask', lambda *args, **kwargs: response(guard=True))
    def forbidden(*args):
        raise AssertionError('A local guard must not contact a provider')
    monkeypatch.setattr(agent, '_post', forbidden)
    report = audit.execute(plan, agent.LLMConfig(model='glm-fixture', answer_mode='evidence'), None, None, None, tmp_path / 'report.json')
    assert report['model_requests_attempted'] == 0
    assert report['results'][0]['checks']['expected_local_guard'] is True
    assert agent._post is forbidden


def test_default_cli_only_prepares_and_failed_cli_returns_nonzero(monkeypatch, tmp_path):
    plan, _ = make_plan()
    monkeypatch.setattr(audit, 'prepare', lambda *args: plan)
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--out', str(output)])
    assert audit.main() == 0
    assert not output.exists()
    assert output.with_name('report_plan.json').exists()
    class Store:
        def __init__(self, *args):
            pass
        def config(self, *args):
            return agent.LLMConfig(model='glm-fixture')
    monkeypatch.setattr(audit, 'ConnectionStore', Store)
    monkeypatch.setattr(audit, 'execute', lambda *args: {'status': 'stopped_after_failure'})
    monkeypatch.setattr(sys, 'argv', ['audit', '--out', str(output), '--execute', '--connection', 'fixture'])
    assert audit.main() == 1
    output.write_text('{}')
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2

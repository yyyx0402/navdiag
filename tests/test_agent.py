import json

import pandas as pd
import pytest

from navdiag import agent
from navdiag.agent import EvidenceTools, LLMConfig, ask
from navdiag.data import generate_demo
from navdiag.learning import train_and_evaluate


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp('agent-artifacts')
    train_and_evaluate(generate_demo(root / 'data.csv'), root / 'experiment')
    return root / 'experiment'


def test_evidence_tools_have_no_arbitrary_file_or_shell_capability(artifacts):
    tools = EvidenceTools(artifacts)
    with pytest.raises(ValueError, match='Unknown tool'):
        tools.invoke('read_file', {'path': '.env'})
    with pytest.raises(ValueError, match='allowed schema'):
        tools.invoke('list_trips', {'command': 'anything'})
    result = tools.invoke('trip_summary', {'trip_id': None})
    assert 'horizontal_error_m' not in result['payload']


def test_offline_answer_is_labelled_and_cites_actual_tools(artifacts):
    result = ask('比较模型的 F1', artifacts)
    assert result['backend'] == 'offline'
    assert '模拟' in result['answer']
    assert result['citation_check'] == {'has_evidence_citations': True, 'unknown_citations': []}
    assert result['usage'] == []


@pytest.mark.parametrize('provider', ['openai', 'deepseek'])
def test_both_api_protocols_return_tool_results_to_the_model(monkeypatch, artifacts, provider):
    recorded = []
    def fake_post(config, endpoint, payload):
        recorded.append((endpoint, json.loads(json.dumps(payload))))
        if len(recorded) == 1:
            if provider == 'openai':
                return {'output': [{'type': 'function_call', 'call_id': 'call_1',
                                    'name': 'compare_models', 'arguments': '{}'}]}
            return {'choices': [{'message': {'role': 'assistant', 'content': None,
                        'reasoning_content': 'fixture reasoning', 'tool_calls': [{'id': 'call_1', 'type': 'function',
                        'function': {'name': 'compare_models', 'arguments': '{}'}}]}}]}
        if provider == 'openai':
            assert payload['input'][-1]['type'] == 'function_call_output'
            assert payload['store'] is False
            return {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Recorded comparison [E1]'}]}]}
        assert payload['messages'][-1]['role'] == 'tool'
        assert payload['messages'][-2]['reasoning_content'] == 'fixture reasoning'
        return {'choices': [{'message': {'role': 'assistant', 'content': 'Recorded comparison [E1]'}}]}
    monkeypatch.setattr(agent, '_post', fake_post)
    result = ask('Compare the models', artifacts, LLMConfig(provider, 'fixture-model', 'https://fixture.invalid', 'fixture-key'))
    assert len(recorded) == 2
    assert result['citation_check']['unknown_citations'] == []
    assert result['evidence'][0]['tool'] == 'compare_models'


def test_generic_api_key_cannot_be_forwarded_to_another_provider(monkeypatch):
    for name in ['OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'NAVDIAG_MODEL', 'NAVDIAG_BASE_URL']:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('NAVDIAG_PROVIDER', 'openai')
    monkeypatch.setenv('NAVDIAG_API_KEY', 'fixture-openai-key')
    with pytest.raises(ValueError, match='API mode needs'):
        LLMConfig.from_environment('deepseek')


def test_segment_summary_excludes_reference_labels(artifacts):
    tools = EvidenceTools(artifacts)
    result = tools.invoke('segment_summary', {'trip_id': None, 'start_s': 0, 'end_s': 10})
    assert result['payload']['epochs'] > 0
    assert result['payload']['reference_labels_used'] is False
    with pytest.raises(ValueError):
        tools.invoke('segment_summary', {'trip_id': None, 'start_s': 20, 'end_s': 10})

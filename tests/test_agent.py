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


def test_technical_notes_are_not_mislabelled_as_synthetic_observations(artifacts):
    evidence = EvidenceTools(artifacts).invoke('search_knowledge', {'query': 'HDOP'})
    assert evidence['data_kind'] == 'knowledge'
    assert evidence['payload']['notes']
    assert all(note['source_url'] and note['source_type'] for note in evidence['payload']['notes'])


def test_paper_concept_question_uses_passages_without_unrelated_trip_metrics(artifacts):
    result = ask('论文中各颗卫星怎样交换信息？', artifacts)
    assert [item['tool'] for item in result['evidence']] == ['search_knowledge']
    assert '模拟实验' not in result['answer']


def test_no_evidence_abstains_locally_without_any_model_request(monkeypatch, artifacts):
    monkeypatch.setenv('NAVDIAG_RETRIEVAL_MODE', 'sparse')
    def forbidden(*args):
        raise AssertionError('An unsupported question must not make a model API request')
    monkeypatch.setattr(agent, '_post', forbidden)
    config = LLMConfig('compatible', 'fixture-model', 'https://fixture.invalid', answer_mode='evidence')
    result = ask('苹果派怎么烤？', artifacts, config)
    assert result['generation_skipped'] is True and result['backend'] == 'local_evidence_guard'
    assert result['usage'] == [] and result['model_diagnostics'] == []
    assert result['evidence'][0]['payload']['notes'] == []
    assert [item['tool'] for item in result['evidence']] == ['search_knowledge']
    assert '没有找到' in result['answer'] and '行程' not in result['answer']


def test_tool_agent_can_rewrite_a_query_without_local_baseline_rejection(monkeypatch, artifacts):
    requests = []
    def fake_post(config, endpoint, payload):
        requests.append(payload)
        if len(requests) == 1:
            return {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'rewrite', 'type': 'function', 'function': {'name': 'search_knowledge',
                'arguments': json.dumps({'query': 'HDOP 卫星几何'})}}]}}]}
        assert json.loads(payload['messages'][-1]['content'])['payload']['notes']
        return {'choices': [{'message': {'content': 'Explanation using rewritten-query evidence [E1]'}}]}
    monkeypatch.setattr(agent, '_post', fake_post)
    result = ask('这些点的形状为什么会影响位置计算？', artifacts,
                 LLMConfig('compatible', 'fixture-model', 'https://fixture.invalid', answer_mode='tools'))
    assert len(requests) == 2 and result['generation_skipped'] is False
    assert result['evidence'][0]['payload']['query'] == 'HDOP 卫星几何'


def test_conceptual_paper_question_avoids_model_metrics_in_nlos_context(artifacts):
    result = ask('当前 Transformer 完整复现论文了吗？', artifacts, task_context='satellite_nlos')
    assert [item['tool'] for item in result['evidence']] == ['search_knowledge']


@pytest.mark.parametrize('provider', ['openai', 'deepseek', 'compatible'])
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


def test_compatible_service_never_inherits_deepseek_key(monkeypatch):
    monkeypatch.setenv('NAVDIAG_PROVIDER', 'compatible')
    monkeypatch.setenv('NAVDIAG_MODEL', 'my-model')
    monkeypatch.setenv('NAVDIAG_BASE_URL', 'http://127.0.0.1:11434/v1')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'fixture-key-for-another-provider')
    monkeypatch.delenv('NAVDIAG_API_KEY', raising=False)
    assert LLMConfig.from_environment('compatible').api_key == ''


def test_evidence_mode_supplies_real_tool_results_without_requiring_model_tools(monkeypatch, artifacts):
    recorded = []
    def fake_post(config, endpoint, payload):
        recorded.append(payload)
        assert endpoint == '/chat/completions'
        assert payload['model'] == 'user-selected-model'
        assert 'tools' not in payload and 'tool_choice' not in payload
        context = json.loads(payload['messages'][-1]['content'])
        assert context['evidence'][0]['tool'] == 'compare_models'
        assert context['evidence'][0]['payload']['selection_data'] == 'validation'
        assert context['evidence'][1]['data_kind'] == 'knowledge'
        return {'choices': [{'message': {'role': 'assistant', 'content': 'Comparison grounded in saved metrics [E1]'}}]}
    monkeypatch.setattr(agent, '_post', fake_post)
    config = LLMConfig('compatible', 'user-selected-model', 'http://localhost:11434/v1', answer_mode='evidence')
    result = ask('比较模型的 F1', artifacts, config)
    assert len(recorded) == 1 and result['answer_mode'] == 'evidence'
    assert result['citation_check'] == {'has_evidence_citations': True, 'unknown_citations': []}


def test_tool_mode_does_not_accept_an_answer_without_evidence(monkeypatch, artifacts):
    monkeypatch.setattr(agent, '_post', lambda *args: {'choices': [{'message': {'content': 'Unsupported guess'}}]})
    with pytest.raises(RuntimeError, match='did not call an evidence tool'):
        ask('比较模型的 F1', artifacts, LLMConfig('compatible', 'my-model', 'http://localhost:8000/v1'))


def test_malformed_chat_response_has_an_actionable_error(monkeypatch, artifacts):
    monkeypatch.setattr(agent, '_post', lambda *args: {'different_protocol': 'fixture'})
    with pytest.raises(RuntimeError, match='Chat Completions message'):
        ask('比较模型的 F1', artifacts, LLMConfig('compatible', 'my-model', 'http://localhost:8000/v1'))


def test_segment_summary_excludes_reference_labels(artifacts):
    tools = EvidenceTools(artifacts)
    result = tools.invoke('segment_summary', {'trip_id': None, 'start_s': 0, 'end_s': 10})
    assert result['payload']['epochs'] > 0
    assert result['payload']['reference_labels_used'] is False
    with pytest.raises(ValueError):
        tools.invoke('segment_summary', {'trip_id': None, 'start_s': 20, 'end_s': 10})


def test_reasoning_budget_exhaustion_is_diagnosed_without_exposing_thoughts(monkeypatch, artifacts):
    calls = []
    def fake_post(config, endpoint, payload):
        calls.append(payload)
        assert payload['reasoning_effort'] == 'low'
        assert 'thinking' not in payload
        return {'choices': [{'finish_reason': 'length', 'message': {'content': '', 'reasoning_content': 'private thinking'}}],
                'usage': {'completion_tokens': 2048, 'completion_tokens_details': {'reasoning_tokens': 2048},
                          'untrusted_text': 'private value'}}
    monkeypatch.setattr(agent, '_post', fake_post)
    config = LLMConfig('compatible', 'glm-5.3', 'https://fixture.invalid', 'fixture-key',
                       answer_mode='evidence', reasoning_effort='low', max_output_tokens=2048)
    with pytest.raises(agent.ModelResponseError) as caught:
        ask('比较模型的 F1', artifacts, config)
    assert caught.value.code == 'output_budget_exhausted'
    assert caught.value.diagnostics['usage']['reasoning_tokens'] == 2048
    assert 'private' not in str(caught.value) + json.dumps(caught.value.diagnostics)
    assert len(calls) == 1


@pytest.mark.parametrize('reason,thinking,code', [
    ('stop', 'private thinking', 'reasoning_without_answer'),
    ('stop', '', 'empty_model_answer'), ('content_filter', '', 'provider_content_filter')])
def test_empty_answers_distinguish_provider_outcomes(monkeypatch, artifacts, reason, thinking, code):
    monkeypatch.setattr(agent, '_post', lambda *args: {'choices': [
        {'finish_reason': reason, 'message': {'content': None, 'reasoning_content': thinking}}]})
    with pytest.raises(agent.ModelResponseError) as caught:
        ask('比较模型的 F1', artifacts, LLMConfig('compatible', 'my-model', 'https://fixture.invalid', answer_mode='evidence'))
    assert caught.value.code == code


def test_truncated_nonempty_answer_is_marked_incomplete(monkeypatch, artifacts):
    monkeypatch.setattr(agent, '_post', lambda *args: {'choices': [
        {'finish_reason': 'length', 'message': {'content': 'Partial answer [E1]'}}]})
    result = ask('比较模型的 F1', artifacts, LLMConfig('compatible', 'my-model', 'https://fixture.invalid', answer_mode='evidence'))
    assert result['answer_complete'] is False and result['warnings']
    assert result['answer'] == 'Partial answer [E1]'


def test_unknown_evidence_citation_is_exposed_as_a_warning(monkeypatch, artifacts):
    monkeypatch.setattr(agent, '_post', lambda *args: {'choices': [
        {'finish_reason': 'stop', 'message': {'content': 'Unsupported citation [E999]'}}]})
    result = ask('比较模型的 F1', artifacts, LLMConfig('compatible', 'my-model', 'https://fixture.invalid', answer_mode='evidence'))
    assert result['citation_check']['unknown_citations'] == ['E999']
    assert any('不存在' in warning for warning in result['warnings'])

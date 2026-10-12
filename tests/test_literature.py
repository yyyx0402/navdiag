import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from navdiag import agent
from navdiag.data import generate_demo
from navdiag.learning import train_and_evaluate
from navdiag.dashboard import make_server
from navdiag.literature import reading_list, registry


def test_registry_covers_learning_routes_and_preserves_preprint_year():
    records = {record['id']: record for record in registry()}
    assert records['L09']['learning']['topics'] == ['positioning', 'nlos', 'preprocessing']
    assert records['L15']['year'] == 2022 and records['L15']['edition'] == 'preprint'
    assert '2023' in records['L15']['verification']['locator']
    assert records['L19']['year'] == 2025 and records['L19']['edition'] == 'journal'
    assert all(record['learning']['why'] and record['learning']['how'] and record['learning']['questions']
               and record['verification']['read_scope'] for record in records.values())


def test_paper_only_search_can_find_sources_without_project_passages(monkeypatch):
    monkeypatch.setattr(agent, '_post', lambda *args: pytest.fail('Reading search must not call a model'))
    result = reading_list('开放天空预归一化 多频特征缺失 分支逻辑回归', mode='sparse')
    assert result['papers'][0]['id'] == 'L04'
    assert all(record['source_type'] == 'research_summary' for record in result['papers'])
    assert result['model_api_called'] is False and len(result['papers']) <= 5


def test_filters_apply_before_ranking_and_unrelated_query_does_not_force_recommendations():
    result = reading_list('载波相位 RTK LiDAR 紧耦合', topic='maps', difficulty='advanced', mode='sparse')
    assert any(record['id'] == 'L19' for record in result['papers'])
    assert all('maps' in record['learning']['topics'] and record['learning']['difficulty'] == 'advanced'
               for record in result['papers'])
    assert reading_list('苹果派怎么烤？', mode='sparse')['papers'] == []
    correlated = reading_list('相关器输出波形归一化 SVM 神经网络', mode='sparse')
    assert correlated['papers'][0]['id'] == 'L02'
    comparison = reading_list('Transformer是否一定比随机森林更好？', mode='sparse')
    assert all('generalization' in record['learning']['topics'] for record in comparison['papers'])


def test_browse_is_filtered_and_does_not_need_embeddings(monkeypatch):
    from navdiag import literature
    monkeypatch.setattr(literature, 'retrieve_knowledge', lambda *args, **kwargs: pytest.fail('Browse must not retrieve'))
    result = reading_list(topic='smartphone', difficulty='introductory')
    assert {record['id'] for record in result['papers']} == {'L18'}
    assert result['mode'] == 'browse' and result['retrieval_method'] is None


def test_learning_guides_are_not_added_to_llm_retrieval_context():
    from navdiag.knowledge import retrieve_knowledge
    result = retrieve_knowledge('Doppler TDCP 两步 手机定位 Suzuki', mode='sparse', source_type='research_summary')
    assert any(record['id'] == 'L14' for record in result['notes'])
    assert all('learning' not in record for record in result['notes'])


@pytest.mark.parametrize('arguments', [{'query':[]}, {'query':'x'*2001}, {'topic':[]}, {'difficulty':'expert'}, {'limit':True}])
def test_invalid_search_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        reading_list(**arguments)


def test_library_api_is_local_readonly_and_does_not_use_model_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv('NAVDIAG_RETRIEVAL_MODE', 'sparse')
    monkeypatch.setattr(agent, '_post', lambda *args: pytest.fail('Library endpoint must not use a model'))
    train_and_evaluate(generate_demo(tmp_path / 'data.csv'), tmp_path / 'artifacts')
    server = make_server(tmp_path / 'artifacts', port=0, connections_file=tmp_path/'connections.json')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url=f'http://127.0.0.1:{server.server_port}/api/literature/search'
    def request(payload, headers=None):
        return Request(url, json.dumps(payload).encode(), {'Content-Type':'application/json', **(headers or {})})
    try:
        with urlopen(request({'query':'开放天空预归一化 多频分支'})) as response:
            data=json.load(response)
            assert data['papers'][0]['id']=='L04' and data['model_api_called'] is False
        assert not (tmp_path/'connections.json').exists()
        for bad in [{'query':[]}, {'path':'.env'}, {'topic':'unknown'}]:
            with pytest.raises(HTTPError) as error:
                urlopen(request(bad))
            assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(request({}, {'Origin':'https://unrelated.invalid'}))
        assert error.value.code == 403
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)

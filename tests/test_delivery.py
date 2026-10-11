import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import pytest

from navdiag.agent_eval import evaluate_agent
from navdiag.dashboard import make_server
from navdiag.data import generate_demo, load_dataset, split_by_trip
from navdiag.learning import evaluate, predict_frame, train_and_evaluate


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp('delivery')
    train_and_evaluate(generate_demo(root / 'epochs.csv'), root / 'artifacts')
    return root / 'artifacts'


def test_explicit_split_covers_all_trips_and_rejects_overlap(tmp_path):
    frame = load_dataset(generate_demo(tmp_path / 'epochs.csv', trips=5))
    manifest = {'train': ['demo_trip_00', 'demo_trip_01', 'demo_trip_02'],
                'validation': ['demo_trip_03'], 'test': ['demo_trip_04']}
    splits = split_by_trip(frame, manifest=manifest)
    assert set(splits['test'].trip_id) == {'demo_trip_04'}
    with pytest.raises(ValueError, match='without overlap'):
        split_by_trip(frame, manifest={**manifest, 'test': ['demo_trip_03', 'demo_trip_04']})
    with pytest.raises(ValueError, match='at least one trip'):
        split_by_trip(frame, manifest={**manifest, 'test': None})


def test_single_class_metrics_do_not_report_unmeasurable_recall_as_zero():
    metrics = evaluate(np.zeros(4, dtype=int), np.array([0.1, 0.8, 0.2, 0.9]), 0.5)
    assert metrics['recall'] is None and metrics['f1'] is None
    assert metrics['false_positive_rate'] == 0.5
    assert metrics['confusion_matrix'] == [[2, 2], [0, 0]]


def test_serialized_model_inference_needs_no_reference_columns(experiment):
    frame = pd.read_csv(experiment / 'predictions.csv')
    bundle = joblib.load(experiment / 'models.joblib')
    observed_only = frame.drop(columns=['horizontal_error_m', 'is_unreliable'])
    np.testing.assert_allclose(predict_frame(bundle, observed_only), frame.predicted_risk)
    with pytest.raises(ValueError, match='Missing inference columns'):
        predict_frame(bundle, observed_only.drop(columns=bundle['features']))


def test_dashboard_serves_packaged_assets_and_rejects_cross_origin(experiment):
    server = make_server(experiment, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base + '/') as response:
            assert 'NavDiag'.encode() in response.read()
        with urlopen(base + '/api/trajectory') as response:
            assert json.load(response)['points']
        body = json.dumps({'question': '比较模型的 F1', 'provider': 'offline'}).encode()
        request = Request(base + '/api/ask', body, {'Content-Type': 'application/json'})
        with urlopen(request) as response:
            assert json.load(response)['evidence'][0]['tool'] == 'compare_models'
        request = Request(base + '/api/ask', body, {'Origin': 'https://unrelated.invalid'})
        with pytest.raises(HTTPError) as caught:
            urlopen(request)
        assert caught.value.code == 403
        body = json.dumps({'question': 'question', 'provider': ['offline']}).encode()
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(base + '/api/ask', body, {'Content-Type': 'application/json'}))
        assert caught.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_agent_evaluation_exposes_errors_instead_of_hiding_them(experiment, tmp_path):
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps([
        {'id': 'valid', 'question': '前 10 秒的风险', 'required_tool': 'segment_summary'},
        {'id': 'invalid', 'question': '前 999999 秒的风险', 'required_tool': 'segment_summary'},
    ]), encoding='utf-8')
    report = evaluate_agent(experiment, tmp_path / 'evaluation.json', cases_path=cases)
    assert report['passed'] == 1 and report['cases'] == 2
    assert 'error' in report['results'][1]
    assert 'not an LLM evaluation' in report['limitations']


def test_dashboard_custom_connection_roundtrip_and_inference(experiment, tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    requests = []
    class ModelHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, self.headers.get('Authorization'), payload))
            assert payload['model'] == 'my-selected-model'
            assert 'tools' not in payload
            context = json.loads(payload['messages'][-1]['content'])
            assert context['evidence'][0]['tool'] == 'compare_models'
            self.send_response(200)
            self.end_headers()
            result = {'choices': [{'message': {'content': 'Recorded model comparison [E1]'}}]}
            if '空回答' in context['question']:
                result = {'choices': [{'finish_reason': 'length', 'message': {'content': '', 'reasoning_content': 'private thought'}}]}
            self.wfile.write(json.dumps(result).encode())
    model_server = ThreadingHTTPServer(('127.0.0.1', 0), ModelHandler)
    model_thread = threading.Thread(target=model_server.serve_forever, daemon=True)
    model_thread.start()
    path = tmp_path / '.navdiag' / 'llm_connections.json'
    server = make_server(experiment, port=0, connections_file=path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def post(route, payload, headers=None):
        body = json.dumps(payload).encode()
        with urlopen(Request(base + route, body, {'Content-Type': 'application/json', **(headers or {})})) as response:
            return json.load(response)
    try:
        payload = {'name': 'Custom local service', 'base_url': f'http://127.0.0.1:{model_server.server_port}/v1',
                   'model': 'my-selected-model', 'api_key': 'fixture-private-value', 'answer_mode': 'evidence'}
        saved = post('/api/connections/save', payload)['connection']
        assert 'api_key' not in saved
        with urlopen(base + '/api/config') as response:
            body = response.read().decode()
            assert 'fixture-private-value' not in body and json.loads(body)['connections'][0]['id'] == saved['id']
        result = post('/api/ask', {'question': '比较模型的 F1', 'provider': 'connection', 'connection_id': saved['id']})
        assert result['model'] == 'my-selected-model' and result['answer_mode'] == 'evidence'
        assert result['citation_check']['unknown_citations'] == []
        assert len(requests) == 1 and requests[0][:2] == ('/v1/chat/completions', 'Bearer fixture-private-value')
        assert 'fixture-private-value' not in json.dumps(result)
        with pytest.raises(HTTPError) as caught:
            post('/api/ask', {'question': '比较模型的 F1，复现空回答', 'provider': 'connection', 'connection_id': saved['id']})
        error = json.loads(caught.value.read())
        assert caught.value.code == 502 and error['error_code'] == 'output_budget_exhausted'
        assert error['model_diagnostics']['finish_reason'] == 'length'
        assert 'private' not in json.dumps(error)
        for route, body in [('/api/connections/save', payload), ('/api/connections/models', payload),
                            ('/api/connections/delete', {'id': saved['id']})]:
            with pytest.raises(HTTPError) as caught:
                post(route, body, {'Origin': 'https://unrelated.invalid'})
            assert caught.value.code == 403
        with pytest.raises(HTTPError) as caught:
            post('/api/connections/save', payload, {'Host': 'rebound.example.test'})
        assert caught.value.code == 403
        assert post('/api/connections/delete', {'id': saved['id']})['deleted']
        with pytest.raises(HTTPError) as caught:
            post('/api/ask', {'question': 'question', 'provider': 'connection', 'connection_id': saved['id']})
        assert caught.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        model_server.shutdown()
        model_server.server_close()
        model_thread.join(timeout=2)

import json
import socket
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from navdiag.agent import LLMConfig, _request_json
from navdiag import agent
from navdiag.llm_connections import ConnectionStore


def profile(**changes):
    return {'name': 'My service', 'base_url': 'https://models.example.test/v1', 'model': 'my-model',
            'api_key': 'fixture-private-value', **changes}


def test_saved_profiles_hide_credentials_and_keep_them_bound_to_the_address(tmp_path):
    store = ConnectionStore(tmp_path / '.navdiag' / 'llm_connections.json')
    public = store.save(profile())
    assert 'api_key' not in public and public['has_api_key'] is True
    assert 'fixture-private-value' not in json.dumps(store.list())
    assert 'fixture-private-value' not in repr(store.config(public['id']))
    changed = store.save(profile(id=public['id'], model='other-model', api_key=''))
    assert store.config(changed['id']).api_key == 'fixture-private-value'
    with pytest.raises(ValueError, match='address changed'):
        store.save(profile(id=public['id'], base_url='https://other.example.test/v1', api_key=''))
    cleared = store.save(profile(id=public['id'], base_url='https://other.example.test/v1', api_key='', clear_api_key=True))
    assert not cleared['has_api_key'] and store.config(cleared['id']).api_key == ''
    reloaded = ConnectionStore(store.path)
    assert reloaded.list() == store.list()
    reloaded.delete(public['id'])
    assert reloaded.list() == []


def test_profiles_are_isolated_and_full_chat_endpoint_is_normalized(tmp_path):
    store = ConnectionStore(tmp_path / 'connections.json')
    first = store.save(profile())
    second = store.save(profile(name='Local model', base_url='http://127.0.0.1:11434/v1/chat/completions/', api_key=''))
    assert store.config(second['id']).base_url == 'http://127.0.0.1:11434/v1'
    assert store.config(second['id']).api_key == ''
    assert store.config(first['id']).api_key == 'fixture-private-value'
    with pytest.raises(ValueError, match='Unknown model connection'):
        store.config('../anything')


@pytest.mark.parametrize('changes', [
    {'base_url': 'http://remote.example.test/v1'}, {'base_url': 'https://u:pass@models.example.test/v1'},
    {'base_url': 'https://models.example.test/v1?key=secret'}, {'base_url': 'https:///v1'},
    {'api_key': 'line\nbreak'}, {'timeout_s': 0}, {'max_output_tokens': True},
    {'answer_mode': 'unknown'}, {'name': ''}, {'model': ''}, {'unexpected': 'value'},
    {'reasoning_effort': 'unsupported'},
])
def test_invalid_connection_cannot_be_saved(tmp_path, changes):
    store = ConnectionStore(tmp_path / 'connections.json')
    with pytest.raises(ValueError):
        store.save(profile(**changes))
    assert not store.path.exists()


def test_corrupt_file_error_does_not_echo_file_contents(tmp_path):
    path = tmp_path / 'connections.json'
    path.write_text('fixture-private-value: invalid-json', encoding='utf-8')
    with pytest.raises(ValueError) as caught:
        ConnectionStore(path).list()
    assert 'fixture-private-value' not in str(caught.value)
    path.write_text(json.dumps({'version': 1, 'connections': [{'id': 'a' * 32, 'model': 'my-model',
                    'base_url': 'https://models.example.test/v1', 'api_key': 'fixture-private-value'}]}), encoding='utf-8')
    with pytest.raises(ValueError, match='Local connection file is invalid'):
        ConnectionStore(path).list()


def test_model_discovery_does_not_need_a_model_or_key_and_does_not_generate(tmp_path):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            requests.append((self.command, self.path, self.headers.get('Authorization')))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'data': [{'id': 'local-model'}, {'id': 'other-model'}, {'invalid': True}]}).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        store = ConnectionStore(tmp_path / 'connections.json')
        result = store.models(profile(base_url=f'http://127.0.0.1:{server.server_port}/v1', model='', api_key=''))
        assert result['models'] == ['local-model', 'other-model']
        assert requests == [('GET', '/v1/models', None)]
        assert not store.path.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_redirect_does_not_forward_a_saved_credential():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            requests.append(self.path)
            self.send_response(302)
            self.send_header('Location', '/credential-sink')
            self.end_headers()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = LLMConfig('compatible', 'my-model', f'http://127.0.0.1:{server.server_port}/v1', 'fixture-private-value')
        with pytest.raises(RuntimeError, match='HTTP 302'):
            _request_json(config, '/models')
        assert requests == ['/v1/models']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('reason, expected', [
    (socket.gaierror(11001, 'fixture-private-value'), '域名无法解析'),
    (TimeoutError('fixture-private-value'), '读取超时'),
    (ConnectionRefusedError(10061, 'fixture-private-value'), '拒绝连接'),
    (ConnectionResetError(10054, 'fixture-private-value'), '传输中断开'),
    (ssl.SSLError(1, 'fixture-private-value'), 'TLS/SSL'),
    (ssl.SSLCertVerificationError(1, 'fixture-private-value'), '证书验证失败'),
])
def test_transport_errors_are_specific_and_do_not_echo_sensitive_text(monkeypatch, reason, expected):
    def fail(*args, **kwargs):
        raise URLError(reason)
    monkeypatch.setattr(agent, 'build_opener', lambda *args: SimpleNamespace(open=fail))
    config = LLMConfig('compatible', 'my-model', 'https://models.example.test/v1', 'fixture-private-value')
    with pytest.raises(RuntimeError) as caught:
        _request_json(config, '/models')
    assert expected in str(caught.value)
    assert 'fixture-private-value' not in str(caught.value)

"""User-selected Chat Completions connections; credentials stay in ignored local state."""
import json
import os
import re
import tempfile
import threading
import uuid
from pathlib import Path

from navdiag.agent import LLMConfig, _request_json


def connection_path(env_file: Path | None = None) -> Path:
    return (env_file.resolve().parent if env_file else Path.cwd()) / '.navdiag' / 'llm_connections.json'


def _public(item: dict) -> dict:
    return {**{key: item[key] for key in ('id', 'name', 'base_url', 'model', 'answer_mode', 'max_output_tokens', 'timeout_s')},
            'reasoning_effort': item.get('reasoning_effort', 'default'),
            'has_api_key': bool(item.get('api_key'))}


class ConnectionStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()

    def _read(self) -> dict:
        if not self.path.exists():
            return {'version': 1, 'connections': []}
        try:
            if self.path.stat().st_size > 256_000:
                raise ValueError
            result = json.loads(self.path.read_text(encoding='utf-8'))
            if result.get('version') != 1 or not isinstance(result.get('connections'), list) or len(result['connections']) > 20:
                raise ValueError
            for item in result['connections']:
                if not isinstance(item, dict) or not re.fullmatch(r'[a-f0-9]{32}', item.get('id', '')):
                    raise ValueError
                self._config(item).validate()
                _public(item)
            return result
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            raise ValueError('Local connection file is invalid; restore it locally or remove the damaged file.') from None

    def _write(self, state: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.connections-', dir=self.path.parent)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(state, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _config(item: dict) -> LLMConfig:
        return LLMConfig(provider='compatible', model=item['model'], base_url=item['base_url'],
                         api_key=item.get('api_key', ''), answer_mode=item.get('answer_mode', 'evidence'),
                         reasoning_effort=item.get('reasoning_effort', 'default'),
                         max_output_tokens=item.get('max_output_tokens', 2048), timeout_s=item.get('timeout_s', 90))

    def list(self) -> list[dict]:
        with self.lock:
            return [_public(item) for item in self._read()['connections']]

    def config(self, identifier: str) -> LLMConfig:
        if not isinstance(identifier, str):
            raise ValueError('Select a saved connection')
        with self.lock:
            item = next((item for item in self._read()['connections'] if item['id'] == identifier), None)
            if item is None:
                raise ValueError('Unknown model connection')
            return self._config(item).validate()

    def prepare(self, payload: dict) -> dict:
        allowed = {'id', 'name', 'base_url', 'model', 'api_key', 'clear_api_key', 'answer_mode',
                   'max_output_tokens', 'timeout_s', 'reasoning_effort'}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError('Unexpected connection fields')
        identifier = payload.get('id')
        state = self._read()
        previous = next((item for item in state['connections'] if item['id'] == identifier), None)
        if identifier is not None and previous is None:
            raise ValueError('Unknown model connection')
        base_url, model, name = (payload.get(key, '') for key in ('base_url', 'model', 'name'))
        if not all(isinstance(value, str) for value in (base_url, model, name)):
            raise ValueError('Connection name, URL and model must be text')
        base_url = base_url.strip().rstrip('/')
        # Accept the complete chat endpoint pasted from a provider's example.
        if base_url.endswith('/chat/completions'):
            base_url = base_url[:-len('/chat/completions')]
        name = name.strip()
        if not 1 <= len(name) <= 80 or any(ord(char) < 32 for char in name):
            raise ValueError('Connection name must contain 1-80 characters')
        key = payload.get('api_key', '')
        clear = payload.get('clear_api_key', False)
        if not isinstance(key, str) or not isinstance(clear, bool):
            raise ValueError('Invalid API key fields')
        key = key.strip()
        if clear and key:
            raise ValueError('Choose either a replacement key or clear key')
        if not clear and not key and previous and previous.get('api_key'):
            if base_url != previous['base_url']:
                raise ValueError('API address changed: enter its own key or explicitly clear the saved key.')
            key = previous['api_key']
        item = {'id': identifier or uuid.uuid4().hex, 'name': name, 'base_url': base_url, 'model': model.strip(),
                'api_key': key, 'answer_mode': payload.get('answer_mode', 'evidence'),
                'max_output_tokens': payload.get('max_output_tokens', 4096), 'timeout_s': payload.get('timeout_s', 90),
                'reasoning_effort': payload.get('reasoning_effort', previous.get('reasoning_effort', 'default') if previous else 'default')}
        self._config(item).validate()
        return item

    def save(self, payload: dict) -> dict:
        with self.lock:
            item = self.prepare(payload)
            state = self._read()
            if not payload.get('id') and len(state['connections']) >= 20:
                raise ValueError('At most 20 model connections can be saved')
            state['connections'] = [value for value in state['connections'] if value['id'] != item['id']] + [item]
            self._write(state)
            return _public(item)

    def delete(self, identifier: str):
        with self.lock:
            state = self._read()
            if not isinstance(identifier, str) or not any(item['id'] == identifier for item in state['connections']):
                raise ValueError('Unknown model connection')
            state['connections'] = [item for item in state['connections'] if item['id'] != identifier]
            self._write(state)

    def models(self, payload: dict) -> dict:
        if isinstance(payload, dict):
            payload = dict(payload)
            if isinstance(payload.get('model', ''), str) and not payload.get('model', '').strip():
                payload['model'] = '__model_discovery__'
            if isinstance(payload.get('name', ''), str) and not payload.get('name', '').strip():
                payload['name'] = 'Model discovery'
        with self.lock:
            config = self._config(self.prepare(payload))
        response = _request_json(config, '/models')
        data = response.get('data')
        if not isinstance(data, list):
            raise RuntimeError('The service did not return a compatible model list; enter the model ID manually.')
        models = sorted({item['id'] for item in data if isinstance(item, dict)
                         and isinstance(item.get('id'), str) and 0 < len(item['id']) <= 2048})
        return {'models': models[:500], 'notice': 'Model discovery checks GET /models only; inference and tool support are not tested.'}

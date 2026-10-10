"""Loopback-only dashboard for experiment results and evidence-backed diagnostics."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from navdiag.agent import LLMConfig, ask


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient='records', double_precision=10))


def make_server(artifacts: Path, port: int = 8765, env_file: Path | None = None) -> ThreadingHTTPServer:
    artifacts = artifacts.resolve()
    report = json.loads((artifacts / 'report.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(artifacts / 'predictions.csv', dtype={'trip_id': 'string'})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed, query = urlparse(self.path), parse_qs(urlparse(self.path).query)
            if parsed.path == '/api/report':
                self.send_json(report)
            elif parsed.path == '/api/trips':
                result = []
                for trip, part in predictions.groupby('trip_id', sort=True):
                    result.append({'trip_id': str(trip), 'epochs': len(part),
                                   'flagged': int(part.predicted_unreliable.sum()),
                                   'devices': sorted(part.device_id.unique().tolist()) if 'device_id' in part else ['receiver']})
                self.send_json(result)
            elif parsed.path == '/api/trajectory':
                trip = query.get('trip', [sorted(predictions.trip_id.unique())[0]])[0]
                part = predictions[predictions.trip_id == trip].copy()
                if 'device_id' in part and len(part):
                    device = query.get('device', [sorted(part.device_id.unique())[0]])[0]
                    part = part[part.device_id == device]
                if part.empty:
                    self.send_json({'error': 'Unknown trip/device'}, 404)
                    return
                part = part.sort_values('timestamp_ms')
                part['time_s'] = (part.timestamp_ms - part.timestamp_ms.min()) / 1000
                columns = ['time_s', 'predicted_risk', 'horizontal_error_m', 'residual_rms_m', 'hdop',
                           'cn0_mean_dbhz', 'cn0_min_dbhz', 'satellite_count', 'predicted_unreliable']
                columns += [name for name in part if name.endswith('_e_m') or name.endswith('_n_m') or name.endswith('_error_m')]
                # Limit rendering size, retaining the first and last epoch.
                step = max(1, (len(part) + 1499) // 1500)
                sampled = part.iloc[::step]
                if sampled.index[-1] != part.index[-1]:
                    sampled = pd.concat([sampled, part.iloc[[-1]]])
                self.send_json({'trip_id': trip, 'total_epochs': len(part), 'sampled_epochs': len(sampled),
                                'data_kind': report['data_kind'], 'points': _records(sampled[list(dict.fromkeys(columns))])})
            elif parsed.path == '/api/config':
                self.send_json({'default_provider': 'offline', 'providers': ['offline', 'openai', 'deepseek', 'compatible'],
                                'notice': 'API mode uses locally configured credentials. Offline mode is a deterministic baseline.'})
            elif parsed.path == '/risk_plot.png':
                self.send_file((artifacts / 'risk_plot.png').read_bytes(), 'image/png')
            else:
                names = {'/': ('index.html', 'text/html; charset=utf-8'),
                         '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                         '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if parsed.path not in names:
                    self.send_json({'error': 'Not found'}, 404)
                    return
                name, kind = names[parsed.path]
                self.send_file(files('navdiag').joinpath('resources/dashboard', name).read_bytes(), kind)

        def send_file(self, body, content_type):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != '/api/ask':
                self.send_json({'error': 'Not found'}, 404)
                return
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if origin is not None and origin not in allowed:
                self.send_json({'error': 'Cross-origin requests are not allowed'}, 403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16000:
                    raise ValueError('Request body must contain 1-16000 bytes')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) - {'question', 'provider', 'trip_id'}:
                    raise ValueError('Unexpected request fields')
                if not isinstance(payload.get('question'), str):
                    raise ValueError('A text question is required')
                provider = payload.get('provider', 'offline')
                config = LLMConfig.from_environment(provider, env_file)
                response = ask(payload['question'], artifacts, config, payload.get('trip_id'))
                self.send_json(response)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({'error': str(error)}, 400)
            except RuntimeError as error:
                self.send_json({'error': str(error)}, 502)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def serve(artifacts: Path, port: int = 8765, env_file: Path | None = None):
    server = make_server(artifacts, port, env_file)
    print(f'NavDiag dashboard: http://127.0.0.1:{server.server_port} | {artifacts.resolve()}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

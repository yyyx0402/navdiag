"""Loopback-only dashboard for experiment results and evidence-backed diagnostics."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from navdiag.agent import LLMConfig, ModelResponseError, ask
from navdiag.llm_connections import ConnectionStore, connection_path


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient='records', double_precision=10))


def make_server(artifacts: Path, port: int = 8765, env_file: Path | None = None,
                nlos_artifacts: Path | None = None, generalization_artifacts: Path | None = None,
                connections_file: Path | None = None) -> ThreadingHTTPServer:
    artifacts = artifacts.resolve()
    report = json.loads((artifacts / 'report.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(artifacts / 'predictions.csv', dtype={'trip_id': 'string'})
    nlos_experiments, nlos_suite = {}, None
    if nlos_artifacts is not None:
        suite_path = nlos_artifacts / 'suite_report.json'
        if suite_path.exists():
            nlos_suite = json.loads(suite_path.read_text(encoding='utf-8'))
            if nlos_suite.get('task_type') != 'satellite_nlos_suite':
                raise ValueError('Invalid NLOS protocol suite')
            roots = {item['id']: nlos_artifacts / 'folds' / item['id'] for item in nlos_suite['experiments']}
            if any(not name.replace('_', '').isalnum() for name in roots):
                raise ValueError('Invalid experiment ID in protocol artifacts')
        else:
            roots = {'default': nlos_artifacts} if (nlos_artifacts / 'report.json').exists() else {}
        for name, root in roots.items():
            result = json.loads((root / 'report.json').read_text(encoding='utf-8'))
            if result.get('task_type') != 'satellite_nlos':
                raise ValueError('NLOS artifacts must contain a satellite_nlos experiment')
            nlos_experiments[name] = {'report': result, 'root': root,
                                      'predictions': pd.read_csv(root / 'predictions.csv', dtype={'trip_id': 'string'})}
    default_nlos = next(iter(nlos_experiments), None)
    connections = ConnectionStore(connections_file or connection_path(env_file))

    def nlos_context(experiment=None):
        return nlos_experiments.get(experiment or default_nlos)

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
            if not self.valid_local_request():
                return
            parsed, query = urlparse(self.path), parse_qs(urlparse(self.path).query)
            if parsed.path == '/api/report':
                self.send_json(report)
            elif parsed.path == '/api/nlos/report':
                context = nlos_context(query.get('experiment', [None])[0])
                if context is None and query.get('experiment'):
                    self.send_json({'error': 'Unknown NLOS experiment'}, 400)
                    return
                self.send_json({'available': context is not None, 'report': context['report'] if context else None,
                                'experiment_id': query.get('experiment', [default_nlos])[0],
                                'suite': nlos_suite})
            elif parsed.path == '/api/nlos/generalization':
                path = generalization_artifacts / 'generalization_report.json' if generalization_artifacts else None
                result = json.loads(path.read_text(encoding='utf-8')) if path and path.exists() else None
                if result and result.get('task_type') != 'satellite_nlos_generalization':
                    self.send_json({'error': 'Invalid generalization report'}, 500)
                    return
                self.send_json({'available': result is not None, 'report': result})
            elif parsed.path == '/api/nlos/epoch':
                context = nlos_context(query.get('experiment', [None])[0])
                if context is None:
                    self.send_json({'error': 'NLOS experiment not available'}, 404)
                    return
                nlos_predictions = context['predictions']
                trip = query.get('trip', [sorted(nlos_predictions.trip_id.unique())[0]])[0]
                part = nlos_predictions[nlos_predictions.trip_id == trip]
                times = sorted(part.timestamp_ms.unique())
                try:
                    index = int(query.get('index', ['0'])[0])
                except ValueError:
                    index = -1
                if not 0 <= index < len(times):
                    self.send_json({'error': 'Unknown NLOS trip or epoch index'}, 400)
                    return
                epoch = part[part.timestamp_ms == times[index]].sort_values('satellite_id')
                columns = ['satellite_id', 'system', 'cn0_dbhz', 'elevation_deg', 'signal_label',
                           'signal_label_source', 'obstruction_type', 'predicted_nlos_score',
                           'prediction_status', 'outcome']
                self.send_json({'trip_id': str(trip), 'total_epochs': len(times), 'epoch_index': index,
                                'time_s': float((times[index] - times[0]) / 1000),
                                'satellites': _records(epoch[[name for name in columns if name in epoch]])})
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
                try:
                    self.send_json({'default_provider': 'offline', 'providers': ['offline', 'connection', 'compatible'],
                                    'connections': connections.list(),
                                    'notice': 'Choose a saved connection or configure a compatible service. Credentials are never returned.'})
                except ValueError as error:
                    self.send_json({'error': str(error)}, 400)
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

        def valid_local_request(self):
            allowed_hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed_hosts:
                self.send_json({'error': 'Use the dashboard loopback address'}, 403)
                return False
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if (origin is not None and origin not in allowed) or self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.send_json({'error': 'Cross-origin requests are not allowed'}, 403)
                return False
            return True

        def do_POST(self):
            if not self.valid_local_request():
                return
            if self.path not in {'/api/ask', '/api/connections/save', '/api/connections/delete', '/api/connections/models'}:
                self.send_json({'error': 'Not found'}, 404)
                return
            try:
                if self.path != '/api/ask' and self.headers.get_content_type() != 'application/json':
                    raise ValueError('Connection requests require application/json')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16000:
                    raise ValueError('Request body must contain 1-16000 bytes')
                payload = json.loads(self.rfile.read(length))
                if self.path == '/api/connections/save':
                    self.send_json({'connection': connections.save(payload)})
                    return
                if self.path == '/api/connections/models':
                    self.send_json(connections.models(payload))
                    return
                if self.path == '/api/connections/delete':
                    if not isinstance(payload, dict) or set(payload) != {'id'}:
                        raise ValueError('Specify the connection ID to delete')
                    connections.delete(payload['id'])
                    self.send_json({'deleted': True})
                    return
                if not isinstance(payload, dict) or set(payload) - {'question', 'provider', 'connection_id', 'trip_id', 'nlos_experiment', 'task_context'}:
                    raise ValueError('Unexpected request fields')
                if not isinstance(payload.get('question'), str):
                    raise ValueError('A text question is required')
                provider = payload.get('provider', 'offline')
                experiment = payload.get('nlos_experiment')
                if experiment is not None and (not isinstance(experiment, str) or nlos_context(experiment) is None):
                    raise ValueError('Unknown NLOS experiment')
                context = nlos_context(experiment)
                if provider == 'connection':
                    config = connections.config(payload.get('connection_id'))
                else:
                    if payload.get('connection_id') is not None:
                        raise ValueError('A connection ID requires the connection provider')
                    config = LLMConfig.from_environment(provider, env_file)
                response = ask(payload['question'], artifacts, config, payload.get('trip_id'),
                               context['root'] if context else nlos_artifacts, generalization_artifacts,
                               payload.get('task_context'))
                self.send_json(response)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({'error': str(error)}, 400)
            except ModelResponseError as error:
                self.send_json({'error': str(error), 'error_code': error.code,
                                'model_diagnostics': error.diagnostics}, 502)
            except RuntimeError as error:
                self.send_json({'error': str(error)}, 502)
            except OSError:
                self.send_json({'error': 'Cannot read or save the local connection file; check directory access.'}, 500)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def serve(artifacts: Path, port: int = 8765, env_file: Path | None = None,
          nlos_artifacts: Path | None = None, generalization_artifacts: Path | None = None):
    server = make_server(artifacts, port, env_file, nlos_artifacts, generalization_artifacts)
    print(f'NavDiag dashboard: http://127.0.0.1:{server.server_port} | {artifacts.resolve()}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

"""Five complete neural classifiers sharing causal windows and training rules."""
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from navdiag.nlos_transformer import NLOSTransformer, TransformerPredictor, satellite_windows

NEURAL_MODELS = ('mlp', 'cnn', 'lstm', 'gru', 'transformer')


class HistoryMLP(nn.Module):
    """Dense classifier over the same flattened past/current window."""
    def __init__(self, features: int, window: int = 32, width: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks, inputs = [nn.Flatten()], features * window
        for hidden in [width * 2] + [width] * (layers - 1):
            blocks.extend([nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout)])
            inputs = hidden
        self.layers = nn.Sequential(*blocks, nn.Linear(inputs, 1))

    def forward(self, inputs):
        return self.layers(inputs).squeeze(-1)


class SatelliteCNN(nn.Module):
    """Temporal convolutions; summarize the full window plus its latest token."""
    def __init__(self, features: int, width: int = 64, dropout: float = 0.1,
                 kernel_size: int = 5, layers: int = 2):
        super().__init__()
        self.first = nn.Conv1d(features, width, kernel_size=kernel_size)
        self.second = nn.Conv1d(width, width, kernel_size=kernel_size) if layers > 1 else nn.Identity()
        self.extra = nn.ModuleList([nn.Conv1d(width, width, kernel_size=kernel_size) for _ in range(max(0, layers - 2))])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width * 2, 1)

    def forward(self, inputs):
        hidden = inputs.transpose(1, 2)
        hidden = self.dropout(F.relu(self.first(F.pad(hidden, (self.first.kernel_size[0] - 1, 0)))))
        for layer in [self.second, *getattr(self, 'extra', [])]:
            if isinstance(layer, nn.Conv1d):
                hidden = self.dropout(F.relu(layer(F.pad(hidden, (layer.kernel_size[0] - 1, 0)))))
        return self.output(torch.cat([hidden.mean(dim=2), hidden[:, :, -1]], dim=1)).squeeze(-1)


class SatelliteRNN(nn.Module):
    def __init__(self, features: int, architecture: str, width: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        recurrent = nn.LSTM if architecture == 'lstm' else nn.GRU
        self.recurrent = recurrent(features, width, num_layers=layers,
                                   dropout=dropout if layers > 1 else 0., batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, inputs):
        hidden, _ = self.recurrent(inputs)
        return self.output(hidden[:, -1]).squeeze(-1)


DEFAULT_NEURAL_CONFIG = {'window': 32, 'width': 64, 'layers': 2, 'dropout': 0.1,
                        'learning_rate': 0.0005, 'weight_decay': 0.01,
                        'batch_size': 128, 'heads': 4, 'kernel_size': 5}


def neural_config(config: dict | None = None) -> dict:
    config = config or {}
    unknown = set(config) - set(DEFAULT_NEURAL_CONFIG)
    if unknown:
        raise ValueError(f'Unknown neural hyperparameters: {sorted(unknown)}')
    result = {**DEFAULT_NEURAL_CONFIG, **config}
    for key in ['window', 'width', 'layers', 'batch_size', 'heads', 'kernel_size']:
        if not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['dropout', 'learning_rate', 'weight_decay']:
        if not np.isfinite(result[key]):
            raise ValueError(f'{key} must be finite')
    if not 0 <= result['dropout'] < 1 or result['learning_rate'] <= 0 or result['weight_decay'] < 0:
        raise ValueError('Invalid dropout, learning rate or weight decay')
    if result['width'] % 2 or result['width'] % result['heads']:
        raise ValueError('width must be even and divisible by heads')
    return result


def build_network(architecture: str, features: int, window: int = 32,
                  config: dict | None = None) -> nn.Module:
    cfg = neural_config({**(config or {}), 'window': window})
    if architecture == 'mlp':
        return HistoryMLP(features, window, cfg['width'], cfg['layers'], cfg['dropout'])
    if architecture == 'cnn':
        return SatelliteCNN(features, cfg['width'], cfg['dropout'], cfg['kernel_size'], cfg['layers'])
    if architecture in {'lstm', 'gru'}:
        return SatelliteRNN(features, architecture, cfg['width'], cfg['layers'], cfg['dropout'])
    if architecture == 'transformer':
        return NLOSTransformer(features, window, cfg['width'], cfg['heads'], cfg['layers'], cfg['dropout'])
    raise ValueError(f'Unknown neural architecture: {architecture}')


@dataclass
class NeuralPredictor(TransformerPredictor):
    architecture: str = 'transformer'


def train_neural(train: pd.DataFrame, validation: pd.DataFrame, features: list[str],
                 architecture: str, seed: int = 42, epochs: int = 30, window: int = 32,
                 config: dict | None = None):
    """Train one specified configuration using train/validation only."""
    if epochs < 1:
        raise ValueError('Neural epochs must be positive')
    cfg = neural_config({'window': window, **(config or {})})
    window = cfg['window']
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    imputer = SimpleImputer(strategy='median').fit(train[features])
    scaler = StandardScaler().fit(imputer.transform(train[features]))

    def labelled_windows(frame):
        values = scaler.transform(imputer.transform(frame[features]))
        known = frame.signal_label.isin(['los', 'nlos']).to_numpy()
        return satellite_windows(frame, values, window)[known], (
            frame.loc[known, 'signal_label'].eq('nlos').to_numpy(dtype=np.float32))

    x_train, y_train = labelled_windows(train)
    x_val, y_val = labelled_windows(validation)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError('Neural train and validation need both LOS and NLOS')
    network = build_network(architecture, len(features), window, cfg)
    objective = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(
        (len(y_train) - y_train.sum()) / y_train.sum())))
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=cfg['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_state, best_loss, best_epoch, patience, history = None, float('inf'), 0, 0, []
    for epoch in range(epochs):
        network.train()
        total = 0.
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = objective(network(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        network.eval()
        with torch.no_grad():
            total_val = 0.
            for start in range(0, len(x_val), 256):
                inputs = torch.from_numpy(x_val[start:start + 256])
                labels = torch.from_numpy(y_val[start:start + 256])
                total_val += float(objective(network(inputs), labels)) * len(labels)
            val_loss = total_val / len(y_val)
        history.append({'epoch': epoch + 1, 'training_loss': total / len(y_train),
                        'validation_loss': val_loss})
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, patience = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            patience += 1
            if patience >= 6:
                break
    network.load_state_dict(best_state)
    predictor = NeuralPredictor(network, imputer, scaler, features, window, architecture)
    configurations = {
        'mlp': {'hidden_widths': [cfg['width'] * 2] + [cfg['width']] * (cfg['layers'] - 1), 'input': 'flattened history'},
        'cnn': {'channels': [cfg['width']] * cfg['layers'], 'kernel_sizes': [cfg['kernel_size']] * cfg['layers'],
                'summary': 'mean over history plus last token', 'padding': 'left only'},
        'lstm': {'hidden_size': cfg['width'], 'layers': cfg['layers'], 'direction': 'forward only'},
        'gru': {'hidden_size': cfg['width'], 'layers': cfg['layers'], 'direction': 'forward only'},
        'transformer': {'width': cfg['width'], 'heads': cfg['heads'], 'layers': cfg['layers'], 'feedforward': cfg['width'] * 2},
    }
    metadata = {
        'architecture': architecture, 'architecture_settings': configurations[architecture],
        'torch_version': torch.__version__, 'device': 'cpu', 'window_observations': window,
        'configuration': cfg, 'dropout': cfg['dropout'], 'parameters': sum(parameter.numel() for parameter in network.parameters()),
        'optimizer': 'AdamW', 'learning_rate': cfg['learning_rate'], 'weight_decay': cfg['weight_decay'], 'batch_size': cfg['batch_size'],
        'maximum_epochs': epochs, 'best_epoch': best_epoch, 'history': history,
        'checkpoint_policy': 'Lowest class-weighted validation BCE; patience 6; threshold chosen afterwards on validation F1',
        'context_policy': 'Same trip/device/satellite; past/current only; reset on >5000ms gaps; left-pad first observation; unknown labels may supply observed history',
    }
    return predictor, metadata

"""Causal per-satellite Transformer for LOS/NLOS, with train-only scaling."""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn


def satellite_windows(frame: pd.DataFrame, values: np.ndarray, window: int = 32,
                      max_gap_ms: float = 5000) -> np.ndarray:
    """Past/current observations only; isolate satellite, receiver, run and gaps.

    At warm-up repeat the first available observation on the left. These are
    observation windows, not fixed-duration seconds. Preserve input row order.
    """
    if window < 1 or len(frame) != len(values):
        raise ValueError('Positive window and matching observation/value rows required')
    positions = frame.reset_index(drop=True)
    output = np.empty((len(frame), window, values.shape[1]), dtype=np.float32)
    for _, indices in positions.groupby(['trip_id', 'device_id', 'satellite_id'], sort=False).groups.items():
        ordered = positions.loc[indices].sort_values('timestamp_ms').index.to_numpy()
        history, previous = [], None
        for index in ordered:
            current = positions.at[index, 'timestamp_ms']
            if previous is not None and current - previous > max_gap_ms:
                history = []
            history.append(values[index])
            history = history[-window:]
            output[index] = np.asarray([history[0]] * (window - len(history)) + history)
            previous = current
    return output


class NLOSTransformer(nn.Module):
    def __init__(self, features: int, window: int = 32, width: int = 64,
                 heads: int = 4, layers: int = 2, dropout: float = 0.1,
                 feedforward: int | None = None):
        super().__init__()
        self.input = nn.Linear(features, width)
        # Instantiate layers separately: each starts with independent weights.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=feedforward or width * 2,
                                       dropout=dropout, batch_first=True, activation='gelu')
            for _ in range(layers)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        position = torch.arange(window).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        encoding = torch.zeros(window, width)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer('position', encoding.unsqueeze(0))

    def encode(self, inputs):
        length = inputs.shape[1]
        hidden = self.input(inputs) + self.position[:, :length]
        mask = torch.ones(length, length, dtype=torch.bool, device=inputs.device).triu(1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask)
        return hidden

    def forward(self, inputs):
        return self.output(self.encode(inputs)[:, -1]).squeeze(-1)


@dataclass
class TransformerPredictor:
    network: NLOSTransformer
    imputer: SimpleImputer
    scaler: StandardScaler
    features: list[str]
    window: int = 32

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.scaler.transform(self.imputer.transform(frame[self.features]))
        windows = satellite_windows(frame, values, self.window)
        self.network.eval()
        with torch.no_grad():
            scores = np.concatenate([
                torch.sigmoid(self.network(torch.from_numpy(windows[start:start + 256]))).numpy()
                for start in range(0, len(windows), 256)
            ])
        return np.column_stack([1 - scores, scores])


def train_transformer(train: pd.DataFrame, validation: pd.DataFrame, features: list[str],
                      seed: int = 42, epochs: int = 30, window: int = 32):
    """Compatibility entry point; training rules are shared by all neural models."""
    from navdiag.nlos_neural import train_neural
    return train_neural(train, validation, features, 'transformer', seed, epochs, window)

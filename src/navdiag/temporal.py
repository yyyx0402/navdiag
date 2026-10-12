"""Small causal GRU with train-only preprocessing and validation early stopping."""
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def causal_windows(frame: pd.DataFrame, values: np.ndarray, window: int = 10, max_gap_ms: float = 5000) -> np.ndarray:
    """Windows end at the current epoch; reset on devices, trips, and time gaps."""
    if window < 1:
        raise ValueError('window must be positive')
    output = np.empty((len(frame), window, values.shape[1]), dtype=np.float32)
    keys = ['trip_id'] + (['device_id'] if 'device_id' in frame else [])
    positions = frame.reset_index(drop=True)
    for _, indices in positions.groupby(keys, sort=False).groups.items():
        ordered = positions.loc[indices].sort_values('timestamp_ms').index.to_numpy()
        history, previous_time = [], None
        for index in ordered:
            current_time = positions.loc[index, 'timestamp_ms']
            if previous_time is not None and current_time - previous_time > max_gap_ms:
                history = []
            history.append(values[index])
            history = history[-window:]
            padded = [history[0]] * (window - len(history)) + history
            output[index] = np.asarray(padded, dtype=np.float32)
            previous_time = current_time
    return output


class QualityGRU(nn.Module):
    def __init__(self, features: int, hidden: int = 24):
        super().__init__()
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, inputs):
        _, hidden = self.gru(inputs)
        return self.output(hidden[-1]).squeeze(-1)


@dataclass
class GRUPredictor:
    network: QualityGRU
    imputer: SimpleImputer
    scaler: StandardScaler
    features: list[str]
    window: int = 10

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.scaler.transform(self.imputer.transform(frame[self.features]))
        windows = causal_windows(frame, values, self.window)
        self.network.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(windows), 512):
                scores.append(torch.sigmoid(self.network(torch.from_numpy(windows[start:start + 512]))).numpy())
        probabilities = np.concatenate(scores)
        return np.column_stack([1 - probabilities, probabilities])


def train_gru(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], error_limit_m: float,
              seed: int = 42, epochs: int = 20, window: int = 10,
              train_context: pd.DataFrame | None = None,
              validation_context: pd.DataFrame | None = None) -> tuple[GRUPredictor, list[dict]]:
    if epochs < 1:
        raise ValueError('GRU epochs must be positive')
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    imputer = SimpleImputer(strategy='median').fit(train[features])
    scaler = StandardScaler().fit(imputer.transform(train[features]))
    from navdiag.context import context_indices
    train_context = train if train_context is None else train_context
    validation_context = validation if validation_context is None else validation_context
    x_train = causal_windows(train_context, scaler.transform(imputer.transform(train_context[features])), window)[context_indices(train, train_context)]
    x_val = causal_windows(validation_context, scaler.transform(imputer.transform(validation_context[features])), window)[context_indices(validation, validation_context)]
    y_train = (train.horizontal_error_m.to_numpy() > error_limit_m).astype(np.float32)
    y_val = (validation.horizontal_error_m.to_numpy() > error_limit_m).astype(np.float32)
    network = QualityGRU(len(features))
    positive_weight = torch.tensor(float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1)))
    objective = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_state, best_loss, patience, history = None, float('inf'), 0, []
    for epoch in range(epochs):
        network.train()
        losses = []
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = objective(network(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        network.eval()
        with torch.no_grad():
            val_loss = float(objective(network(torch.from_numpy(x_val)), torch.from_numpy(y_val)))
        history.append({'epoch': epoch + 1, 'training_loss': float(np.mean(losses)), 'validation_loss': val_loss})
        if val_loss < best_loss - 1e-4:
            best_loss, patience, best_state = val_loss, 0, copy.deepcopy(network.state_dict())
        else:
            patience += 1
            if patience >= 5:
                break
    network.load_state_dict(best_state)
    return GRUPredictor(network, imputer, scaler, features, window), history

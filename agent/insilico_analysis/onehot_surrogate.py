"""
One-hot MLP ensemble surrogate for in-silico benchmarking baselines.

A lightweight surrogate (one-hot encoder + simple MLP ensemble) that can be
trained/retrained each BO round on small datasets, used for the onehot_mlp
baseline search methods.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple, Dict
from copy import deepcopy

_AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


def _one_hot_encode(sequences: List[str], alphabet: List[str] = _AA_ALPHABET, pad_to_length: Optional[int] = None) -> np.ndarray:
    seqs = [s.replace("*", "") for s in sequences]
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    Lmax = max(len(s) for s in seqs) if pad_to_length is None else int(pad_to_length)
    A = len(alphabet)
    out = np.zeros((len(seqs), Lmax * A), dtype=np.float32)
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    for n, s in enumerate(seqs):
        for pos, aa in enumerate(s):
            j = aa_to_idx.get(aa)
            if j is not None:
                out[n, pos * A + j] = 1.0
    return out


class _SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_one_mlp(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    num_layers: int = 2,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> _SimpleMLP:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _SimpleMLP(input_dim=X.shape[1], hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


class OneHotMLPSurrogate:
    """Ensemble of one-hot MLPs for use as a lightweight surrogate in BO baselines."""

    def __init__(
        self,
        n_ensemble: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: Optional[str] = None,
    ):
        self.n_ensemble = n_ensemble
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: List[_SimpleMLP] = []
        self._pad_length: Optional[int] = None

    def fit(self, sequences: List[str], fitness_values) -> None:
        """One-hot encode sequences, train N MLP ensemble members."""
        X = _one_hot_encode(sequences)
        y = np.array(fitness_values, dtype=np.float32)
        self._pad_length = X.shape[1] // len(_AA_ALPHABET)
        self.models = []
        for i in range(self.n_ensemble):
            model = _train_one_mlp(
                X, y, seed=i,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                num_layers=self.num_layers,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
            )
            self.models.append(model)

    def predict(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) across ensemble for each sequence."""
        X = _one_hot_encode(sequences, pad_to_length=self._pad_length)
        X_t = torch.from_numpy(X).float().to(self.device)
        preds = []
        for model in self.models:
            with torch.no_grad():
                preds.append(model(X_t).cpu().numpy())
        preds = np.stack(preds)  # (n_ensemble, n_sequences)
        return preds.mean(axis=0), preds.std(axis=0)

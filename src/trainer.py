import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader,
               criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item() * len(y)
    return total_loss / len(loader.dataset)


def fit(model: nn.Module, loaders: dict, epochs: int, lr: float,
        patience: int, device: torch.device) -> dict:
    """
    Full training loop with early stopping and LR scheduler.
    Returns history dict with train/val loss per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2
    )
    criterion = nn.MSELoss()
    stopper  = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        tr_loss  = train_epoch(model, loaders['train'], optimizer, criterion, device)
        val_loss = eval_epoch(model, loaders['val'],   criterion, device)

        scheduler.step(val_loss)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)

        if stopper.step(val_loss, model):
            break

    stopper.restore(model)
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for X, _ in loader:
        preds.append(model(X.to(device)).cpu().numpy())
    return np.concatenate(preds)

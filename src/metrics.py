import numpy as np


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def mae(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(obs - pred)))


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))


def willmott_index(obs: np.ndarray, pred: np.ndarray) -> float:
    obs_mean = obs.mean()
    num = np.sum((obs - pred) ** 2)
    den = np.sum((np.abs(pred - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    return float(1 - num / den)


def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    return {
        'RMSE': rmse(obs, pred),
        'MAE':  mae(obs, pred),
        'NSE':  nse(obs, pred),
        'WI':   willmott_index(obs, pred),
    }

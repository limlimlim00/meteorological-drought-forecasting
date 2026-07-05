"""
Drought forecasting experiment runner.

Example
-------
python src/experiment.py \
    --model lstm \
    --station 군산 \
    --target SPI3 \
    --seq_len 4 \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.1 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --patience 20
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from dataset import load_splits, make_loaders
from metrics import compute_metrics
from trainer import fit, predict

# ── helpers ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description='SPI drought forecasting')

    # Data
    p.add_argument('--data_path', default='data/monthly_SPI_1991_2025.xlsx')
    p.add_argument('--station',   default='군산', choices=['군산', '전주', '부안'])
    p.add_argument('--target',    default='SPI3',
                   choices=['SPI1','SPI2','SPI3','SPI4','SPI5',
                            'SPI6','SPI9','SPI12','SPI18','SPI24'])
    p.add_argument('--seq_len',   type=int, default=4,
                   help='Look-back window length (months)')
    p.add_argument('--features',  type=str, default='all',
                   help='Comma-separated SPI columns to use as input, or "all" (default)')

    # Model
    p.add_argument('--model', required=True,
                   choices=['lstm', 'bilstm', 'cnn', 'tcn',
                            'transformer', 'tft', 'cnn_lstm'])
    p.add_argument('--hidden_size', type=int,   default=64)
    p.add_argument('--num_layers',  type=int,   default=2)
    p.add_argument('--dropout',     type=float, default=0.1)

    # CNN / TCN-specific
    p.add_argument('--kernel_size',  type=int, default=3)
    p.add_argument('--num_channels', type=str, default='64,64,64',
                   help='Comma-separated channel sizes per TCN level')

    # CNN-LSTM-specific
    p.add_argument('--cnn_channels', type=str, default='8,8',
                   help='Comma-separated CNN channel sizes (CNN-LSTM)')
    p.add_argument('--lstm_hidden',  type=int, default=9,
                   help='LSTM hidden size (CNN-LSTM)')
    p.add_argument('--lstm_layers',  type=int, default=3,
                   help='LSTM layer count (CNN-LSTM)')
    p.add_argument('--fc_hidden',    type=int, default=8,
                   help='Intermediate Dense size (CNN-LSTM)')

    # Transformer / TFT-specific
    p.add_argument('--d_model',      type=int,   default=64)
    p.add_argument('--num_heads',    type=int,   default=4)
    p.add_argument('--attn_dropout', type=float, default=0.0)

    # Training
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--epochs',     type=int,   default=200)
    p.add_argument('--patience',   type=int,   default=20)
    p.add_argument('--seed',       type=int,   default=42)

    # Output
    p.add_argument('--output_dir', default='results/experiments')

    return p.parse_args()


def build_model(args, input_size: int):
    """Instantiate the requested model."""
    kw = dict(
        input_size  = input_size,
        seq_len     = args.seq_len,
        hidden_size = args.hidden_size,
        num_layers  = args.num_layers,
        dropout     = args.dropout,
    )

    if args.model == 'lstm':
        from models.lstm import LSTM
        return LSTM(**kw)

    if args.model == 'bilstm':
        from models.bilstm import BiLSTM
        return BiLSTM(**kw)

    if args.model == 'cnn':
        from models.cnn import CNN
        return CNN(input_size=input_size, hidden_size=args.hidden_size,
                   num_layers=args.num_layers, kernel_size=args.kernel_size,
                   dropout=args.dropout)

    if args.model == 'tcn':
        from models.tcn import TCN
        channels = [int(c) for c in args.num_channels.split(',')]
        return TCN(input_size=input_size, num_channels=channels,
                   kernel_size=args.kernel_size, dropout=args.dropout)

    if args.model == 'transformer':
        from models.transformer import Transformer
        return Transformer(input_size=input_size, d_model=args.d_model,
                           num_heads=args.num_heads, num_layers=args.num_layers,
                           dropout=args.dropout)

    if args.model == 'tft':
        from models.tft import TFT
        return TFT(input_size=input_size, seq_len=args.seq_len,
                   d_model=args.d_model, num_layers=args.num_layers,
                   num_heads=args.num_heads, dropout=args.dropout,
                   attn_dropout=args.attn_dropout)

    if args.model == 'cnn_lstm':
        from models.cnn_lstm import CNNLSTM
        cnn_ch = [int(c) for c in args.cnn_channels.split(',')]
        return CNNLSTM(input_size=input_size,
                       cnn_channels=cnn_ch,
                       kernel_size=args.kernel_size,
                       lstm_hidden=args.lstm_hidden,
                       lstm_layers=args.lstm_layers,
                       fc_hidden=args.fc_hidden,
                       dropout=args.dropout)

    raise ValueError(f'Unknown model: {args.model}')


def feature_tag(args) -> str:
    """Short tag representing the input feature set."""
    if args.features == 'all':
        return ''               # 기존 실험과 호환 (태그 없음)
    cols = args.features.split(',')
    if len(cols) == 1:
        return f'_f{cols[0]}'  # 단일 스케일: _fSPI3
    return f'_f{len(cols)}sc'  # 복수 선택: _f3sc


def run_id(args) -> str:
    """Human-readable identifier for this experiment configuration."""
    ft = feature_tag(args)

    if args.model == 'cnn':
        return (f"{args.model}_{args.station}_{args.target}"
                f"_sl{args.seq_len}_hs{args.hidden_size}_nl{args.num_layers}"
                f"_ks{args.kernel_size}_dr{args.dropout}_lr{args.lr}{ft}")
    if args.model == 'cnn_lstm':
        ch = args.cnn_channels.replace(',', '-')
        return (f"cnn_lstm_{args.station}_{args.target}"
                f"_sl{args.seq_len}_ch{ch}_lh{args.lstm_hidden}"
                f"_ll{args.lstm_layers}_ks{args.kernel_size}_dr{args.dropout}_lr{args.lr}{ft}")
    if args.model == 'tcn':
        ch = args.num_channels.replace(',', '-')
        return (f"{args.model}_{args.station}_{args.target}"
                f"_sl{args.seq_len}_ch{ch}_ks{args.kernel_size}"
                f"_dr{args.dropout}_lr{args.lr}{ft}")
    if args.model in ('transformer', 'tft'):
        return (f"{args.model}_{args.station}_{args.target}"
                f"_sl{args.seq_len}_dm{args.d_model}_nh{args.num_heads}"
                f"_nl{args.num_layers}_dr{args.dropout}_lr{args.lr}{ft}")
    return (f"{args.model}_{args.station}_{args.target}"
            f"_sl{args.seq_len}_hs{args.hidden_size}_nl{args.num_layers}"
            f"_dr{args.dropout}_lr{args.lr}_bs{args.batch_size}{ft}")


def inverse_target(scaler, arr: np.ndarray, target_idx: int) -> np.ndarray:
    """Inverse-transform only the target column."""
    dummy = np.zeros((len(arr), scaler.n_features_in_), dtype=np.float32)
    dummy[:, target_idx] = arr
    return scaler.inverse_transform(dummy)[:, target_idx]


# ── main ────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── load data ────────────────────────────────────────────────────────────
    data_path = ROOT / args.data_path

    # feature_cols 해석: 'all' → None(전체), 그 외 → 쉼표 구분 리스트
    feature_cols = None if args.features == 'all' else args.features.split(',')

    splits, scaler, target_idx, feat_cols = load_splits(
        str(data_path), args.station, args.target, args.seq_len,
        feature_cols=feature_cols
    )
    input_size = splits['train'][0].shape[-1]

    # ── output directory ─────────────────────────────────────────────────────
    rid      = run_id(args)
    out_dir  = ROOT / args.output_dir / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── build model ───────────────────────────────────────────────────────────
    model = build_model(args, input_size)

    t0 = time.time()

    # ── train ─────────────────────────────────────────────────────────────────
    model = model.to(device)
    loaders = make_loaders(splits, args.batch_size)
    history = fit(model, loaders, args.epochs, args.lr, args.patience, device)

    elapsed = time.time() - t0

    # ── evaluate ──────────────────────────────────────────────────────────────
    all_metrics = {}
    all_preds   = {}

    for split_name, (X, y_sc, dates) in splits.items():
        from dataset import TimeSeriesDataset
        from torch.utils.data import DataLoader
        ds     = TimeSeriesDataset(X, y_sc)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        pred_sc = predict(model, loader, device)

        # Inverse-transform to original SPI scale
        obs  = inverse_target(scaler, y_sc,    target_idx)
        pred = inverse_target(scaler, pred_sc, target_idx)

        all_metrics[split_name] = compute_metrics(obs, pred)
        all_preds[split_name]   = {'dates': dates, 'obs': obs, 'pred': pred}

    # ── save predictions ──────────────────────────────────────────────────────
    import pandas as pd
    pred_rows = []
    for split_name, d in all_preds.items():
        for date, obs, pred in zip(d['dates'], d['obs'], d['pred']):
            pred_rows.append({'split': split_name, 'date': date,
                              'obs': obs, 'pred': pred})
    pd.DataFrame(pred_rows).to_csv(out_dir / 'predictions.csv', index=False)

    # ── save metrics ──────────────────────────────────────────────────────────
    result = {
        'run_id':   rid,
        'model':    args.model,
        'station':  args.station,
        'target':   args.target,
        'seq_len':  args.seq_len,
        'hidden_size': args.hidden_size,
        'num_layers':  args.num_layers,
        'dropout':     args.dropout,
        'lr':          args.lr,
        'batch_size':  args.batch_size,
        'kernel_size': args.kernel_size,
        'num_channels': args.num_channels,
        'd_model':     args.d_model,
        'num_heads':   args.num_heads,
        'attn_dropout': args.attn_dropout,
        'seed':        args.seed,
        'elapsed_sec': round(elapsed, 1),
        'epochs_run':  len(history.get('train_loss', [])),
    }
    for split_name, m in all_metrics.items():
        for metric, val in m.items():
            result[f'{split_name}_{metric}'] = round(val, 5)

    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── append to master CSV ──────────────────────────────────────────────────
    master_csv = ROOT / args.output_dir / 'results.csv'
    write_header = not master_csv.exists()
    with open(master_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    # ── save model checkpoint ─────────────────────────────────────────────────
    torch.save(model.state_dict(), out_dir / 'model.pt')

    # ── print summary ─────────────────────────────────────────────────────────
    print(f'\n[{rid}]')
    for split_name, m in all_metrics.items():
        print(f"  {split_name:5s} │ "
              + '  '.join(f"{k}={v:.4f}" for k, v in m.items()))
    print(f"  elapsed: {elapsed:.1f}s  |  saved → {out_dir}")


if __name__ == '__main__':
    main()

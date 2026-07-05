"""
Exp3 XAI 분석 스크립트
SHAP, Integrated Gradients, TFT VSN+Attention
대상: 부안 / SPI3, SPI6, SPI12, SPI24 / BiLSTM, TCN, TFT
"""
import sys, os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from dataset import load_splits
from models.bilstm import BiLSTM
from models.tcn    import TCN
from models.tft    import TFT
import shap
from scipy.stats import spearmanr

DATA_PATH  = os.path.join(ROOT, 'data', 'monthly_SPI_1991_2025.xlsx')
EXP_DIR    = os.path.join(ROOT, 'results', 'experiments')
STATION    = '부안'
FEAT_NAMES = ['SPI1','SPI2','SPI3','SPI4','SPI5','SPI6','SPI9','SPI12','SPI18','SPI24']
N_FEAT     = len(FEAT_NAMES)
TARGETS    = ['SPI3', 'SPI6', 'SPI12', 'SPI24']

RUNS = {
    'SPI3':  {
        'BiLSTM': 'bilstm_부안_SPI3_sl6_hs128_nl2_dr0.1_lr0.001_bs32',
        'TCN'   : 'tcn_부안_SPI3_sl4_ch64-64-64-64_ks3_dr0.0_lr0.0005',
        'TFT'   : 'tft_부안_SPI3_sl12_dm32_nh4_nl2_dr0.0_lr0.001',
    },
    'SPI6':  {
        'BiLSTM': 'bilstm_부안_SPI6_sl4_hs32_nl2_dr0.1_lr0.001_bs32',
        'TCN'   : 'tcn_부안_SPI6_sl6_ch64-64-64_ks5_dr0.1_lr0.0005',
        'TFT'   : 'tft_부안_SPI6_sl4_dm32_nh4_nl1_dr0.0_lr0.001',
    },
    'SPI12': {
        'BiLSTM': 'bilstm_부안_SPI12_sl4_hs32_nl2_dr0.0_lr0.001_bs32',
        'TCN'   : 'tcn_부안_SPI12_sl12_ch32-32-32_ks5_dr0.1_lr0.0005',
        'TFT'   : 'tft_부안_SPI12_sl4_dm32_nh4_nl2_dr0.0_lr0.001',
    },
    'SPI24': {
        'BiLSTM': 'bilstm_부안_SPI24_sl6_hs128_nl1_dr0.0_lr0.001_bs32',
        'TCN'   : 'tcn_부안_SPI24_sl6_ch64-64-64_ks5_dr0.0_lr0.0005',
        'TFT'   : 'tft_부안_SPI24_sl6_dm128_nh4_nl1_dr0.0_lr0.0005',
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def build_model(m):
    name = m['model']
    if name == 'bilstm':
        return BiLSTM(N_FEAT, m['hidden_size'], m['num_layers'], m['dropout'])
    if name == 'tcn':
        ch = [int(c) for c in m['num_channels'].split(',')]
        return TCN(N_FEAT, ch, m['kernel_size'], m['dropout'])
    if name == 'tft':
        return TFT(N_FEAT, m['seq_len'], m['d_model'], m['num_layers'],
                   m['num_heads'], m['dropout'], m['attn_dropout'])

def load_all(run_id, target):
    with open(f'{EXP_DIR}/{run_id}/metrics.json') as f:
        m = json.load(f)
    model = build_model(m)
    state = torch.load(f'{EXP_DIR}/{run_id}/model.pt',
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    splits, _, _, _ = load_splits(DATA_PATH, STATION, target, m['seq_len'])
    return model, torch.FloatTensor(splits['train'][0]), torch.FloatTensor(splits['test'][0]), m

class Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x).unsqueeze(-1)

def compute_shap(model, X_tr, X_te, n_bg=100):
    torch.manual_seed(42)
    bg = X_tr[torch.randperm(len(X_tr))[:n_bg]]
    sv = shap.GradientExplainer(Wrapper(model), bg).shap_values(X_te)
    if isinstance(sv, list): sv = np.array(sv[0])
    else: sv = np.array(sv)
    if sv.ndim == 4: sv = sv[..., 0]
    return sv

def integrated_gradients(model, X_te, X_tr, n_steps=100):
    baseline = X_tr.mean(dim=0, keepdim=True)
    all_ig = []
    for i in range(len(X_te)):
        x = X_te[[i]]
        alphas = torch.linspace(0, 1, n_steps).view(-1, 1, 1)
        interp = (baseline + alphas * (x - baseline)).requires_grad_(True)
        model(interp).sum().backward()
        avg_g = interp.grad.detach().mean(0)
        all_ig.append(((x.squeeze(0) - baseline.squeeze(0)) * avg_g).numpy())
    return np.array(all_ig)

def normalize(arr): return arr / arr.sum()
def feat_imp(arr): return np.abs(arr).sum(axis=1).mean(axis=0)

# ── load & compute ────────────────────────────────────────────────────────────
results = {t: {} for t in TARGETS}

for target in TARGETS:
    for model_name, rid in RUNS[target].items():
        print(f"Loading {target} {model_name}...", flush=True)
        model, X_tr, X_te, m = load_all(rid, target)
        results[target][model_name] = {'model': model, 'X_tr': X_tr, 'X_te': X_te, 'm': m}

for target in TARGETS:
    for model_name in ['BiLSTM', 'TCN']:
        print(f"SHAP {target} {model_name}...", end=' ', flush=True)
        r = results[target][model_name]
        sv = compute_shap(r['model'], r['X_tr'], r['X_te'])
        results[target][model_name]['shap'] = sv
        print(f"shape={sv.shape}")

for target in TARGETS:
    for model_name in ['BiLSTM', 'TCN']:
        print(f"IG {target} {model_name}...", end=' ', flush=True)
        r = results[target][model_name]
        ig = integrated_gradients(r['model'], r['X_te'], r['X_tr'])
        results[target][model_name]['ig'] = ig
        print(f"shape={ig.shape}")

for target in TARGETS:
    tft = results[target]['TFT']['model']
    with torch.no_grad():
        _ = tft(results[target]['TFT']['X_te'])
    results[target]['TFT']['vsn']  = tft.get_vsn_weights().numpy()
    results[target]['TFT']['attn'] = tft.get_attention_weights().numpy()

# ── print results ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FEATURE IMPORTANCE (normalized, SHAP / IG / TFT VSN)")
print("="*70)
header = f"{'Feature':<8}" + "".join(f"{'S_BL':>8}{'I_BL':>8}{'S_TC':>8}{'I_TC':>8}{'VSN':>8}" for _ in [1])
for target in TARGETS:
    print(f"\n--- {target} ---")
    print(f"{'Feature':<8}{'SHAP_BL':>9}{'IG_BL':>9}{'SHAP_TCN':>10}{'IG_TCN':>9}{'TFT_VSN':>9}")
    s_bl  = normalize(feat_imp(results[target]['BiLSTM']['shap']))
    g_bl  = normalize(feat_imp(results[target]['BiLSTM']['ig']))
    s_tcn = normalize(feat_imp(results[target]['TCN']['shap']))
    g_tcn = normalize(feat_imp(results[target]['TCN']['ig']))
    vsn   = normalize(results[target]['TFT']['vsn'].mean(axis=(0, 1)))
    for i, fn in enumerate(FEAT_NAMES):
        print(f"{fn:<8}{s_bl[i]:>9.4f}{g_bl[i]:>9.4f}{s_tcn[i]:>10.4f}{g_tcn[i]:>9.4f}{vsn[i]:>9.4f}")

print("\n" + "="*70)
print("SPEARMAN rho: SHAP(BiLSTM) vs IG(BiLSTM)")
for target in TARGETS:
    s = normalize(feat_imp(results[target]['BiLSTM']['shap']))
    g = normalize(feat_imp(results[target]['BiLSTM']['ig']))
    rho, p = spearmanr(s, g)
    print(f"  {target}: rho={rho:.3f} p={p:.3f}")

print("\nSPEARMAN rho: SHAP(BiLSTM) vs SHAP(TCN)")
for target in TARGETS:
    s = normalize(feat_imp(results[target]['BiLSTM']['shap']))
    t = normalize(feat_imp(results[target]['TCN']['shap']))
    rho, p = spearmanr(s, t)
    print(f"  {target}: rho={rho:.3f} p={p:.3f}")

print("\nSPEARMAN rho: SHAP(BiLSTM) vs TFT VSN")
for target in TARGETS:
    s = normalize(feat_imp(results[target]['BiLSTM']['shap']))
    v = normalize(results[target]['TFT']['vsn'].mean(axis=(0, 1)))
    rho, p = spearmanr(s, v)
    print(f"  {target}: rho={rho:.3f} p={p:.3f}")

print("\n" + "="*70)
print("TFT ATTENTION (last timestep)")
for target in TARGETS:
    attn_w = results[target]['TFT']['attn']
    sl = results[target]['TFT']['m']['seq_len']
    last = attn_w[:, :, -1, :].mean(axis=(0, 1))
    top = sorted(range(sl), key=lambda i: last[i], reverse=True)[:3]
    print(f"  {target} sl={sl}: " + ", ".join(f"t-{sl-i}={last[i]:.3f}" for i in top))

print("\n" + "="*70)
print("LAG IMPORTANCE (SHAP, BiLSTM)")
for target in TARGETS:
    sv = results[target]['BiLSTM']['shap']
    sl = results[target]['BiLSTM']['m']['seq_len']
    lag = normalize(np.abs(sv).mean(axis=0).sum(axis=1))
    top = sorted(range(sl), key=lambda i: lag[i], reverse=True)[:3]
    print(f"  {target} sl={sl}: " + ", ".join(f"t-{sl-i}={lag[i]:.3f}" for i in top))

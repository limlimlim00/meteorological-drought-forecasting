# 다중 스케일 SPI 입력 기반 딥러닝 기상 가뭄 예측 및 XAI 분석

전라북도 군산·전주·부안 (1991–2025) | 관개배수공학 기말 프로젝트

---

## 개요

전라북도 3개 관측소의 35년치 월별 SPI 자료를 이용해 **1개월 선행 기상 가뭄 예측** 딥러닝 모델 7종을 비교하고, SHAP·Integrated Gradients·TFT 내장 해석(VSN, Interpretable Attention)으로 예측 근거를 분석한 연구입니다.

**핵심 질문**: 예측 타깃과 동일한 SPI 스케일만 입력으로 쓰는 *단일 스케일* 방식과, SPI1~SPI24 전체 10개 스케일을 동시에 입력하는 *다중 스케일* 방식 중 어느 쪽이 더 좋은가? 그리고 왜 그런가?

---

## 실험 구성

| 단계 | 내용 | 규모 |
|------|------|------|
| **Exp1** | CNN / LSTM / BiLSTM / CNN-LSTM — 단일 vs 다중 스케일 비교 | 13,104 runs |
| **Exp2** | 7종 전체 모델 — 다중 스케일 입력, SPI3~SPI12 구간 비교 | 5,040 runs |
| **Exp3** | BiLSTM / TCN / TFT — SHAP·IG·TFT VSN+Attention (부안) | — |
| 합계 | | **18,144 runs** |

- **데이터 분할**: 훈련 1991–2015 / 검증 2016–2020 / 테스트 2021–2025
- **선정 기준**: 검증 RMSE 최소 하이퍼파라미터 → 테스트 성능 비교

---

## 주요 결과

### Exp1 — 단일 vs 다중 스케일 비교

| 타깃 | ΔRMSE 범위 | 결론 |
|------|-----------|------|
| SPI1  | −0.12 ~ +0.17 | 두 방식 모두 NSE < 0, 예측 곤란 |
| SPI3  | −0.22 ~ −0.07 | **전 모델·관측소(12/12) 다중 우세, 개선 폭 최대** |
| SPI6  | −0.14 ~ −0.03 | **전 모델·관측소(12/12) 다중 우세** |
| SPI9  | −0.16 ~ +0.05 | 다중 우세 (11/12) |
| SPI12 | −0.36 ~ 0.00 | **전 모델·관측소(12/12) 다중 우세, 개선 폭 최대** |
| SPI18 | −0.11 ~ +0.14 | 혼재 (7/12) |
| SPI24 | −0.05 ~ +0.18 | 혼재 (7/12) |

SPI3~SPI12에서 다중 스케일 입력이 **모든 조합에서 일관되게 우세**. SPI18·SPI24는 자기상관이 강해 추가 스케일의 한계 효용이 감소.

---

### Exp2 — 모델 간 성능 비교 (다중 스케일, SPI3~SPI12)

| Target | CNN | LSTM | BiLSTM | CNN-LSTM | TCN | Transformer | TFT |
|--------|:---:|:----:|:------:|:--------:|:---:|:-----------:|:---:|
| SPI3  | 0.71 | 0.68 | **0.67** | 0.75 | 0.72 | 0.76 | 0.73 |
| SPI6  | 0.60 | 0.61 | 0.59 | 0.62 | **0.57** | 0.58 | 0.59 |
| SPI9  | 0.51 | 0.50 | **0.48** | 0.52 | 0.51 | 0.49 | 0.51 |
| SPI12 | **0.36** | 0.39 | 0.39 | 0.42 | 0.41 | 0.39 | 0.42 |
| **avg RMSE** | 0.55 | 0.54 | **0.53** | 0.58 | 0.55 | 0.56 | 0.56 |
| **avg NSE**  | 0.72 | 0.73 | **0.74** | 0.69 | 0.72 | 0.71 | 0.71 |

**종합 순위**: BiLSTM > LSTM ≈ CNN ≈ TCN > Transformer ≈ TFT > CNN-LSTM

- **BiLSTM** (avg RMSE 0.53, NSE 0.74): 전 구간에서 가장 안정적
- **CNN**: SPI12에서 최우수 (RMSE 0.36)
- **TCN**: SPI6에서 최우수 (RMSE 0.57)
- **Transformer/TFT**: 데이터 규모(~420 샘플)가 작아 이점 없음
- **CNN-LSTM**: 전 구간 최하위 (avg RMSE 0.58)

---

### Exp3 — XAI 분석 (부안, SPI3~SPI12)

#### 타깃−1 스케일 법칙

12개 (모델 × 타깃) 조합 중 **10개에서** 예측 타깃보다 한 단계 짧은 스케일이 가장 중요한 입력 피처로 확인.

| 타깃 | 타깃−1 | SHAP/IG (BiLSTM) | SHAP/IG (TCN) | TFT VSN |
|------|--------|:----------------:|:-------------:|:-------:|
| SPI3  | SPI2 | **SPI2** (.43/.43) | **SPI2** (.29/.29) | **SPI2** (.72) |
| SPI6  | SPI5 | **SPI5** (.32/.31) | **SPI5** (.15/.14) | SPI4 (.37) ⚠️ |
| SPI9  | SPI6 | **SPI6** (.24/.24) | **SPI6** (.14/.15) | **SPI6** (.22) |
| SPI12 | SPI9 | **SPI9** (.40/.39) | SPI12 (.15/.14) ⚠️ | **SPI9** (.49) |

⚠️ = 예외 2건. 모두 타깃 인근 스케일로 법칙의 큰 틀과 정합.

→ **다중 스케일 입력이 단일 스케일보다 우수한 이유를 모델 내부에서 직접 확인**: SPI-*k* 예측에 가장 유용한 선행 신호는 SPI-*k* 자신이 아니라 한 단계 짧은 누적 스케일.

#### SHAP–IG 일관성 및 아키텍처 간 불일치

| 비교 | Spearman ρ | 유의성 |
|------|:----------:|:------:|
| SHAP vs IG (BiLSTM 내, 전 타깃) | ≥ 0.988 (SPI3: 1.000) | p < 0.001 |
| BiLSTM SHAP vs TCN SHAP | 0.358 ~ 0.539 | p > 0.1 |
| BiLSTM SHAP vs TFT VSN | −0.030 ~ 0.442 | p > 0.1 |

BiLSTM 내에서는 SHAP과 IG가 거의 완벽히 일치 → 해석 신뢰도 높음.  
아키텍처 간에는 4개 타깃 모두 약한 일치 → 동일 성능이라도 내부 전략은 아키텍처마다 다름.

#### Lag 중요도 및 TFT Attention 패턴

| 타깃 | seq_len | t−1 비중 (SHAP, BiLSTM) | 패턴 |
|------|:-------:|:------------------------:|------|
| SPI3  | 6 | **71.2%** | 최근 시점에 강하게 집중 |
| SPI6  | 4 | 52.7% | 최근 시점에 집중 |
| SPI9  | 4 | 32.2% | 시점 간 분산 |
| SPI12 | 4 | 37.8% | 시점 간 분산 (소폭 반등) |

TFT Interpretable Attention은 **4개 타깃 모두** 시점 간 균등 분포 → TFT의 시간 의존성 처리는 어텐션 모듈이 아닌 LSTM 인코더 주도.

---

## 프로젝트 구조

```
meteorological-drought-forecasting/
├── src/
│   ├── dataset.py      # 데이터 로딩, MinMaxScaler, 슬라이딩 윈도우
│   ├── metrics.py      # RMSE, MAE, NSE, Willmott d
│   ├── trainer.py      # 학습 루프, EarlyStopping
│   ├── experiment.py   # argparse 실험 러너
│   └── models/
│       ├── lstm.py / bilstm.py / cnn.py / cnn_lstm.py
│       ├── tcn.py          # Temporal Convolutional Network
│       ├── transformer.py  # Transformer Encoder
│       └── tft.py          # Temporal Fusion Transformer (VSN + Interpretable Attn)
│
├── scripts/
│   ├── exp1_single_scale.sh   # Exp1-A: 단일 스케일
│   ├── exp1_multi_scale.sh    # Exp1-B: 다중 스케일
│   ├── exp2_new_models.sh     # Exp2: TCN / Transformer / TFT
│   └── xai_run.py             # XAI CLI 스크립트 (부안, 수치 출력)
│
├── environment_cpu.yml
├── environment_gpu.yml
└── environment_xai.yml

# 저장소에 미포함 (로컬에서만 사용):
#   data/         원본·전처리 SPI 자료 (기상청, 아래 '데이터' 참고)
#   notebooks/    EDA·XAI 분석 노트북
#   results/      18,144개 실험 산출물 및 그림
```

---

## 환경 설정

```bash
# 모델 학습
conda env create -f environment_cpu.yml   # CPU
conda env create -f environment_gpu.yml   # GPU
conda activate pytorch_lstm

# XAI 분석
conda env create -f environment_xai.yml
conda activate drought_xai
```

---

> 재현하려면 먼저 `data/`(기상청 SPI 자료)를 준비해야 합니다. 아래 **데이터** 참고.

### 단일 실험

```bash
conda activate pytorch_lstm
python src/experiment.py \
    --model bilstm --station 부안 --target SPI12 \
    --seq_len 4 --hidden_size 32 --num_layers 2 \
    --dropout 0.0 --lr 1e-3 --batch_size 32
```

### 그리드서치 (이어하기 지원)

```bash
bash scripts/exp1_single_scale.sh   # Exp1-A
bash scripts/exp1_multi_scale.sh    # Exp1-B
bash scripts/exp2_new_models.sh     # Exp2
```

### XAI 분석

```bash
conda activate drought_xai
# 그리드서치로 학습된 best 모델(results/experiments/)에 SHAP·IG·TFT 해석 적용
python scripts/xai_run.py
```

---

## 데이터

| 관측소 | 지점 코드 | 기간 | 월 샘플 수 |
|--------|:--------:|------|:--------:|
| 군산 | 140 | 1991-01 ~ 2025-12 | 420 |
| 전주 | 146 | 1991-01 ~ 2025-12 | 420 |
| 부안 | 243 | 1991-01 ~ 2025-12 | 420 |

- **입력 변수**: SPI1–SPI6, SPI9, SPI12, SPI18, SPI24 (10개 스케일)
- **전처리**: MinMaxScaler(−1, 1), 훈련 데이터 기준 추정
- **출처**: 기상청 기후통계 시스템 (https://data.kma.go.kr)
- ⚠️ 원자료 및 전처리 파일(`data/`)은 저장소에 포함하지 않습니다. 위 출처에서 내려받아 `data/`에 배치하세요.

---

## 주요 인자 (`src/experiment.py`)

| 인자 | 설명 | 선택지 |
|------|------|--------|
| `--model` | 모델 종류 | `lstm` `bilstm` `cnn` `cnn_lstm` `tcn` `transformer` `tft` |
| `--station` | 관측소 | `군산` `전주` `부안` |
| `--target` | 예측 타깃 | `SPI1`~`SPI24` |
| `--features` | 입력 피처 | `all` (다중 스케일) / 쉼표 구분 (단일 스케일) |
| `--seq_len` | look-back 길이 (개월) | `4` `6` `12` |

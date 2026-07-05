#!/usr/bin/env bash
# Experiment 3-B: Multi-scale input (멀티 스케일 입력)
# 입력 = SPI1~SPI24 전체 10개 스케일 (우리 방식)
#
# Models : CNN, LSTM(S-LSTM), BiLSTM(S-BiLSTM), CNN-LSTM
# Targets: SPI1, SPI3, SPI6, SPI9, SPI12, SPI18, SPI24
# Usage  : bash scripts/exp3_multi_scale.sh [--dry-run]

cd "$(dirname "$0")/.."

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

SCRIPT="src/experiment.py"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

SKIP=0; RUN=0

run_or_skip() {
    local rid="$1"; shift
    if [ -d "results/experiments/$rid" ]; then
        (( SKIP++ ))
    else
        (( RUN++ ))
        echo "▶  $rid"
        $DRY_RUN || python "$@" 2>&1 | tee -a "$LOG_DIR/exp3_multi_$(date +%Y%m%d).log"
    fi
}

STATIONS=("군산" "전주" "부안")
TARGETS=("SPI1" "SPI3" "SPI6" "SPI9" "SPI12" "SPI18" "SPI24")
EPOCHS=200; PATIENCE=20; SEED=42

# ── CNN ───────────────────────────────────────────────────────────────────────
for station     in "${STATIONS[@]}"; do
for target      in "${TARGETS[@]}"; do
for seq_len     in 4 6 12; do
for hidden_size in 32 64 128; do
for kernel_size in 3 5; do
for dropout     in 0.0 0.1; do
for lr          in 1e-3 5e-4; do
    rid="cnn_${station}_${target}_sl${seq_len}_hs${hidden_size}_nl2_ks${kernel_size}_dr${dropout}_lr${lr}"
    run_or_skip "$rid" $SCRIPT \
        --model cnn \
        --station "$station" --target "$target" \
        --features all \
        --seq_len $seq_len \
        --hidden_size $hidden_size --num_layers 2 \
        --kernel_size $kernel_size \
        --dropout $dropout --lr $lr --batch_size 32 \
        --epochs $EPOCHS --patience $PATIENCE --seed $SEED
done; done; done; done; done; done; done

# ── LSTM (S-LSTM: num_layers=1,2) ─────────────────────────────────────────────
for station     in "${STATIONS[@]}"; do
for target      in "${TARGETS[@]}"; do
for seq_len     in 4 6 12; do
for hidden_size in 32 64 128; do
for num_layers  in 1 2; do            # 1=LSTM, 2=S-LSTM
for dropout     in 0.0 0.1; do
for lr          in 1e-3 5e-4; do
    rid="lstm_${station}_${target}_sl${seq_len}_hs${hidden_size}_nl${num_layers}_dr${dropout}_lr${lr}_bs32"
    run_or_skip "$rid" $SCRIPT \
        --model lstm \
        --station "$station" --target "$target" \
        --features all \
        --seq_len $seq_len \
        --hidden_size $hidden_size --num_layers $num_layers \
        --dropout $dropout --lr $lr --batch_size 32 \
        --epochs $EPOCHS --patience $PATIENCE --seed $SEED
done; done; done; done; done; done; done

# ── BiLSTM (S-BiLSTM: num_layers=1,2) ────────────────────────────────────────
for station     in "${STATIONS[@]}"; do
for target      in "${TARGETS[@]}"; do
for seq_len     in 4 6 12; do
for hidden_size in 32 64 128; do
for num_layers  in 1 2; do            # 1=BiLSTM, 2=S-BiLSTM
for dropout     in 0.0 0.1; do
for lr          in 1e-3 5e-4; do
    rid="bilstm_${station}_${target}_sl${seq_len}_hs${hidden_size}_nl${num_layers}_dr${dropout}_lr${lr}_bs32"
    run_or_skip "$rid" $SCRIPT \
        --model bilstm \
        --station "$station" --target "$target" \
        --features all \
        --seq_len $seq_len \
        --hidden_size $hidden_size --num_layers $num_layers \
        --dropout $dropout --lr $lr --batch_size 32 \
        --epochs $EPOCHS --patience $PATIENCE --seed $SEED
done; done; done; done; done; done; done

# ── CNN-LSTM ──────────────────────────────────────────────────────────────────
for station      in "${STATIONS[@]}"; do
for target       in "${TARGETS[@]}"; do
for seq_len      in 4 6 12; do
for cnn_channels in "8,8" "32,32"; do
for lstm_hidden  in 32 64; do
for lstm_layers  in 2 3; do
for dropout      in 0.0 0.1; do
for lr           in 1e-3 5e-4; do
    ch="${cnn_channels//,/-}"
    rid="cnn_lstm_${station}_${target}_sl${seq_len}_ch${ch}_lh${lstm_hidden}_ll${lstm_layers}_ks3_dr${dropout}_lr${lr}"
    run_or_skip "$rid" $SCRIPT \
        --model cnn_lstm \
        --station "$station" --target "$target" \
        --features all \
        --seq_len $seq_len \
        --cnn_channels "$cnn_channels" --kernel_size 3 \
        --lstm_hidden $lstm_hidden --lstm_layers $lstm_layers --fc_hidden 8 \
        --dropout $dropout --lr $lr --batch_size 16 \
        --epochs $EPOCHS --patience $PATIENCE --seed $SEED
done; done; done; done; done; done; done; done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [Exp3-B: Multi-scale] skip: $SKIP  |  run: $RUN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

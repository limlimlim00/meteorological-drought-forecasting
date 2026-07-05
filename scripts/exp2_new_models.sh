#!/usr/bin/env bash
# Experiment 2: New models on Multi-scale input
# 입력 = SPI1~SPI24 전체 스케일 (Multi-scale, exp1에서 우위 확인)
#
# Models : TCN, Transformer, TFT
# Targets: SPI1, SPI3, SPI6, SPI9, SPI12, SPI18, SPI24
# Usage  : bash scripts/exp2_new_models.sh [--dry-run]

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
        $DRY_RUN || python "$@" 2>&1 | tee -a "$LOG_DIR/exp2_$(date +%Y%m%d).log"
    fi
}

STATIONS=("군산" "전주" "부안")
TARGETS=("SPI1" "SPI3" "SPI6" "SPI9" "SPI12" "SPI18" "SPI24")
EPOCHS=200; PATIENCE=20; SEED=42

# ── TCN ───────────────────────────────────────────────────────────────────────
for station      in "${STATIONS[@]}"; do
for target       in "${TARGETS[@]}"; do
for seq_len      in 4 6 12; do
for num_channels in "32,32,32" "64,64,64" "32,32,32,32" "64,64,64,64"; do
for kernel_size  in 3 5; do
for dropout      in 0.0 0.1; do
for lr           in 1e-3 5e-4; do
    ch="${num_channels//,/-}"
    rid="tcn_${station}_${target}_sl${seq_len}_ch${ch}_ks${kernel_size}_dr${dropout}_lr${lr}"
    run_or_skip "$rid" $SCRIPT \
        --model tcn \
        --station "$station" --target "$target" \
        --features all \
        --seq_len $seq_len \
        --num_channels "$num_channels" --kernel_size $kernel_size \
        --dropout $dropout --lr $lr --batch_size 32 \
        --epochs $EPOCHS --patience $PATIENCE --seed $SEED
done; done; done; done; done; done; done

# ── Transformer ───────────────────────────────────────────────────────────────
for station    in "${STATIONS[@]}"; do
for target     in "${TARGETS[@]}"; do
for seq_len    in 4 6 12; do
for d_model    in 32 64 128; do
for num_layers in 1 2; do
for dropout    in 0.0 0.1; do
for lr         in 1e-3 5e-4; do
    rid="transformer_${station}_${target}_sl${seq_len}_dm${d_model}_nh4_nl${num_layers}_dr${dropout}_lr${lr}"
    run_or_skip "$rid" $SCRIPT \
        --model transformer \
        --station "$station" --target "$target" \
        --features all \
        --seq_len $seq_len \
        --d_model $d_model --num_heads 4 --num_layers $num_layers \
        --dropout $dropout --lr $lr --batch_size 32 \
        --epochs $EPOCHS --patience $PATIENCE --seed $SEED
done; done; done; done; done; done; done

# ── TFT ───────────────────────────────────────────────────────────────────────
for station    in "${STATIONS[@]}"; do
for target     in "${TARGETS[@]}"; do
for seq_len    in 4 6 12; do
for d_model    in 32 64 128; do
for num_layers in 1 2; do
for dropout    in 0.0 0.1; do
for lr         in 1e-3 5e-4; do
    rid="tft_${station}_${target}_sl${seq_len}_dm${d_model}_nh4_nl${num_layers}_dr${dropout}_lr${lr}"
    run_or_skip "$rid" $SCRIPT \
        --model tft \
        --station "$station" --target "$target" \
        --features all \
        --seq_len $seq_len \
        --d_model $d_model --num_heads 4 --num_layers $num_layers \
        --dropout $dropout --lr $lr --batch_size 32 \
        --epochs $EPOCHS --patience $PATIENCE --seed $SEED
done; done; done; done; done; done; done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [Exp2: TCN/Transformer/TFT] skip: $SKIP  |  run: $RUN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

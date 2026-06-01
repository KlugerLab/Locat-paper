#!/bin/bash
# Launch all 8 seeds in parallel (one GPU each) and open a live monitor.
#
# Usage: bash run_all_seeds.sh [--dry-run]
#   --dry-run  Print commands without executing

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/banach2/wes/.conda/envs/mulde_jax/bin/python
LOG_DIR="$HERE/scores/logs"
mkdir -p "$LOG_DIR"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

SEEDS=(0 1 2 3 4 5 6 7)
GPUS=( 0 1 2 3 4 5 6 7)

echo "============================================"
echo "  3k PBMC multi-seed run  (${#SEEDS[@]} seeds)"
echo "  Logs: $LOG_DIR"
echo "============================================"

PIDS=()
for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${GPUS[$i]}"
    log="$LOG_DIR/seed_${seed}.log"

    cmd="$PYTHON $HERE/run_multiseed.py --seed $seed --gpu $gpu"
    echo "[$(date '+%H:%M:%S')] Launching seed $seed on GPU $gpu  →  $log"

    if [[ $DRY -eq 0 ]]; then
        nohup $cmd > "$log" 2>&1 &
        PIDS+=($!)
    else
        echo "  (dry-run) $cmd"
    fi
done

if [[ $DRY -eq 1 ]]; then
    echo "Dry-run complete."
    exit 0
fi

echo ""
echo "All ${#SEEDS[@]} seeds launched (PIDs: ${PIDS[*]})"
echo ""
echo "Monitor progress with:"
echo "  $PYTHON $HERE/monitor_seeds.py"
echo ""
echo "Or tail a single log:"
echo "  tail -f $LOG_DIR/seed_0.log"
echo ""

# Open the monitor inline (Ctrl-C exits monitor but leaves jobs running)
echo "Starting monitor in 2s... (Ctrl-C exits monitor, jobs keep running)"
sleep 2
"$PYTHON" "$HERE/monitor_seeds.py" --dir "$LOG_DIR" --interval 5

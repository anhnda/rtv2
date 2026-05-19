#!/bin/bash

LOG=noise_stable.log
CMD="python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml -r rtdetrv2_r50vd_6x_coco_ema.pth --test-only"

> "$LOG"

run_noise() {
    local n=$1
    echo "======== noise=$n ========" | tee -a "$LOG"
    $CMD --noise "$n" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

# 0.000 to 0.100 in steps of 0.001
for i in $(seq 0 100); do
    n=$(printf "%.3f" "$(echo "$i * 0.001" | bc)")
    run_noise "$n"
done

# extra points
run_noise 0.5
run_noise 1.0

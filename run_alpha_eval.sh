#!/bin/bash

CONFIG="configs/rtdetrv2/include/rtdetrv2_r50vd.yml"
LOG="alpha_eval.log"
CMD="python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml -r rtdetrv2_r50vd_6x_coco_ema.pth --test-only --noise 0"
DEFAULT_ALPHA=2.5

> "$LOG"

set_alpha() {
    local a=$1
    awk -v NEW="$a" '
        /^[^ \t]/ { in_section = (/^RTDETRTransformerv2:/ ? 1 : 0) }
        in_section && /^  alpha:/ { print "  alpha: " NEW; next }
        { print }
    ' "$CONFIG" > "${CONFIG}.tmp" && mv "${CONFIG}.tmp" "$CONFIG"
}

for alpha in 0 0.25 0.5 1 1.5 2 2.5 3 3.5 4 5; do
    echo "======== alpha=$alpha ========" | tee -a "$LOG"
    set_alpha "$alpha"
    $CMD 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

# Restore default
set_alpha "$DEFAULT_ALPHA"
echo "Alpha restored to $DEFAULT_ALPHA"

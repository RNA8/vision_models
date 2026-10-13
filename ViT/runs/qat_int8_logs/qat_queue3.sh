#!/bin/bash
# runs after qat_queue2.sh (pid 1112251); one GPU process at a time
cd /home/ubuntu/workspace/vision_models/ViT; source env.sh
PY=../.venv/bin/python
while [ -e /proc/1112251 ]; do sleep 20; done
q(){ n=$1; shift; $PY qat_int8.py --mode qat --run-name $n "$@" > runs/qat_int8_logs/$n.log 2>&1; grep -h "PTQ (fake\|final fake\|ORT int8" runs/qat_int8_logs/$n.log | sed "s/^/$n /"; }
q qat_ptq_full_s0 --epochs 0 --threads 6
for s in 1 2; do
  q qat_ptq_full_s$s --epochs 0 --threads 6 --ckpt-run runs/c100_spt_slvit_1000ep_s$s
  q qat_full_strong_e10_s$s --epochs 10 --aug strong --threads 6 --ckpt-run runs/c100_spt_slvit_1000ep_s$s
done

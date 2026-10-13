#!/bin/bash
# waits for the running qat_full_e30 (pid 1110894) to exit, then runs sequentially (one GPU process)
cd /home/ubuntu/workspace/vision_models/ViT; source env.sh
PY=../.venv/bin/python
while [ -e /proc/1110894 ]; do sleep 20; done
q(){ n=$1; shift; $PY qat_int8.py --mode qat --run-name $n "$@" > runs/qat_int8_logs/$n.log 2>&1; grep -h "PTQ (fake\|final fake\|ORT int8" runs/qat_int8_logs/$n.log | sed "s/^/$n /"; }
q qat_full_strong_e10 --epochs 10 --aug strong --threads 6

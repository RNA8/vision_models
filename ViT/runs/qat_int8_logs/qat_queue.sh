#!/bin/bash
# QAT runs, sequential (one GPU process); waits for the PTQ queue first
cd /home/ubuntu/workspace/vision_models/ViT; source env.sh
PY=../.venv/bin/python
while pgrep -f ptq_queue.sh >/dev/null; do sleep 20; done
q(){ n=$1; shift; $PY qat_int8.py --mode qat --run-name $n "$@" > runs/qat_int8_logs/$n.log 2>&1; grep -h "PTQ (fake\|final fake\|ORT int8" runs/qat_int8_logs/$n.log | sed "s/^/$n /"; }
q qat_full_e10 --epochs 10 --threads 6
q qat_full_e30 --epochs 30 --threads 6

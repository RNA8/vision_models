#!/bin/bash
# PTQ (epochs 0) ablations with our placement, fake-quant on GPU, sequential (one GPU process)
cd /home/ubuntu/workspace/vision_models/ViT; source env.sh
PY=../.venv/bin/python
r(){ n=$1; shift; $PY qat_int8.py --mode qat --epochs 0 --no-onnx --run-name $n "$@" > runs/qat_int8_logs/$n.log 2>&1; grep -h "PTQ (fake" runs/qat_int8_logs/$n.log | sed "s/^/$n /"; }
r ptq_full_mse --calib mse
r ptq_full_minmax --calib minmax
r ptq_full_pct --calib percentile
r ptq_mm_w --groups mm w
r ptq_mm_w_input --groups mm w input
r ptq_no_residual --groups input mm w linout softmax
r ptq_no_softmax --groups input mm w linout residual
r ptq_no_linout --groups input mm w softmax residual
r ptq_full_a16res --a16 residual
r ptq_full_a16res_linout --a16 residual linout
r ptq_w_only --groups w
r ptq_acts_only --groups input mm linout softmax residual

#!/usr/bin/env bash
python3  /nfs/project/libo_i/IOU.pytorch/tools/train_net_step.py \
        --dataset coco2017 \
        --cfg /nfs/project/libo_i/IOU.pytorch/configs/baselines/e2e_faster_rcnn_R-50-C4_2x.yaml \
        --use_tfboard --bs 4 --nw 4 --iter_size 4
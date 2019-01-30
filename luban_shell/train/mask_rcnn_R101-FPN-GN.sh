#!/usr/bin/env bash
python3  /nfs/project/libo_i/IOU.pytorch/tools/train_net_step.py \
        --dataset coco2017 \
        --cfg /nfs/project/libo_i/IOU.pytorch/configs/gn_baselines/e2e_mask_rcnn_R-101-FPN_2x_gn.yaml \
        --use_tfboard --bs 16 --nw 4 --iter_size 4
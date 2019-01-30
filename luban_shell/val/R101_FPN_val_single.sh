#!/usr/bin/env bash
python3 /nfs/project/libo_i/IOU.pytorch/tools/test_net.py \
    --dataset coco2017 \
    --cfg /nfs/project/libo_i/IOU.pytorch/configs/baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml \
    --load_ckpt /nfs/project/good_model/coco_101r/model_step_89999.pth
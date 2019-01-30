#!/usr/bin/env bash
python3 /nfs/project/libo_i/IOU.pytorch/tools/test_net.py \
    --dataset coco2017 \
    --cfg /nfs/project/libo_i/IOU.pytorch/configs/baselines/e2e_faster_rcnn_R-50-C4_2x.yaml \
    --load_ckpt /nfs/project/libo_i/IOU.pytorch/Outputs/e2e_faster_rcnn_R-50-C4_2x/Jan24-19-03-09_ml-gpu-ser130.nmg01_step/ckpt/model_step19999.pth \
    --output_dir /nfs/project/libo_i/IOU.pytorch/Outputs/R50_faster_rcnn
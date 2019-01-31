import subprocess
import os
import cv2
import matplotlib
import sys

matplotlib.use('Agg')
sys.path.insert(0, '/nfs/project/libo_i/IOU.pytorch/info_output')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def subproc(cmd):
	print(cmd)
	try:
		out_bytes = subprocess.call(cmd, shell = True)
	except subprocess.CalledProcessError as e:
		out_bytes = e.output  # Output generated before error
		code = e.returncode  # Return code


root_path = "/nfs/project/libo_i/IOU.pytorch/luban_shell/train/Outputs"

test_list = []

for root, dirs, files in os.walk(root_path):
	for name in files:
		if ".pth" in name:
			test_list.append(os.path.join(root, name))

info_save_path = "/nfs/project/libo_i/IOU.pytorch/info_output"

for item in test_list:
	yaml = item.split('/')[-4]
	model_info = item.split('/')[-3]
	steps = item.split('/')[-1]
	dir_save_path = os.path.join(info_save_path, yaml)
	if not os.path.exists(dir_save_path):
		os.makedirs(dir_save_path)
	
	# 修改.pth后缀名为.txt
	txt_save_path = dir_save_path + "/" + model_info + "_" + steps[:-4] + '.txt'
	print(txt_save_path)
	
	infer_cmd = "python3 /nfs/project/libo_i/IOU.pytorch/tools/test_net.py \
	            --dataset coco2017 \
	            --cfg /nfs/project/libo_i/IOU.pytorch/configs/baselines/{}.yaml \
				--multi-gpu-testing \
	            --load_ckpt {} > {}".format(yaml, item, txt_save_path)
	
	if not "gn" in yaml:
		subproc(infer_cmd)

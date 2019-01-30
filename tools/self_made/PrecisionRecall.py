import sys
import json
import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

sys.path.insert(0, '/nfs/project/libo_i/IOU.pytorch/lib')
json_path = '/nfs/project/libo_i/IOU.pytorch/pred_iou.json'
json_ = json.loads(open(json_path).read())

def precision_recall(gt, pred):
	for i in range(10):
		gt_thresh = i * 0.1
		ind = np.where(pred >= gt_thresh)
		gt_select = gt[ind]
		right_ind = np.where(gt_select >= 0.5)[0]
		precision = (len(right_ind)) * 1.0 / (len(gt_select))
		print("{}: P:{:.2f} NUM:{}".format(gt_thresh, precision, len(right_ind)))

# 1. rois_score
# 2. pred_iou
x_name = 'gt_iou'
y_name = 'pred_iou'
gt_thresh = 0.0

x_info = np.array(json_[x_name])
y_info = np.array(json_[y_name])

for score in range(5, 10):
	score = score * 0.1
	ind = np.where(y_info > 0.5)[0]
	y_pred = x_info[ind]
	y_right = np.where(y_pred > score)[0]
	print("score {}: right {} pred {} precison {:.2f}".format(score, len(y_right), len(y_pred),
	                                                          len(y_right) * 1. / len(y_pred)))

# draw_point(x_info, y_info, gt_thresh)
# draw_line(x_info, y_info)

precision_recall(x_info, y_info)

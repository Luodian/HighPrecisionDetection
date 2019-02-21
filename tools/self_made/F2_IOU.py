from scipy import stats
import numpy as np
import os
import json
import sys

sys.path.insert(0, '/nfs/project/libo_i/IOU.pytorch/lib')
import utils.boxes as box_utils
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def predbox_roi_iou(raw_roi, pred_box):
	iou = box_utils.bbox_overlaps(raw_roi, pred_box)
	roi_iou = iou.max(axis = 1)
	return roi_iou


def precision_recall(gt, pred, thrs, flag):
	print("PR of gt_iou and {}".format(flag))
	ind = np.where(np.array(pred, dtype = np.float32) > thrs)[0]
	y_pred = gt[ind]
	for score in range(0, 10):
		score = score * 0.1
		y_right = np.where(y_pred > score)[0]
		print("Thresh {:.1f}: right {} pred {} precison {:.2f}".format(score, len(y_right), len(y_pred),
		                                                               len(y_right) * 1. / len(y_pred)))
	print("\n")


def stats_on_iou_vertical(iou_vertical, method):
	print("PR of shift_iou and gt_iou filtered by {}".format(method))
	for score in range(0, 10):
		score = score * 0.1
		y_right = np.where(np.array(iou_vertical, dtype = np.float32) > score)[0]
		print("Thresh {:.1f}: right {} pred {} recall {:.2f}".format(score, len(y_right), len(iou_vertical),
		                                                             len(y_right) * 1. / len(iou_vertical)))
	
	print("\n")


def BIG(method):
	with open("/nfs/project/libo_i/IOU.pytorch/IOU_Validation/{}.json".format(method)) as f:
		dict_all = json.load(f)
	
	total_shift_iou = []
	total_final_iou = []
	total_score = []
	
	for item in dict_all:
		shift_i = dict_all[item]['shift_iou']
		total_shift_iou.extend(shift_i)
		
		final_i = dict_all[item]['final_iou']
		total_final_iou.extend(final_i)
		
		score_i = dict_all[item]['rois_score']
		
		total_score.extend(score_i)
	
	precision_recall(np.array(total_final_iou),
	                 np.array(total_shift_iou), 0.5, "IOU")
	precision_recall(np.array(total_final_iou),
	                 np.array(total_score), 0.5, "SCORE")


method = "FPN_SCORE_NMS"
with open("/nfs/project/libo_i/IOU.pytorch/IOU_Validation/{}.json".format(method)) as f:
	dict_all = json.load(f)

total_shift_iou = []
total_final_iou = []
total_score = []

for item in dict_all:
	shift_i = dict_all[item]['shift_iou']
	total_shift_iou.extend(shift_i)
	
	final_i = dict_all[item]['final_iou']
	total_final_iou.extend(final_i)
	
	score_i = dict_all[item]['stage1_score']
	
	total_score.extend(score_i)
	
	assert len(shift_i) == len(final_i) == len(score_i)
	print("Length = {}".format(len(shift_i)))

precision_recall(np.array(total_final_iou), total_shift_iou, 0.3, "IOU")

precision_recall(np.array(total_final_iou), total_score, 0.8, "SCORE")

gt_iou_above_ths = []
shift_iou_above_ths = []
score_above_ths = []

for ind, item in enumerate(total_final_iou):
	if item > 0.5 and total_shift_iou[ind] > 0.5:
		gt_iou_above_ths.append(item)
		shift_iou_above_ths.append(total_shift_iou[ind])
		score_above_ths.append(total_score[ind])

# 对数据进行sort之后再计算
sorted_index = np.argsort(gt_iou_above_ths)
sorted_gt_iou = np.array(gt_iou_above_ths, dtype = np.float32)[sorted_index]
shift_iou_above_ths = np.array(
	shift_iou_above_ths, dtype = np.float32)[sorted_index]
score_above_ths = np.array(score_above_ths, dtype = np.float32)[sorted_index]

# gt_iou_ab_ths和shift_iou的数据进行逐0.1的算均值
shift_mean_value = []
x_line = np.linspace(0.5, 1, 6)[0:-1]
for item in x_line:
	indx_left = np.searchsorted(sorted_gt_iou, item)
	indx_right = np.searchsorted(sorted_gt_iou, item + 0.1)
	if indx_right >= len(shift_iou_above_ths):
		indx_right = len(shift_iou_above_ths) - 1
	
	shift_mean_value.append(
		np.mean(shift_iou_above_ths[indx_left: indx_right + 1]))

x_line += 0.05

# 对gt_iou和shift_iou的数据进行逐0.1的算均值

sorted_index_all = np.argsort(total_final_iou)
sorted_total_iou = np.array(total_final_iou, dtype = np.float32)[
	sorted_index_all]
sorted_total_score = np.array(total_score, dtype = np.float32)[sorted_index_all]
score_mean_value = []
score_xline = np.linspace(0, 1, 11)[0:-1]

for item in score_xline:
	indx_left = np.searchsorted(sorted_total_iou, item)
	indx_right = np.searchsorted(sorted_total_iou, item + 0.1)
	if indx_right >= len(sorted_total_score):
		indx_right = len(sorted_total_score) - 1
	
	score_mean_value.append(
		np.mean(sorted_total_score[indx_left:indx_right + 1]))

score_xline += 0.05

plt.subplot(211)
plt.scatter(sorted_total_iou, sorted_total_score, c = "b", alpha = 0.5, s = 0.05)
plt.plot(score_xline, score_mean_value, c = 'r', marker = 'o')
plt.ylabel("score")
plt.xlabel("iou_with_gt")
# plt.autoscale(tight = True)
plt.grid()

plt.subplot(212)
plt.scatter(sorted_gt_iou, shift_iou_above_ths, c = "b", alpha = 0.5, s = 0.1)
plt.plot(x_line, shift_mean_value, c = 'r', marker = 'o')
plt.ylabel("iou_with_shift")
plt.xlabel("iou_with_gt")
plt.suptitle('{}'.format(method))
# plt.suptitle("Score NMS")
# plt.autoscale(tight = True)
plt.grid()
# plt.show()

plt.savefig("/nfs/project/libo_i/IOU.pytorch/{}.png".format(method), dpi = 200)

p = stats.pearsonr(sorted_total_iou, sorted_total_score)
s = stats.spearmanr(sorted_total_iou, sorted_total_score)

print("Results with gt_iou and score")
print("Pearson: {}".format(p))
print(s)

p = stats.pearsonr(sorted_gt_iou, shift_iou_above_ths)
s = stats.spearmanr(sorted_gt_iou, shift_iou_above_ths)

print("\n\nResults with gt_iou and shift_iou")
print("Pearson: {}".format(p))
print(s)

print("KKKKKKK!!!!!!")

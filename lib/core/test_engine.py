# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml
import json
from six.moves import cPickle as pickle

import torch

from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
import utils.blob as blob_utils
from utils.io import save_object
from utils.timer import Timer
from core.test import predbox_roi_iou

# Use a non-interactive backend
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

logger = logging.getLogger(__name__)


# def predbox_roi_iou(raw_roi , pred_box):
# 	if raw_roi.size==0:
# 		raw_roi=np.zeros((1 , 4) , dtype="float32")
# 	if pred_box.size==0:
# 		pred_box=np.zeros((1 , 4) , dtype="float32")
#
# 	iou=box_utils.bbox_overlaps(raw_roi , pred_box)
# 	roi_iou=iou.max(axis=1)
# 	return roi_iou


def get_eval_functions():
	# Determine which parent or child function should handle inference
	if cfg.MODEL.RPN_ONLY:
		raise NotImplementedError
	# child_func = generate_rpn_on_range
	# parent_func = generate_rpn_on_dataset
	else:
		# Generic case that handles all network types other than RPN-only nets
		# and RetinaNet
		child_func = test_net
		parent_func = test_net_on_dataset
	
	return parent_func, child_func


def get_inference_dataset(index, is_parent = True):
	assert is_parent or len(cfg.TEST.DATASETS) == 1, \
		'The child inference process can only work on a single dataset'
	
	dataset_name = cfg.TEST.DATASETS[index]
	
	if cfg.TEST.PRECOMPUTED_PROPOSALS:
		assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
			'The child inference process can only work on a single proposal file'
		assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
			'If proposals are used, one proposal file must be specified for ' \
			'each dataset'
		proposal_file = cfg.TEST.PROPOSAL_FILES[index]
	else:
		proposal_file = None
	
	return dataset_name, proposal_file


def run_inference(
		args, ind_range = None,
		multi_gpu_testing = False, gpu_id = 0,
		check_expected_results = False):
	parent_func, child_func = get_eval_functions()
	is_parent = ind_range is None
	
	def result_getter():
		if is_parent:
			# Parent case:
			# In this case we're either running inference on the entire dataset in a
			# single process or (if multi_gpu_testing is True) using this process to
			# launch subprocesses that each run inference on a range of the dataset
			all_results = {}
			for i in range(len(cfg.TEST.DATASETS)):
				dataset_name, proposal_file = get_inference_dataset(i)
				output_dir = args.output_dir
				results = parent_func(
					args,
					dataset_name,
					proposal_file,
					output_dir,
					multi_gpu = multi_gpu_testing
				)
				all_results.update(results)
			
			return all_results
		else:
			# Subprocess child case:
			# In this case test_net was called via subprocess.Popen to execute on a
			# range of inputs on a single dataset
			dataset_name, proposal_file = get_inference_dataset(0, is_parent = False)
			output_dir = args.output_dir
			return child_func(
				args,
				dataset_name,
				proposal_file,
				output_dir,
				ind_range = ind_range,
				gpu_id = gpu_id
			)
	
	all_results = result_getter()
	if check_expected_results and is_parent:
		task_evaluation.check_expected_results(
			all_results,
			atol = cfg.EXPECTED_RESULTS_ATOL,
			rtol = cfg.EXPECTED_RESULTS_RTOL
		)
		task_evaluation.log_copy_paste_friendly_results(all_results)
	
	return all_results


def test_net_on_dataset(
		args,
		dataset_name,
		proposal_file,
		output_dir,
		multi_gpu = False,
		gpu_id = 0):
	"""Run inference on a dataset."""
	dataset = JsonDataset(dataset_name)
	test_timer = Timer()
	test_timer.tic()
	if multi_gpu:
		num_images = len(dataset.get_roidb())
		all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
			args, dataset_name, proposal_file, num_images, output_dir
		)
	else:
		all_boxes, all_segms, all_keyps = test_net(
			args, dataset_name, proposal_file, output_dir, gpu_id = gpu_id
		)
	
	test_timer.toc()
	logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
	results = task_evaluation.evaluate_all(
		dataset, all_boxes, all_segms, all_keyps, output_dir
	)
	return results


def multi_gpu_test_net_on_dataset(
		args, dataset_name, proposal_file, num_images, output_dir):
	"""Multi-gpu inference on a dataset."""
	binary_dir = envu.get_runtime_dir()
	binary_ext = envu.get_py_bin_ext()
	binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
	assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)
	
	# Pass the target dataset and proposal file (if any) via the command line
	opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
	if proposal_file:
		opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]
	
	# Run inference in parallel in subprocesses
	# Outputs will be a list of outputs from each subprocess, where the output
	# of each subprocess is the dictionary saved by test_net().
	outputs = subprocess_utils.process_in_parallel(
		'detection', num_images, binary, output_dir,
		args.load_ckpt, args.load_detectron, opts
	)
	
	# Collate the results from each subprocess
	all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
	all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
	all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
	for det_data in outputs:
		all_boxes_batch = det_data['all_boxes']
		all_segms_batch = det_data['all_segms']
		all_keyps_batch = det_data['all_keyps']
		for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
			all_boxes[cls_idx] += all_boxes_batch[cls_idx]
			all_segms[cls_idx] += all_segms_batch[cls_idx]
			all_keyps[cls_idx] += all_keyps_batch[cls_idx]
	det_file = os.path.join(output_dir, 'detections.pkl')
	cfg_yaml = yaml.dump(cfg)
	save_object(
		dict(
			all_boxes = all_boxes,
			all_segms = all_segms,
			all_keyps = all_keyps,
			cfg = cfg_yaml
		), det_file
	)
	logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
	
	return all_boxes, all_segms, all_keyps


def test_net(
		args,
		dataset_name,
		proposal_file,
		output_dir,
		ind_range = None,
		gpu_id = 0):
	"""Run inference on all images in a dataset or over an index range of images
	in a dataset using a single GPU.
	"""
	assert not cfg.MODEL.RPN_ONLY, \
		'Use rpn_generate to generate proposals from RPN-only models'
	
	roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
		dataset_name, proposal_file, ind_range
	)
	# 在这里获得gt的信息
	model = initialize_model_from_cfg(args, gpu_id = gpu_id)
	num_images = len(roidb)
	num_classes = cfg.MODEL.NUM_CLASSES
	all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
	timers = defaultdict(Timer)
	
	dict_all = {}
	
	if cfg.TEST.IOU_OUT or cfg.FAST_RCNN.FAST_HEAD2_DEBUG:
		with open("/nfs/project/libo_i/IOU.pytorch/data/cache/coco_2017_val_gt_roidb.pkl", 'rb') as fp:
			cached_roidb = pickle.load(fp)
		assert len(roidb) == len(cached_roidb)
	
	for i, entry in enumerate(roidb):
		if cfg.TEST.PRECOMPUTED_PROPOSALS:
			# The roidb may contain ground-truth rois (for example, if the roidb
			# comes from the training or val split). We only want to evaluate
			# detection on the *non*-ground-truth rois. We select only the rois
			# that have the gt_classes field set to 0, which means there's no
			# ground truth.
			box_proposals = entry['boxes'][entry['gt_classes'] == 0]
			if len(box_proposals) == 0:
				continue
		else:
			# Faster R-CNN type models generate proposals on-the-fly with an
			# in-network RPN; 1-stage models don't require proposals.
			box_proposals = None
		
		im = cv2.imread(entry['image'])
		im_name = entry['image'].split('/')[-1][:-4]
		
		cls_boxes_i, cls_segms_i, cls_keyps_i, dict_all[im_name] = im_detect_all(model, im, box_proposals, timers,
		                                                                         im_name_tag = im_name)
		if cfg.FAST_RCNN.FAST_HEAD2_DEBUG:
			gt_i = cached_roidb[i]['boxes']
			shift_gt_iou = predbox_roi_iou(np.array(dict_all[im_name]['stage1_out'], dtype = np.float32),
			                               np.array(gt_i, dtype = "float32"))
			
			dict_all[im_name]['final_iou'] = shift_gt_iou.tolist()
			dict_all[im_name]['shift_iou'] = dict_all[im_name]['shift_iou'].tolist()
			
			if cfg.FAST_RCNN.FAST_HEAD2_DEBUG_VIS and i < 100:
				
				if cfg.FAST_RCNN.IOU_NMS:
					with open("/nfs/project/libo_i/IOU.pytorch/IOU_Validation/cls_tracker.json", 'r') as f:
						cls_tracker = json.load(f)
				elif cfg.FAST_RCNN.SCORE_NMS:
					with open("/nfs/project/libo_i/IOU.pytorch/IOU_Validation/cls_tracker.json", 'r') as f:
						cls_tracker = json.load(f)
				
				# Draw stage1 pred_boxes onto im and gt
				dpi = 200
				fig = plt.figure(frameon = False)
				fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
				ax = plt.Axes(fig, [0., 0., 1., 1.])
				ax.axis('off')
				fig.add_axes(ax)
				ax.imshow(im[:, :, ::-1])
				# 在im上添加gt
				for item in gt_i:
					ax.add_patch(
						plt.Rectangle((item[0], item[1]),
						              item[2] - item[0],
						              item[3] - item[1],
						              fill = False, edgecolor = 'r',
						              linewidth = 0.3, alpha = 1))
				
				# 在im上添加proposals
				length = len(dict_all[im_name]['boxes'])
				for ind in range(length):
					# stage1_item = dict_all[im_name]['stage1_pred_boxes'][ind]
					stage1_item = dict_all[im_name]['boxes'][ind]
					score_item = dict_all[im_name]['score'][ind]
					score_item = round(score_item, 2)
					ax.add_patch(
						plt.Rectangle((stage1_item[0], stage1_item[1]),
						              stage1_item[2] - stage1_item[0],
						              stage1_item[3] - stage1_item[1],
						              fill = False, edgecolor = 'g',
						              linewidth = 0.5, alpha = 1))
					ax.text(
						stage1_item[0], stage1_item[1] - 2,
						str(score_item),
						fontsize = 4,
						family = 'serif',
						bbox = dict(facecolor = 'g', alpha = 1, pad = 0, edgecolor = 'none'), color = 'white')
				
				length = len(dict_all[im_name]['stage1_out'])
				for ind in range(length):
					# stage1_item = dict_all[im_name]['stage1_pred_boxes'][ind]
					stage1_item = dict_all[im_name]['stage1_out'][ind]
					ax.add_patch(
						plt.Rectangle((stage1_item[0], stage1_item[1]),
						              stage1_item[2] - stage1_item[0],
						              stage1_item[3] - stage1_item[1],
						              fill = False, edgecolor = 'orange',
						              linewidth = 0.1, alpha = 0.6))
				
				fig.savefig("/nfs/project/libo_i/IOU.pytorch/2stage_iminfo/{}.png".format(im_name), dpi = dpi)
				plt.close('all')
			
			dict_all[im_name].pop('stage1_out')
			dict_all[im_name].pop('stage2_out')
			dict_all[im_name].pop('stage2_score')
			dict_all[im_name].pop('score')
			dict_all[im_name].pop('boxes')
		
		if cfg.TEST.IOU_OUT:
			gt_i = cached_roidb[i]['boxes']
			
			# NMS
			keep = np.array(dict_all[im_name]['keep'])
			
			dict_all[im_name]['shift_iou'] = np.array(dict_all[im_name]['shift_iou'], dtype = np.float32)[
				keep].tolist()
			dict_all[im_name]['rois_score'] = np.array(dict_all[im_name]['rois_score'], dtype = np.float32)[
				keep].tolist()
			dict_all[im_name]['rois'] = np.array(dict_all[im_name]['rois'], dtype = np.float32)[keep].tolist()
			pred_boxes_scores = dict_all[im_name]['pred_boxes_scores']
			
			# Thresh filter
			iou_thrsh_keep = np.where(np.array(dict_all[im_name]['shift_iou'], dtype = np.float32) >= 0.1)[0]
			score_thrsh_keep = np.where(np.array(dict_all[im_name]['rois_score'], dtype = np.float32) >= 0.8)[0]
			
			iou_rois = np.array(dict_all[im_name]['rois'], dtype = np.float32)[iou_thrsh_keep]
			score_rois = np.array(dict_all[im_name]['rois'], dtype = np.float32)[score_thrsh_keep]
			
			roi_to_final = predbox_roi_iou(np.array(dict_all[im_name]['rois'], dtype = np.float32),
			                               np.array(gt_i, dtype = "float32"))
			iou_final_to_rois = predbox_roi_iou(np.array(gt_i, dtype = "float32"), iou_rois)
			score_final_to_rois = predbox_roi_iou(np.array(gt_i, dtype = "float32"), score_rois)
			
			dict_all[im_name]['final_iou'] = roi_to_final.tolist()
			dict_all[im_name]['iou_final_vertical'] = iou_final_to_rois.tolist()
			dict_all[im_name]['score_final_vertical'] = score_final_to_rois.tolist()
			
			if cfg.TEST.IOU_OUT_VIS:
				# 试着画出图像看一看
				dpi = 200
				fig = plt.figure(frameon = False)
				fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
				ax = plt.Axes(fig, [0., 0., 1., 1.])
				ax.axis('off')
				fig.add_axes(ax)
				ax.imshow(im[:, :, ::-1])
				# 在im上添加gt
				for item in gt_i:
					ax.add_patch(
						plt.Rectangle((item[0], item[1]),
						              item[2] - item[0],
						              item[3] - item[1],
						              fill = False, edgecolor = 'g',
						              linewidth = 0.6, alpha = 1))
				
				# 在im上添加proposals
				cnt = 0
				for ind, item in enumerate(dict_all[im_name]['rois']):
					iou_value = dict_all[im_name]['shift_iou'][ind]
					if iou_value > 0.8:
						cnt += 1
						ax.add_patch(
							plt.Rectangle((item[0], item[1]),
							              item[2] - item[0],
							              item[3] - item[1],
							              fill = False, edgecolor = 'orange',
							              linewidth = 0.5, alpha = 1))
						ax.text(
							item[0], item[1] - 2,
							str(round(iou_value, 2)),
							fontsize = 4,
							family = 'serif',
							bbox = dict(
								facecolor = 'g', alpha = 1, pad = 0, edgecolor = 'none'),
							color = 'white')
				
				for ind, item in enumerate(dict_all[im_name]['pred_boxes']):
					score_value = dict_all[im_name]['pred_boxes_scores'][ind]
					if score_value > 0.5:
						cnt += 1
						ax.add_patch(
							plt.Rectangle((item[0], item[1]),
							              item[2] - item[0],
							              item[3] - item[1],
							              fill = False, edgecolor = 'red',
							              linewidth = 0.5, alpha = 1))
						ax.text(
							item[0], item[1] - 2,
							str(round(score_value, 2)),
							fontsize = 4,
							family = 'serif',
							bbox = dict(
								facecolor = 'red', alpha = 1, pad = 0, edgecolor = 'none'),
							color = 'white')
				
				print("Here is {} proposals above 0.5 in im {}".format(cnt, im_name))
				fig.savefig("/nfs/project/libo_i/IOU.pytorch/im_out/{}.png".format(im_name), dpi = dpi)
				plt.close('all')
			
			dict_all[im_name].pop('rois')
			dict_all[im_name].pop('pred_boxes')
			dict_all[im_name].pop('keep')
		
		if i == 100:
			method = "IOU_Exp"
			if cfg.FAST_RCNN.IOU_NMS:
				method = "FPN_IOU_NMS"
			elif cfg.FAST_RCNN.SCORE_NMS:
				method = "FPN_SCORE_NMS"
			with open("/nfs/project/libo_i/IOU.pytorch/IOU_Validation/{}.json".format(method), 'w') as f:
				f.write(json.dumps(dict_all))
				print("In {} round, saved dict_all ".format(i))
		extend_results(i, all_boxes, cls_boxes_i)
		if cls_segms_i is not None:
			extend_results(i, all_segms, cls_segms_i)
		if cls_keyps_i is not None:
			extend_results(i, all_keyps, cls_keyps_i)
		
		if i % 10 == 0:  # Reduce log file size
			ave_total_time = np.sum([t.average_time for t in timers.values()])
			eta_seconds = ave_total_time * (num_images - i - 1)
			eta = str(datetime.timedelta(seconds = int(eta_seconds)))
			det_time = (
					timers['im_detect_bbox'].average_time +
					timers['im_detect_mask'].average_time +
					timers['im_detect_keypoints'].average_time
			)
			misc_time = (
					timers['misc_bbox'].average_time +
					timers['misc_mask'].average_time +
					timers['misc_keypoints'].average_time
			)
			logger.info(
				(
					'im_detect: range [{:d}, {:d}] of {:d}: '
					'{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
				).format(
					start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
					start_ind + num_images, det_time, misc_time, eta
				)
			)
		
		if cfg.VIS:
			im_name = os.path.splitext(os.path.basename(entry['image']))[0]
			vis_utils.vis_one_image(
				im[:, :, ::-1],
				'{:d}_{:s}'.format(i, im_name),
				os.path.join(output_dir, 'vis'),
				cls_boxes_i,
				segms = cls_segms_i,
				keypoints = cls_keyps_i,
				thresh = cfg.VIS_TH,
				box_alpha = 0.8,
				dataset = dataset,
				show_class = True
			)
	
	cfg_yaml = yaml.dump(cfg)
	if ind_range is not None:
		det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
	else:
		det_name = 'detections.pkl'
	det_file = os.path.join(output_dir, det_name)
	save_object(
		dict(
			all_boxes = all_boxes,
			all_segms = all_segms,
			all_keyps = all_keyps,
			cfg = cfg_yaml
		), det_file
	)
	logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
	return all_boxes, all_segms, all_keyps


def initialize_model_from_cfg(args, gpu_id = 0):
	"""Initialize a model from the global cfg. Loads test-time weights and
	set to evaluation mode.
	"""
	model = model_builder.Generalized_RCNN()
	model.eval()
	
	if args.cuda:
		model.cuda()
	
	if args.load_ckpt:
		load_name = args.load_ckpt
		logger.info("loading checkpoint %s", load_name)
		checkpoint = torch.load(load_name, map_location = lambda storage, loc:storage)
		net_utils.load_ckpt(model, checkpoint['model'])
	
	if args.load_detectron:
		logger.info("loading detectron weights %s", args.load_detectron)
		load_detectron_weight(model, args.load_detectron)
	
	model = mynn.DataParallel(model, cpu_keywords = ['im_info', 'roidb'], minibatch = True)
	
	return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
	"""Get the roidb for the dataset specified in the global cfg. Optionally
	restrict it to a range of indices if ind_range is a pair of integers.
	"""
	dataset = JsonDataset(dataset_name)
	if cfg.TEST.PRECOMPUTED_PROPOSALS:
		assert proposal_file, 'No proposal file given'
		roidb = dataset.get_roidb(
			proposal_file = proposal_file,
			proposal_limit = cfg.TEST.PROPOSAL_LIMIT
		)
	else:
		roidb = dataset.get_roidb()
	
	if ind_range is not None:
		total_num_images = len(roidb)
		start, end = ind_range
		roidb = roidb[start:end]
	else:
		start = 0
		end = len(roidb)
		total_num_images = end
	
	return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
	"""Return empty results lists for boxes, masks, and keypoints.
	Box detections are collected into:
	  all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
	Instance mask predictions are collected into:
	  all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
	  1:1 correspondence with the boxes in all_boxes[cls][image]
	Keypoint predictions are collected into:
	  all_keyps[cls][image] = [...] list of keypoints results, each encoded as
	  a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
	  [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
	  Keypoints are recorded for person (cls = 1); they are in 1:1
	  correspondence with the boxes in all_boxes[cls][image].
	"""
	# Note: do not be tempted to use [[] * N], which gives N references to the
	# *same* empty list.
	all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
	all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
	all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
	return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
	"""Add results for an image to the set of all results at the specified
	index.
	"""
	# Skip cls_idx 0 (__background__)
	for cls_idx in range(1, len(im_res)):
		all_res[cls_idx][index] = im_res[cls_idx]

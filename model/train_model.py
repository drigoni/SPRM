import time
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import copy

from utils.evaluator import Evaluator
from utils.utils import union_target

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = FutureWarning)


def train(model, loss_function, train_loader, test_loader, args, lr = 1e-4, epochs = 25, device_str='cuda', save_checkpoint=None):
	# init wandb
	wandb.watch(model)

	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() and device_str == 'cuda' else "cpu")
	use_gpu = torch.cuda.is_available()
	if use_gpu:
		print("CUDA available, numof GPUs: ", torch.cuda.device_count())
		model.to(device)
		loss_function.to(device)

	model = model.float()
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)

	print("--- EVALUATION BEFORE TRAINING")
	score, point_game_score, loss_val = evaluate(test_loader, model, loss_function, device_str)
	print("Evaluation before training -> score: {}   point_game_score: {}  loss_val: {} .".format(score, point_game_score, loss_val))

	# params for best model
	best_model = None
	best_score = 0
	for epoch in range(epochs):
		print("--- EPOCH", epoch)
		t = time.time()
		total_loss = 0
		n_batches = 0

		model.train(True)
		for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features in tqdm(train_loader):
			# print('===============================================================================')
			# print("bboxes: ", bboxes.shape, bboxes)
			# print("target_bboxes: ", target_bboxes.shape, target_bboxes)
			# print('target_union', [union_target(target_bboxes[0][:num_query[0]])])
			# print("num_obj: ", num_obj.shape, num_obj)
			# print("num_query: ", num_query.shape, num_query)
			
			if (use_gpu):
				idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, \
					head, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features = \
				idx.to(device), labels.to(device), attrs.to(device), feature.to(device), query.to(device), bboxes.to(device), \
					target_bboxes.to(device), num_obj.to(device), num_query.to(device), head.to(device), bert_query_input_ids.to(device), \
					bert_query_attention_mask.to(device), locations.to(device), relations.to(device), spatial_features.to(device)
			
			# training steps
			optimizer.zero_grad()
			prediction_scores, prediction_loss, target_pred, query_similarity = model.forward(
				query, head, labels, feature, attrs, bboxes, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features
			)
			loss = loss_function(prediction_loss, target_pred, query_similarity)
			loss.backward()
			optimizer.step()

			# update variables
			n_batches += 1
			total_loss += loss.item()

		t1 = time.time()
		print("Training -> time: {} | total_loss: {} .".format(t1 - t, total_loss / n_batches))

		# evaluate
		t2 = time.time()
		score, point_game_score, loss_val = evaluate(test_loader, model, loss_function, device_str)
		print("Evaluation -> time: {} | score: {}   point_game_score: {}  loss_val: {} .".format(time.time() - t2, score, point_game_score, loss_val))

		if score > best_score:
			best_model = copy.deepcopy(model)
			best_score = score

		if save_checkpoint:
			torch.save(copy.deepcopy(model).cpu().state_dict(), save_checkpoint.format(epoch))

		wandb.log({	"loss": total_loss / n_batches,
					"acc_val": score,
					"point_acc_val": point_game_score,
					"loss_val": loss_val,
					})
	return best_model


def evaluate(test_loader, model, model2, loss_function, device_str='cuda'):
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() and device_str == 'cuda' else "cpu")
	use_gpu = torch.cuda.is_available()
	model = model.float()
	model2 = model2.float()
	pred_bboxes_list = []
	target_bboxes_list = []
	num_query_list = []
	n_batches = 0
	total_loss = 0
	from utils.utils import load_vocabulary
	wordEmbedding = load_vocabulary(f"data/glove/glove.6B.50d.txt")

	model.eval()
	model2.eval()
	for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features in tqdm(test_loader):
		n_batches += 1

		if (use_gpu):
			idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, \
				head, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features = \
			idx.to(device), labels.to(device), attrs.to(device), feature.to(device), query.to(device), bboxes.to(device), \
				target_bboxes.to(device), num_obj.to(device), num_query.to(device), head.to(device), bert_query_input_ids.to(device), \
				bert_query_attention_mask.to(device), locations.to(device), relations.to(device), spatial_features.to(device)

		prediction, prediction_loss, selected_bbox, target_pred, query_similarity = model.predict(
			query, head, labels, feature, attrs, bboxes, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features
		)		# [B, 32, 4]
	
		prediction2, prediction_loss2, selected_bbox2, target_pred2, query_similarity2 = model2.predict(
			query, head, labels, feature, attrs, bboxes, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features
		)		# [B, 32, 4]

		with torch.no_grad():
			loss = loss_function(prediction_loss, target_pred, query_similarity)
			total_loss += loss.item()

		def show_image(image, title, sentence, queries, boxes_pred, boxes_pred2, boxes_gt):
			"""
			:param image: A numpy array with shape (width, height, depth)
			:param title: A string representing plot title
			:param sentence: A string representing example's sentence
			:param queries: A list of strings with example's queries
			:param boxes_pred: A list of bounding box for each query
			:param boxes_gt: A list of bounding box
			"""
			import random
			import matplotlib as pl
			import matplotlib.pyplot as plt
			import matplotlib.patches as patches

			pl.rcParams["figure.dpi"] = 230

			colors = ([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]] * 10)[:len(queries)] #[(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for _ in queries]
			font_size = 8
			text_props = dict(facecolor="blue", alpha=0.5)

			# plt.figtext(0.5, 0.01, f"{title}", ha="center", fontsize=font_size, wrap=True)

			# plot predictions
			# plt.subplot(1, 1, 1)
			# plt.imshow(image)
			# plt.title("Prediction", fontdict={"fontsize": font_size})



			gt_color = [255/255,102/255,102/255]      # red
			model_color = [153/255,255/255,153/255]   # green
			concept_color = [69/255,205/255,255/255]  # blue

			for i in range(len(queries)):
					
				query = queries[i]
				color = colors[i]

				# for j in range(len(boxes_pred[i])):
				model_box = boxes_pred[i] # [j]
				concept_box = boxes_pred2[i]
				gt_box = boxes_gt[i]

				# shift by 2 pixel model box if it is equal to concept box 
				if model_box[0] == concept_box[0] and model_box[1] == concept_box[1] and model_box[2] == concept_box[2] and model_box[3] == concept_box[3]:
					model_box[0] -= 4
					model_box[1] -= 4
					model_box[2] -= 4
					model_box[3] -= 4

				def draw_box(box, color):
					x1, y1 = box[0], box[1]
					x2, y2 = box[2], box[3]
					xy = (x1, y1)
					width, height = x2 - x1, y2 - y1
					
					rect = patches.Rectangle(xy, width, height, linewidth=2, edgecolor=[*color, 1.0], facecolor=[*color, .2])
					ax.add_patch(rect)

				plt.subplot(1, len(queries), i + 1)
				plt.imshow(image)
				plt.title(query, fontdict={"fontsize": font_size}, y=-0.07)

				ax = plt.gca()

				ax.axes.xaxis.set_visible(False)
				ax.axes.yaxis.set_visible(False)

				draw_box(model_box, model_color)
				draw_box(concept_box, concept_color)
				draw_box(gt_box, gt_color)

				# for box in bboxes[0].cpu().tolist():
				# 	if box == [0, 0, 0, 0]:
				# 		continue
				# 	draw_box(box, [0, 0, 1])
			
			plt.show()

					# plt.text(x, y - 2, query, bbox=text_props, fontsize=5, color="white")

					# if j == 0:
					# 		plt.text(x, y - 2, query, bbox=text_props, fontsize=5, color="white")
					# if j > 0:
					# 		plt.text(x, y - 2, j, bbox=text_props, fontsize=5, color="white")

					

			# plt.subplot(1, 3, 2)
			# plt.imshow(image)
			# plt.title("Prediction", fontdict={"fontsize": font_size})

			# ax = plt.gca()

			# ax.axes.xaxis.set_visible(False)
			# ax.axes.yaxis.set_visible(False)

			# for i in range(len(queries)):
			# 		query = queries[i]
			# 		color = colors[i]

			# 		# for j in range(len(boxes_pred[i])):
			# 		box = boxes_pred2[i] # [j]

			# 		x, y = box[0], box[1]
			# 		xy = (x, y)
			# 		width, height = box[2], box[3]

			# 		plt.text(x, y - 2, query, bbox=text_props, fontsize=5, color="white")

			# 		# if j == 0:
			# 		# 		plt.text(x, y - 2, query, bbox=text_props, fontsize=5, color="white")
			# 		# if j > 0:
			# 		# 		plt.text(x, y - 2, j, bbox=text_props, fontsize=5, color="white")

			# 		rect = patches.Rectangle(xy, width, height, linewidth=1, edgecolor=[*color, .5], facecolor=[*color, .2])
			# 		ax.add_patch(rect)

			# # plot ground truth
			# plt.subplot(1, 3, 3)
			# plt.imshow(image)
			# plt.title("Ground Truth", fontdict={"fontsize": font_size})

			# ax = plt.gca()

			# ax.axes.xaxis.set_visible(False)
			# ax.axes.yaxis.set_visible(False)

			# for query, box, color in zip(queries, boxes_gt, colors):
			# 		x, y = box[0], box[1]
			# 		xy = (x, y)
			# 		width, height = box[2], box[3]

			# 		plt.text(x, y - 2, query, bbox=text_props, fontsize=5, color="white")

			# 		rect = patches.Rectangle(xy, width, height, linewidth=1, edgecolor=[*color, .5], facecolor=[*color, .2])
			# 		ax.add_patch(rect)

			# plt.show()

		image_index = idx[0].item()
		import cv2
		# im = cv2.imread(f"/home/lparolari/Projects/weakvg/data/flickr30k/flickr30k_images/{image_index}.jpg")
		# check file exists
		image_id_str = str(image_index)
		image_id_str = image_id_str.zfill(5)

		image_id_part1 = image_id_str[:2]
		import os
		if not os.path.isfile(f"/home/lparolari/Projects/weakvg/data/referit/refer/data/images/saiapr_tc-12/{image_id_part1}/images/{image_index}.jpg"):
			print(f"File not found: {image_index}")
			continue
		im = cv2.imread(f"/home/lparolari/Projects/weakvg/data/referit/refer/data/images/saiapr_tc-12/{image_id_part1}/images/{image_index}.jpg")
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			
		b = 0
		n_query = num_query[0].item()
		# q = query[b, 0:n_query].cpu().tolist()
		queries = [" ".join([wordEmbedding.word_indexer.get_object(i) for i in q if i > 1]) for q in query[b, 0:n_query].cpu().tolist()]
		labels_str = [wordEmbedding.word_indexer.get_object(i) for i in labels[b, 0:n_query].cpu().tolist()]

		if image_index not in [9107,12288,27561,20820]:
			continue
		# found = False
		# for q in queries:
		# 	if "dog" in q:
		# 		found = True
		# if not found:
		# 	continue

		bb = selected_bbox[0,:num_query].cpu().tolist()
		target_bb = target_bboxes[0,:num_query].cpu().tolist()
		target_bb = union_target(target_bb)

		bb2 = selected_bbox2[0,:num_query].cpu().tolist()
		print(queries)
		print(labels_str)
		print(selected_bbox)

		show_image(im, f"image {image_index}", "sent", queries, bb, bb2, target_bb)
		# show_image(im, "titolo", "sent", queries, bb2, target_bb)
		
		pred_bboxes_list += selected_bbox.cpu().tolist()
		target_bboxes_list += target_bboxes.cpu().tolist()
		num_query_list += num_query.cpu().tolist()
	
	score, point_game_score = evaluate_helper(pred_bboxes_list, target_bboxes_list, num_query_list)
	final_loss = total_loss / n_batches
	return score, point_game_score, final_loss


def evaluate_helper(pred_bboxes, target_bboxes, num_query):
	evaluator = Evaluator()
	gtbox_list = []
	pred_list = []
	for pred, targ, nq in zip(pred_bboxes, target_bboxes, num_query):
		# ipred: [query, 5]
		# itarget: [query, 12, 4]
		if nq > 0:
			pred_list += pred[:nq]
			gtbox_list += union_target(targ[:nq])  # [query, 4]
			# print("pred[:nq]: ", pred[:nq])
			# print("targ[:nq]: ", targ[:nq])
			# print("union_target(targ[:nq]): ", union_target(targ[:nq]))
			# print("nq: ", nq)
			# exit(0)

	accuracy, _, point_game_accuracy= evaluator.evaluate(pred_list, gtbox_list)  # [query, 4]
	return accuracy, point_game_accuracy

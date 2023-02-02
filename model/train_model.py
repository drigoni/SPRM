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
		for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features, image_embedding, text_embedding in tqdm(train_loader):
			# print('===============================================================================')
			# print("bboxes: ", bboxes.shape, bboxes)
			# print("target_bboxes: ", target_bboxes.shape, target_bboxes)
			# print('target_union', [union_target(target_bboxes[0][:num_query[0]])])
			# print("num_obj: ", num_obj.shape, num_obj)
			# print("num_query: ", num_query.shape, num_query)
			
			if (use_gpu):
				idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, \
					head, bert_query_input_ids, bert_query_attention_mask, locations, relations, \
					spatial_features, image_embedding, text_embedding = \
				idx.to(device), labels.to(device), attrs.to(device), feature.to(device), query.to(device), bboxes.to(device), \
					target_bboxes.to(device), num_obj.to(device), num_query.to(device), head.to(device), bert_query_input_ids.to(device), \
					bert_query_attention_mask.to(device), locations.to(device), relations.to(device), spatial_features.to(device), \
					image_embedding.to(device), text_embedding.to(device)
			
			# training steps
			optimizer.zero_grad()
			prediction_scores, prediction_loss, target_pred, query_similarity = model.forward(
				query, head, labels, feature, attrs, bboxes, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features,
				image_embedding, text_embedding
			)
			loss = loss_function(prediction_loss, target_pred, query_similarity)
			# loss.backward()   # TODO: uncomment
			# optimizer.step()  # TODO: uncomment

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


def evaluate(test_loader, model, loss_function, device_str='cuda'):
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() and device_str == 'cuda' else "cpu")
	use_gpu = torch.cuda.is_available()
	model = model.float()
	pred_bboxes_list = []
	target_bboxes_list = []
	num_query_list = []
	n_batches = 0
	total_loss = 0

	model.eval()
	for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features, image_embedding, text_embedding in tqdm(test_loader):
		n_batches += 1

		if (use_gpu):
			idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, \
				head, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features, \
				image_embedding, text_embedding = \
			idx.to(device), labels.to(device), attrs.to(device), feature.to(device), query.to(device), bboxes.to(device), \
				target_bboxes.to(device), num_obj.to(device), num_query.to(device), head.to(device), bert_query_input_ids.to(device), \
				bert_query_attention_mask.to(device), locations.to(device), relations.to(device), spatial_features.to(device), \
				image_embedding.to(device), text_embedding.to(device)

		prediction, prediction_loss, selected_bbox, target_pred, query_similarity = model.predict(
			query, head, labels, feature, attrs, bboxes, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features,
			image_embedding, text_embedding
		)		# [B, 32, 4]

		with torch.no_grad():
			loss = loss_function(prediction_loss, target_pred, query_similarity)
			total_loss += loss.item()

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

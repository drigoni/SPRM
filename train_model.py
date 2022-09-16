import time
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.evaluator import Evaluator
from utils.utils import union_target

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = FutureWarning)


def train(model, loss_function, train_loader, test_loader, lr = 1e-4, epochs = 25, device_str='cuda'):
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() and device_str == 'cuda' else "cpu")
	use_gpu = torch.cuda.is_available()
	if use_gpu:
		print("CUDA available, numof GPUs: ", torch.cuda.device_count())
		model.to(device)
		loss_function.to(device)

	model = model.float()
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)

	print("--- Before Training...")
	score = evaluate(test_loader, model, device_str)
	print("Eval score on test dataset:", score)

	for epoch in range(epochs):
		t = time.time()
		total_loss = 0
		correct_preds = 0
		all_preds = 0
		n_batches = 0

		model.train(True)
		for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head in tqdm(train_loader):
			if (use_gpu):
				idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head = \
				idx.to(device), labels.to(device), attrs.to(device), feature.to(device), query.to(device), bboxes.to(device), target_bboxes.to(device), num_obj.to(device), num_query.to(device), head.to(device)
			
			# training steps
			optimizer.zero_grad()
			prediction_scores, prediction_loss, target_pred = model.forward(query, head, labels, feature, attrs, bboxes, num_query, num_obj)
			loss = loss_function(prediction_loss, target_pred)
			loss.backward()
			optimizer.step()

			# update variables
			n_batches += 1
			final_pred = torch.argmax(prediction_loss, dim = -1)  # [b]
			correct_preds += int(final_pred.eq(target_pred).sum())
			all_preds += len(final_pred)
			total_loss += loss.item()
			
		t1 = time.time()
		print("--- EPOCH", epoch)
		print("     time:", t1 - t)
		print("     total loss:", total_loss / n_batches)
		print("     supervised accuracy on training set: ", correct_preds / all_preds)

		t2 = time.time()
		# evaluate
		score, supacc = evaluate(test_loader, model, device_str)
		print("     eval time:", time.time() - t2)
		print("     supervised accuracy on test dataset:", supacc)
		print("     eval score on test dataset:", score)


def evaluate(test_loader, model, device_str='cuda'):
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() and device_str == 'cuda' else "cpu")
	use_gpu = torch.cuda.is_available()
	
	correct_preds = 0
	all_preds = 0

	model = model.float()

	pred_bboxes_list = []
	target_bboxes_list = []
	num_query_list = []

	model.eval()
	for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head in tqdm(test_loader):
		if (use_gpu):
			idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head = \
			idx.to(device), labels.to(device), attrs.to(device), feature.to(device), query.to(device), bboxes.to(device), target_bboxes.to(device), num_obj.to(device), num_query.to(device), head.to(device)

		prediction, selected_bbox, prediction_loss, target_pred = model.predict(query, head, labels, feature, attrs, num_query, num_obj, bboxes)		# [B, 32, 4]

		# sup acc
		final_pred = torch.argmax(target_pred, dim = -1)  # [B]
		correct_preds += int(final_pred.eq(target_pred).sum())
		all_preds += len(prediction)

		pred_bboxes_list += selected_bbox.cpu().tolist()
		target_bboxes_list += target_bboxes.cpu().tolist()
		num_query_list += num_query.cpu().tolist()
	
	score = evaluate_helper(pred_bboxes_list, target_bboxes_list, num_query_list)
	supacc = correct_preds / all_preds

	return score, supacc


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

	accuracy, _ = evaluator.evaluate(pred_list, gtbox_list)  # [query, 4]
	return accuracy

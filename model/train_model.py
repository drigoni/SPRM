import time
import warnings

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from utils.evaluator import Evaluator
from utils.utils import union_target

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = FutureWarning)


def train(model, loss_function, train_loader, test_loader, args, lr = 1e-4, epochs = 25, device_str='cuda'):
	# init wandb
	wandb.init(project="weakvg", entity="drigoni", config=vars(args))
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

	print("--- Before Training...")
	score = evaluate(test_loader, model, device_str)
	print("Eval score on test dataset:", score)

	for epoch in range(epochs):
		t = time.time()
		total_loss = 0
		n_batches = 0

		model.train(True)
		for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head in tqdm(train_loader):
			# print('===============================================================================')
			# print("bboxes: ", bboxes.shape, bboxes)
			# print("target_bboxes: ", target_bboxes.shape, target_bboxes)
			# print('target_union', [union_target(target_bboxes[0][:num_query[0]])])
			# print("num_obj: ", num_obj.shape, num_obj)
			# print("num_query: ", num_query.shape, num_query)
			
			if (use_gpu):
				idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head = \
				idx.to(device), labels.to(device), attrs.to(device), feature.to(device), query.to(device), bboxes.to(device), target_bboxes.to(device), num_obj.to(device), num_query.to(device), head.to(device)
			
			# training steps
			optimizer.zero_grad()
			prediction_scores, prediction_loss, target_pred = model.forward(query, head, labels, feature, attrs, bboxes)
			loss = loss_function(prediction_loss, target_pred)
			loss.backward()
			optimizer.step()

			# update variables
			n_batches += 1
			total_loss += loss.item()
			
		t1 = time.time()
		print("--- EPOCH", epoch)
		print("     time:", t1 - t)
		print("     total loss:", total_loss / n_batches)

		t2 = time.time()
		# evaluate
		score = evaluate(test_loader, model, device_str)
		print("     eval time:", time.time() - t2)
		print("     eval score on test dataset:", score)

		wandb.log({	"loss": total_loss / n_batches,
					"acc_val": score
					})


def evaluate(test_loader, model, device_str='cuda'):
	# device
	device = torch.device("cuda:0" if torch.cuda.is_available() and device_str == 'cuda' else "cpu")
	use_gpu = torch.cuda.is_available()
	model = model.float()
	pred_bboxes_list = []
	target_bboxes_list = []
	num_query_list = []

	model.eval()
	for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head in tqdm(test_loader):
		if (use_gpu):
			idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head = \
			idx.to(device), labels.to(device), attrs.to(device), feature.to(device), query.to(device), bboxes.to(device), target_bboxes.to(device), num_obj.to(device), num_query.to(device), head.to(device)

		prediction, prediction_loss, selected_bbox = model.predict(query, head, labels, feature, attrs, bboxes)		# [B, 32, 4]

		pred_bboxes_list += selected_bbox.cpu().tolist()
		target_bboxes_list += target_bboxes.cpu().tolist()
		num_query_list += num_query.cpu().tolist()
	
	score = evaluate_helper(pred_bboxes_list, target_bboxes_list, num_query_list)
	return score


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

	accuracy, _ = evaluator.evaluate(pred_list, gtbox_list)  # [query, 4]
	return accuracy
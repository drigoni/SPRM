import argparse

import os
import torch
from torch.utils.data import DataLoader
import wandb

from model.dataset import Flickr30dataset
from model.model import MATnet
from model.train_model import evaluate
from utils.utils import load_vocabulary, init_net


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type = str, default = "saved/model_0527_a20.pt",
						help = "saved model name")
	parser.add_argument('--batch', type = int, default = 64,
						help = "batch size for training")
	parser.add_argument('--device', type= str, default='cuda',
						choices=['cuda', 'cpu'])
	# model params
	parser.add_argument('--cosine_similarity_strategy', type= str, default='mean',
						choices=['mean', 'max'])
	parser.add_argument('--loss_strategy', type= str, default='ce',
						choices=['luca', 'all', 'ce'])
	parser.add_argument('--emb_dim', type= int, default=300)
	parser.add_argument('--word_emb_dim', type= int, default=300)
	parser.add_argument('--feature_dim', type= int, default=2048)
	parser.add_argument('--cosine_weight', type=float, default=0.5)
	parser.add_argument('--use_att_for_query', action = 'store_true', help = "Disable LSTM for query features and use attention.")
	parser.add_argument('--use_mean_in_loss', action = 'store_true', help = "Consider all the couple <query, box> in the loss calculation.")

	# debug mode
	parser.add_argument('--debug', action = 'store_true')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	wordEmbedding = load_vocabulary("data/glove/glove.6B.300d.txt")
	test_dset = Flickr30dataset(wordEmbedding, "test")

	test_loader = DataLoader(test_dset, batch_size = 32, num_workers = 4, drop_last = True, shuffle = True)
	model = MATnet(wordEmbedding, args)
	device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
	use_gpu = torch.cuda.is_available()
	if use_gpu:
		print("CUDA available, numof GPUs: ", torch.cuda.device_count())
	init_net(model, args.file)

	model = model.to(device)
	model.eval()

	score = evaluate(test_loader, model)
	print("Evaluation localization score:", score)

	# wandb.log({"acc_test": score})


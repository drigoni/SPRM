import argparse
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Flickr30dataset
# from model import MATnet
from model_drigoni import MATnet
from loss import WeakVtgLoss
from train_model import train, evaluate
from utils.utils import load_vocabulary

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = FutureWarning)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch', type = int, default = 64,
						help = "batch size for training")
	parser.add_argument('--lr', type = float, default = 1e-5,
						help = "learning rate")
	parser.add_argument('--eval', action = 'store_true',
						help = "evaluation mode")
	parser.add_argument('--epochs', type = int, default = 25,
						help = "training epochs")
	parser.add_argument('--save_name', type = str, default = 'model',
						help = "name for saved model")
	parser.add_argument('--test_set', action = 'store_true',
						help = "use test set for evaluation")
	parser.add_argument('--seed', type = int, default = 0,
						help = "random seed")
	parser.add_argument('--device', type= str, default='cuda',
						choices=['cuda', 'cpu'])
	# model params
	parser.add_argument('--cosine_similarity_strategy', type= str, default='mean',
						choices=['mean', 'max'])
	parser.add_argument('--loss_strategy', type= str, default='luca',
						choices=['luca', 'all', 'ce'])
	parser.add_argument('--emb_dim', type= int, default=300)
	parser.add_argument('--feature_dim', type= int, default=2048)
	parser.add_argument('--cosine_weight', type=float, default=0.5)

	# debug mode
	parser.add_argument('--debug', action = 'store_true')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	print(args)

	# params and seeds
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	save_path = os.path.join("output", args.save_name + '.pt')

	# config
	wordEmbedding = load_vocabulary("data/glove/glove.6B.300d.txt")
	if args.test_set:
		test_dset = Flickr30dataset(wordEmbedding, "test")
	else:
		test_dset = Flickr30dataset(wordEmbedding, "val")
	test_loader = DataLoader(test_dset, batch_size = args.batch, num_workers = 4, drop_last = True, shuffle = True)
	model = MATnet(wordEmbedding, args)
	loss = WeakVtgLoss(args) 

	if args.eval:
		score = evaluate(test_loader, model, device_str=args.device)
		print("untrained eval score:", score)
	else:
		if args.debug:
			train_dset = Flickr30dataset(wordEmbedding, "test")
		else:
			train_dset = Flickr30dataset(wordEmbedding, "train")
		train_loader = DataLoader(train_dset, batch_size = args.batch, num_workers = 4, drop_last = True, shuffle = True)
		train(model, loss, train_loader, test_loader, args, lr = args.lr, epochs = args.epochs, device_str=args.device)
		torch.save(model.cpu().state_dict(), save_path)
		print("save model to", save_path)

import argparse

import os
import torch
from torch.utils.data import DataLoader

from dataset import Flickr30dataset
from model import MATnet
from train_model import evaluate
from utils.utils import load_vocabulary, init_net


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type = str, default = "saved/model_0527_a20.pt",
						help = "saved model name")

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	wordEmbedding = load_vocabulary("data/glove/glove.6B.300d.txt")
	test_dset = Flickr30dataset(wordEmbedding, "test")

	test_loader = DataLoader(test_dset, batch_size = 32, num_workers = 4, drop_last = True, shuffle = True)
	net = MATnet(wordEmbedding)
	if torch.cuda.is_available():
		print("CUDA available")
		net.cuda()
	init_net(net, args.file)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = net.to(device)
	net.eval()

	score, _ = evaluate(test_loader, net)
	print("evaluation localization score:", score)


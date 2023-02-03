import argparse
import os
import random
import warnings

import wandb
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from model.dataset_flickr import Flickr30Dataset
from model.dataset_referit import ReferitDataset
# from model import MATnet
from model.model import ConceptNet
from model.model_MATnet import MATnet
from model.loss import WeakVtgLoss
from model.train_model import train, evaluate
from utils.utils import load_vocabulary, init_net

with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category = FutureWarning)



def get_datasets(args):
	if args.test_set:
		test_split = "test"
	else:
		test_split = "val"

	options = {
		"train_fract": args.train_fract, 
		"do_spellchecker": args.do_spellchecker, 
		"do_oov": args.do_oov,
		"do_head": args.do_head,
		"do_bert": args.do_bert,
		"do_relations": args.do_relations, 
		"do_locations": args.do_locations,
		"relations_strategy": args.relations_strategy,
	}

	if args.dataset == "flickr30k":
		test_dset = Flickr30Dataset(wordEmbedding, test_split, **options)
		if args.dry_run:
			train_dset = test_dset
		else:
			train_dset = Flickr30Dataset(wordEmbedding, "train", **options)
	else:
		test_dset = ReferitDataset(wordEmbedding, test_split, **options)
		if args.dry_run:
			train_dset = test_dset
		else:
			train_dset = ReferitDataset(wordEmbedding, "train", **options)
	return train_dset, test_dset


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
	parser.add_argument('--dataset', type= str, default='flickr30k',
						choices=['flickr30k', 'referit'])
	# model params
	parser.add_argument('--cosine_similarity_strategy', type= str, default='mean',
						choices=['mean', 'max'])
	parser.add_argument('--loss_strategy', type= str, default='luca',
						choices=['luca', 'luca_min', 'luca_max', 'all', 'ce'])
	parser.add_argument('--do_spellchecker', action="store_true", default=False)
	parser.add_argument('--do_oov', action="store_true", default=False)
	parser.add_argument('--do_negative_weighting', action="store_true", default=False)
	parser.add_argument('--do_head', action="store_true", default=False)
	parser.add_argument('--dry_run', action="store_true", default=False)
	parser.add_argument('--emb_dim', type= int, default=300)
	parser.add_argument('--word_emb_dim', type= int, default=300)
	parser.add_argument('--feature_dim', type= int, default=2048)
	parser.add_argument('--cosine_weight', type=float, default=0.5)
	parser.add_argument('--use_att_for_query', action = 'store_true', help = "Disable LSTM for query features and use attention.")
	parser.add_argument('--use_mean_in_loss', action = 'store_true', help = "Consider all the couple <query, box> in the loss calculation.")
	parser.add_argument('--MATnet', action = 'store_true', help = "True when we want to use the original model.")
	parser.add_argument('--train_fract', type=float, default=1.0, help = "Fraction of training set to load for training.")
	parser.add_argument('--similarity_strategy', type=str, default="cosine_similarity", choices=["euclidean_distance", "cosine_similarity"])
	parser.add_argument('--use_head_for_query_embedding', action="store_true", default=False)
	parser.add_argument('--image_net_dropout', type=float, default=0.0)
	parser.add_argument('--query_net_dropout', type=float, default=0.0)
	parser.add_argument('--file', type = str)
	parser.add_argument('--lstm_num_layers', type=int, default=1)
	parser.add_argument('--use_bidirectional_lstm', action="store_true", default=False)
	parser.add_argument('--do_checkpoint', action="store_true", default=False)
	parser.add_argument('--use_head_for_concept_embedding', action="store_true", default=False)
	parser.add_argument('--do_bert', action="store_true", default=False)
	parser.add_argument('--use_minilm_for_query_embedding', action="store_true", default=False)
	parser.add_argument('--loss_sigmoid_slope', type=float, default=1.0)
	parser.add_argument('--do_relations', action="store_true", default=False)
	parser.add_argument('--do_locations', action="store_true", default=False)
	parser.add_argument('--use_wv_freezed', action="store_true", default=False)
	parser.add_argument('--use_spatial_features', action="store_true", default=False)
	parser.add_argument('--use_relations_for_concept_embedding', action="store_true", default=False)
	parser.add_argument('--relations_strategy', type=str, default="none", choices=["none", "pseudo-q", "baseline"])
	parser.add_argument('--use_clip_emb', action="store_true", default=False)

	# debug mode
	parser.add_argument('--debug', action = 'store_true')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	print(args)

	wandb.init(project="weakvg", entity="weakly_guys", config=vars(args))

	# params and seeds
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	output_dir = "output/flickr" if args.dataset == "flickr30k" else "output/referit"
	save_path = os.path.join(output_dir, args.save_name + '.pt')
	save_checkpoint = os.path.join(output_dir, args.save_name + '_{}.pt')
	os.makedirs(output_dir, exist_ok=True)

	# config
	wordEmbedding = load_vocabulary(f"data/glove/glove.6B.{args.word_emb_dim}d.txt")
	# get dataset
	train_dset, test_dset = get_datasets(args)
	train_loader = DataLoader(train_dset, batch_size = args.batch, num_workers = 4, drop_last = True, shuffle = True)
	test_loader = DataLoader(test_dset, batch_size = args.batch, num_workers = 4, drop_last = True, shuffle = True)
	# load model
	if args.MATnet:
		model = MATnet(wordEmbedding, args)
	else:
		model = ConceptNet(wordEmbedding, args)
	loss = WeakVtgLoss(args) 

	if args.test_set:
		if args.file:
			init_net(model, args.file)
			model.to(device=args.device)
		score = evaluate(test_loader, model, loss, device_str=args.device)
		print("untrained eval score:", score)
	else:
		best_model = train(model, loss, train_loader, test_loader, args, lr = args.lr, epochs = args.epochs, device_str=args.device, save_checkpoint=save_checkpoint if args.do_checkpoint else None)
		torch.save(best_model.cpu().state_dict(), save_path)
		print("save model to", save_path)

import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import warnings
import numpy as np
from model.dataset_flickr import Flickr30Dataset
from model.dataset_referit import ReferitDataset
from utils.utils import load_vocabulary, union_target, union
from tqdm import tqdm



def get_datasets(args):
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
        train_dset = Flickr30Dataset(wordEmbedding, 'train', **options)
        val_dset = Flickr30Dataset(wordEmbedding, 'val', **options)
        test_dset = Flickr30Dataset(wordEmbedding, 'test', **options)
    else:
        train_dset = ReferitDataset(wordEmbedding, 'train', **options)
        val_dset = ReferitDataset(wordEmbedding, 'val', **options)
        test_dset = ReferitDataset(wordEmbedding, 'test', **options)
    return train_dset, val_dset, test_dset


def intersection_over_union(boxA, boxB):
    """
    Intersection over union of two bounding boxes.
    :param boxA: bounding box A. Format: [xmin, ymin, xmax, ymax]
    :param boxB: bounding box B.
    :return: intersection over union score.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def evaluate(dataset):
    all_scores = []
    for idx, labels, attrs, feature, query, bboxes, target_bboxes, num_obj, num_query, head, bert_query_input_ids, bert_query_attention_mask, locations, relations, spatial_features in tqdm(dataset):
        query = query.cpu().tolist()
        target_bboxes = target_bboxes.cpu().tolist()
        bboxes = bboxes.cpu().tolist()
        assert len(query) == len(target_bboxes)
        for q in range(num_query):
            target_list = target_bboxes[q]
            t = union(target_list)
            ious = [intersection_over_union(t, bboxes[i]) for i in range(len(bboxes)) if i < num_obj]
            all_scores.append(max(ious))
    hits = [1 if s > 0.5 else 0 for s in all_scores]
    upper_bound = sum(hits) / len(hits)
    return upper_bound


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
	parser.add_argument('--feature_dim', type= int, default=512)
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

	# debug mode
	parser.add_argument('--debug', action = 'store_true')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # params and seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # config
    wordEmbedding = load_vocabulary(f"data/glove/glove.6B.{args.word_emb_dim}d.txt")
    # get dataset
    train_dset, val_dset, test_dset = get_datasets(args)

    splits = {
        'train': train_dset, 
        'val': val_dset,
        'test': test_dset
        }
    for split_name, split_data in splits.items():
        score = evaluate(split_data)
        print("Coverage dataset {}, split={}: {}".format(args.dataset, split_name, score))

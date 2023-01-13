import _pickle as cPickle
import json
import os
import re
from xml.etree.ElementTree import parse

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import utils
from spellchecker import SpellChecker
import spacy
import random
import argparse
import os.path as osp
import json
import pickle as pickle
from collections import defaultdict

from model.dataset import load_boxes_classes, get_spacy_nlp, get_box_relations, get_query_locations


class ReferitDataset(Dataset):
	def __init__(self, wordEmbedding, name = 'train', dataroot = 'data/referit/',  train_fract=1.0, do_spellchecker=False, do_oov=False, do_head=False, do_bert=False, do_relations=False, do_locations=False):
		super(ReferitDataset, self).__init__()
		print("Loading Referit dataset. Split: ", name)
		self.indexer = wordEmbedding.word_indexer
		print("Loading entries...")
		self.entries, self.img_id2idx = load_dataset(name, dataroot, train_fract=train_fract, do_spellchecker=do_spellchecker, do_head=do_head, do_bert=do_bert, do_relations=do_relations, do_locations=do_locations)
		print("Loading classes...")
		self.class_labels = load_boxes_classes('data/objects_vocab.txt', word_embedding=wordEmbedding, word_indexer=self.indexer, do_spellchecker=do_spellchecker, do_oov=do_oov)
		# img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
		print("Loading features...")
		h5_path = os.path.join(dataroot, '%s_features_compress.hdf5' % name)

		with h5py.File(h5_path, 'r') as hf:
			self.features = np.array(hf.get('image_features'))	# different from flickr30k. We generate them
			self.pos_boxes = np.array(hf.get('pos_boxes'))
			self.spatial_features = np.array(hf.get('spatial_features'))
		print("Dataset loaded.")
			

	def _get_entry(self, index):
		entry = self.entries[index]
		attrs = []
			# attrs = entry['attrs']
		return entry['image'], entry['labels'], entry['query'], entry['head'], attrs, \
			entry['detected_bboxes'], entry['target_bboxes'], entry['bert_query_input_ids'], \
			entry['bert_query_attention_mask'], entry['locations'], entry['relations'], \
			entry['width'], entry['height']

	def __getitem__(self, index):
		'''
		:return: labels, query, deteced_bboxes, number of querys

		labels: [K=64] index
		attrs: [K=64] index
		bboxes: [K=64, 5] index
		querys: [Q=32, len=12] index
		query_feats: [Q, dim]
		label_feats: [K, dim]
		target_bboxes: [Q=32, Boxes, 4] index
		'''

		K = 100		# number of boxes
		Q = 39		# number of queries
		lens = 12	# length of the query
		B = 20		# max number of target boxes to consider for each query.

		imgid, labels, querys, heads, attrs, bboxes, target_bboxes, bert_query_input_ids, \
			bert_query_attention_mask, locations, relations, width, height = self._get_entry(index)

		idx = self.img_id2idx[int(imgid)]  # to retrieve pos in pos_box
		pos = self.pos_boxes[idx]

		feature = torch.from_numpy(self.features[pos[0]:pos[1]]).float()

		spatial_features = torch.from_numpy(self.spatial_features[pos[0]:pos[1]]).float()

		if feature.size(0) < K:
			pad = nn.ZeroPad2d((0, 0, 0, K - feature.size(0)))
			feature = pad(feature)
		else:
			feature = feature[:K]

		if spatial_features.size(0) < K:
			pad = nn.ZeroPad2d((0, 0, 0, K - spatial_features.size(0)))
			spatial_features = pad(spatial_features)
		else:
			spatial_features = spatial_features[:K]
		spatial_features[:, 4] = spatial_features[:, 4] * spatial_features[:, 5]
		spatial_features = spatial_features[:, :5]

		num_obj = min(len(labels), K)
		num_query = min(len(querys), Q)

		labels_idx = [0] * K
		labels_idx[:num_obj] = [max(self.indexer.index_of(re.split(' ,', w)[-1]), 1) for w in labels]
		labels_idx = labels_idx[:K]

		attr_idx = [0] * K

		# NOTE: padding is represented with index 0, while missing words are represented with value 1
		querys_idx = []
		for q in querys:
			q = q.lower().split()
			lis = [0] * lens
			for i in range(min(len(q), lens)):
				lis[i] = max(self.indexer.index_of(q[i]), 1)
			querys_idx.append(lis)
		while(len(querys_idx) < Q):
			querys_idx.append([0] * lens)
		querys_idx = querys_idx[:Q]

		# bert tokens
		# should be already padded for `lens`
		while(len(bert_query_input_ids) < Q):
			bert_query_input_ids.append([0] * lens)
		bert_query_input_ids = bert_query_input_ids[:Q]
		while(len(bert_query_attention_mask) < Q):
			bert_query_attention_mask.append([0] * lens)
		bert_query_attention_mask = bert_query_attention_mask[:Q]

		heads_idx = []
		for h in heads:
			h = h.lower().split()
			lis = [0] * lens
			for i in range(min(len(h), lens)):
				lis[i] = max(self.indexer.index_of(h[i]), 1)
			heads_idx.append(lis)
		while (len(heads_idx) < Q):
			heads_idx.append([0] * lens)
		heads_idx = heads_idx[:Q]

		pad_location = [0, 0, 0, 0, 0, 0]
		while (len(locations) < Q):
			locations.append(pad_location)
		locations = locations[:Q]

		padbox = [0, 0, 0, 0]

		while (len(bboxes) < K):
			bboxes.append(padbox)
		bboxes = bboxes[:K]

		bboxes = torch.tensor(bboxes)
		area = (bboxes[..., 3] - bboxes[..., 1]) * (bboxes[..., 2] - bboxes[..., 0])
		bboxes = torch.cat((bboxes, area.unsqueeze_(-1)), -1)

		for bbox in target_bboxes:
			while (len(bbox) < B):
				bbox.append(padbox)
		target_bboxes = [b[:B] for b in target_bboxes]
		padline = [padbox for i in range(B)]
		while (len(target_bboxes) < Q):
			target_bboxes.append(padline)
		target_bboxes = target_bboxes[:Q]

		pad_relation = [0, 0, 0, 0, 0, 0]
		while (len(relations) < K):
			relations.append(pad_relation)
		relations = relations[:K]

		assert len(labels_idx) == K
		assert len(bboxes) == K
		assert len(querys_idx) == Q
		assert len(heads_idx) == Q
		assert len(target_bboxes) == Q
		assert len(bert_query_input_ids) == Q
		assert len(bert_query_attention_mask) == Q
		assert len(relations) == K
		assert len(locations) == Q

		# torch.tensor(int(imgid))
		# torch.tensor(labels_idx)
		# torch.tensor(attr_idx)
		# torch.tensor(querys_idx)
		# torch.tensor(target_bboxes)
		# torch.tensor(num_obj)
		# torch.tensor(num_query)
		# torch.tensor(heads_idx)

		return torch.tensor(int(imgid)), torch.tensor(labels_idx), torch.tensor(attr_idx), feature, \
			torch.tensor(querys_idx), bboxes, torch.tensor(target_bboxes), torch.tensor(num_obj), \
			torch.tensor(num_query), torch.tensor(heads_idx), torch.tensor(bert_query_input_ids), \
			torch.tensor(bert_query_attention_mask), torch.tensor(locations), torch.tensor(relations), \
			spatial_features

	def __len__(self):
		return len(self.entries)


def load_train_referit(dataroot, img_id2idx, obj_detection, annotations, images_size, do_spellchecker=False, do_head=False, do_bert=False, do_relations=False, do_locations=False):
	"""Load entries

	img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
	dataroot: root path of dataset
	name: 'train', 'val', 'test-dev2015', test2015'
	"""
	if do_spellchecker:
		spell = SpellChecker()
	if do_head:
		spacy_nlp = get_spacy_nlp()
	if do_bert:
		from transformers import AutoTokenizer
		
		tokenizer = AutoTokenizer.from_pretrained(
			"sentence-transformers/all-MiniLM-L6-v2",
			padding=True, truncation=True,
		)
	entries = []


	for image_id, rev_index in tqdm(img_id2idx.items()):
		image_id = str(image_id)
		bboxes = obj_detection[image_id]['bboxes']
		labels = obj_detection[image_id]['classes']  # [B, 4]
		attrs = obj_detection[image_id]['attrs'] if 'attrs' in obj_detection[image_id].keys() else []
		image_annotations = annotations[image_id]
		assert (len(bboxes) == len(labels))

		image_size = images_size[image_id]
		image_width = image_size[0]
		image_height = image_size[1]

		for ann in image_annotations:
			query = ann['query']
			target_bboxes = ann['bbox']
			# print(target_bboxes)

			# correct phrase
			if do_spellchecker:
				query_corrected = []
				for noun_phrase in query:
					words = noun_phrase.split(' ')
					words_corrected = [spell.correction(word) or word for word in words]
					noun_phrase_corrected = ' '.join(words_corrected)
					query_corrected.append(noun_phrase_corrected)
				if query != query_corrected:
					print(query, "->", query_corrected)
					query = query_corrected

			head = []
			if do_head:
				for noun_phrase in query:
					doc = spacy_nlp(noun_phrase)
					phrase_heads = [chunk.root.text for chunk in doc.noun_chunks]
					if len(phrase_heads) == 0:
						phrase_heads = [doc[-1].text]  # fallback to last word
					phrase_head = ' '.join(phrase_heads)   # we treat multiple heads as a phrase
					head.append(phrase_head)
			
			relations = get_box_relations(bboxes, labels, image_width, image_height, enabled=do_relations)

			locations = get_query_locations(query, enabled=do_locations)

			bert_query_input_ids = []
			bert_query_attention_mask = []
			if do_bert:
				bert_tokenized = tokenizer(
					query, 
					padding="max_length",
					truncation=True,
					max_length=12,
					return_tensors="np",
				)
				bert_query_input_ids = bert_tokenized["input_ids"].tolist()
				bert_query_attention_mask = bert_tokenized["attention_mask"].tolist()

			# NOTE: flickr30k, for each query, has associated multiple bounding target boxes 
			entry = {
				'image': image_id,
				'target_bboxes': [[target_bboxes] for i in range(len(query))],  # list with target_bboxes for each query
				"detected_bboxes": bboxes,  # list, in order of labels
				'labels': labels,
				'attrs': attrs,
				'query': query,	# list of queries
				'head': head,
				'bert_query_input_ids': bert_query_input_ids,
				'bert_query_attention_mask': bert_query_attention_mask,
				'relations': relations,
				'locations': locations,
				'width': image_width,
				'height': image_height,
			}
			entries.append(entry)
	return entries


def load_dataset(name = 'train', dataroot = 'data/referit/', train_fract=1.0, do_spellchecker=False, do_head=False, do_bert=False, do_relations=False, do_locations=False):
	obj_detection_dict = json.load(open("data/referit/%s_detection_dict.json" % name, "r"))
	img_id2idx = cPickle.load(open(os.path.join(dataroot, '%s_imgid2idx.pkl' % name), 'rb'))
	ref_ann, ref_inst_ann, annotations = load_referit_annotations(dataroot + "refer/data/")
	# print(annotations[0].keys())	# list of dict_keys(['img_id', 'ann_id', 'bbox', 'split', 'query'])
	annotations_dict = defaultdict(list)
	for i in annotations:
		image_id = i['img_id'][:-4] # remove .jpg
		annotations_dict[image_id].append(i)
	
	images_size = json.load(open(os.path.join(dataroot, '%s_images_size.json' % name), 'r'))
	
	# print("Max number of queries: ", max([len(v) for k, v in annotations_dict.items()]))
	# print("Min number of queries: ", min([len(v) for k, v in annotations_dict.items()]))
	# print("Mean number of queries: ", np.mean([len(v) for k, v in annotations_dict.items()]))
	# print("Max length of queries: ", max([len(e['query']) for k, v in annotations_dict.items() for e in v]))
	# print("Min length of queries: ", min([len(e['query']) for k, v in annotations_dict.items() for e in v]))
	# print("Mean length of queries: ", np.mean([len(e['query']) for k, v in annotations_dict.items() for e in v]))
	#
	#annotations_dict_grouped = defaultdict(list)
	#for key, val in annotations_dict.items():
	#	tmp_dict = {
	#		'img_id': [v['img_id'] for v in val],
	#		'ann_id': [v['ann_id'] for v in val],
	#		'bbox': [v['bbox'] for v in val],
	#		'split': [v['split'] for v in val],
	#		'query': [q for v in val for q in v['query']],
	#	}
	#	annotations_dict_grouped[key] = tmp_dict
	#print("Max number of queries: ", max([len(v['query']) for k, v in annotations_dict_grouped.items()]))
	#print("Min number of queries: ", min([len(v['query']) for k, v in annotations_dict_grouped.items()]))
	#print("Max length of queries: ", max([len(q) for k, v in annotations_dict_grouped.items() for q in v['query']]))
	#print("Min length of queries: ", min([len(q) for k, v in annotations_dict_grouped.items() for q in v['query']]))
	#exit(1)

	# subsample dataset accordin to train_fract value only in training set
	if name == 'train' and train_fract < 1.0:
		random.seed(2022)
		n_images = len(img_id2idx)
		n_subset = train_fract * n_images
		subset_idx = random.sample([i for i in img_id2idx.keys()], int(n_subset))
		img_id2idx = {key: img_id2idx[key] for key in subset_idx}

	entries = load_train_referit(dataroot, img_id2idx, obj_detection_dict, annotations_dict, images_size, do_spellchecker=do_spellchecker, do_head=do_head, do_bert=do_bert, do_relations=do_relations, do_locations=do_locations)

	return entries, img_id2idx


def load_referit_annotations(data_root):
	def get_annotations(ref_ann, ref_inst_ann):
		instance_dict_by_ann_id = {
			v['id']: ind for ind, v in enumerate(ref_inst_ann)}
		out_dict_list = []
		for rj in ref_ann:
			spl = rj['split']
			sents = rj['sentences']
			ann_id = rj['ann_id']
			inst_bbox = ref_inst_ann[instance_dict_by_ann_id[ann_id]]['bbox']
			# Saving in [x0, y0, x1, y1] format
			inst_bbox = [inst_bbox[0], inst_bbox[1],
						inst_bbox[2] + inst_bbox[0], inst_bbox[3] + inst_bbox[1]]

			assert inst_bbox[0] <= inst_bbox[2]
			assert inst_bbox[1] <= inst_bbox[3]

			sents = [s['raw'] for s in sents]
			sents = [t.strip().lower() for t in sents]
			out_dict = {}
			out_dict['img_id'] = f"{rj['image_id']}.jpg"
			out_dict['ann_id'] = ann_id
			out_dict['bbox'] = inst_bbox
			out_dict['split'] = spl
			out_dict['query'] = sents
			out_dict_list.append(out_dict)
		return out_dict_list
	splitBy = 'berkeley'
	data_dir = osp.join(data_root, 'refclef')
	ref_ann_file = osp.join(data_dir, f'refs({splitBy}).p')
	ref_instance_file = osp.join(data_dir, 'instances.json')
	ref_ann = pickle.load(open(ref_ann_file, 'rb'))
	ref_inst = json.load(open(ref_instance_file, 'r'))
	ref_inst_ann = ref_inst['annotations']
	annotations = get_annotations(ref_ann, ref_inst_ann)
	return ref_ann, ref_inst_ann, annotations

from typing import List
import _pickle as cPickle
import json
import os
import re
from xml.etree.ElementTree import parse
import logging

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

from model.dataset import load_boxes_classes, get_spacy_nlp


class Flickr30Dataset(Dataset):
	def __init__(self, wordEmbedding, name = 'train', dataroot = 'data/flickr30k/',  train_fract=1.0, do_spellchecker=False, do_oov=False, do_head=False, do_bert=False):
		super(Flickr30Dataset, self).__init__()
		print("Loading flickr30k dataset. Split: ", name)
		self.indexer = wordEmbedding.word_indexer
		print("Loading entries...")
		self.entries, self.img_id2idx = load_dataset(name, dataroot, train_fract=train_fract, do_spellchecker=do_spellchecker, do_head=do_head, do_bert=do_bert)
		print("Loading classes...")
		self.class_labels = load_boxes_classes('data/objects_vocab.txt', word_embedding=wordEmbedding, word_indexer=self.indexer, do_spellchecker=do_spellchecker, do_oov=do_oov)
		# img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
		print("Loading images size...")
		self.images_size = json.load(open(f'{dataroot}/{name}_images_size.json', 'r'))
		print("Loading features...")
		h5_path = os.path.join(dataroot, '%s_features_compress.hdf5' % name)

		with h5py.File(h5_path, 'r') as hf:
			# print(hf.keys()) 	#<KeysViewHDF5 ['bboxes', 'features', 'pos_bboxes']>
			self.features = np.array(hf.get('features'))
			self.pos_boxes = np.array(hf.get('pos_bboxes'))
		print("Dataset loaded.")
			

	def _get_entry(self, index):
		entry = self.entries[index]
		attrs = []
		# attrs = entry['attrs']
		return entry['image'], entry['labels'], entry['query'], entry['head'], attrs, entry['detected_bboxes'], entry['target_bboxes'], entry['bert_query_input_ids'], entry['bert_query_attention_mask'], entry['width'], entry['height']

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

		K = 100
		Q = 32
		lens = 12
		B = 20

		imgid, labels, querys, heads, attrs, bboxes, target_bboxes, bert_query_input_ids, bert_query_attention_mask, width, height = self._get_entry(index)

		idx = self.img_id2idx[int(imgid)]  # to retrieve pos in pos_box
		pos = self.pos_boxes[idx]

		feature = torch.from_numpy(self.features[pos[0]:pos[1]]).float()

		if feature.size(0) < K:
			pad = nn.ZeroPad2d((0, 0, 0, K - feature.size(0)))
			feature = pad(feature)
		else:
			feature = feature[:K]

		num_obj = min(len(labels), K)
		num_query = min(len(querys), Q)

		labels_idx = [0] * K
		labels_idx[:num_obj] = [max(self.indexer.index_of(re.split(' ,', w)[-1]), 1) for w in labels]
		labels_idx = labels_idx[:K]


		attr_idx = [0] * K
		# attr_idx[:num_obj] = [max(self.indexer.index_of(w), 1) for w in attrs]
		# attr_idx = attr_idx[:K]

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
		while(len(bert_query_attention_mask) < Q):
			bert_query_attention_mask.append([0] * lens)

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

		padbox = [0, 0, 0, 0]

		while (len(bboxes) < K):
			bboxes.append(padbox)
		bboxes = bboxes[:K]

		bboxes = torch.tensor(bboxes)
		area = (bboxes[..., 3] - bboxes[..., 1]) * (bboxes[..., 2] - bboxes[..., 0])
		bboxes = torch.cat((bboxes, area.unsqueeze_(-1)), -1)

		spatial_features = bboxes.clone().float()
		spatial_features = spatial_features[..., :4] / torch.tensor([[width, height, width, height]]).float()
		area = (spatial_features[..., 3] - spatial_features[..., 1]) * (spatial_features[..., 2] - spatial_features[..., 0])
		spatial_features = torch.cat((spatial_features, area.unsqueeze_(-1)), -1)

		for bbox in target_bboxes:
			while (len(bbox) < B):
				bbox.append(padbox)
		target_bboxes = [b[:B] for b in target_bboxes]
		padline = [padbox for i in range(B)]
		while (len(target_bboxes) < Q):
			target_bboxes.append(padline)
		target_bboxes = target_bboxes[:Q]

		assert len(labels_idx) == K
		assert len(bboxes) == K
		assert len(querys_idx) == Q
		assert len(heads_idx) == Q
		assert len(target_bboxes) == Q

		return torch.tensor(int(imgid)), torch.tensor(labels_idx), torch.tensor(attr_idx), feature, \
			torch.tensor(querys_idx), bboxes, torch.tensor(target_bboxes), torch.tensor(num_obj), torch.tensor(
			num_query), torch.tensor(heads_idx), torch.tensor(bert_query_input_ids), \
			torch.tensor(bert_query_attention_mask), spatial_features

	def __len__(self):
		return len(self.entries)


def load_train_flickr30k(dataroot, img_id2idx, obj_detection, images_size, do_spellchecker=False, do_head=False, do_bert=False):
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

	pattern_phrase = r'\[(.*?)\]'
	pattern_no = r'\/EN\#(\d+)'
	missing_entity_count = dict()
	entries = []

	for image_id, idx in tqdm(img_id2idx.items()):

		phrase_file = os.path.join(dataroot, 'Flickr30kEntities/Sentences/%d.txt' % image_id)
		anno_file = os.path.join(dataroot, 'Flickr30kEntities/Annotations/%d.xml' % image_id)

		try:
			with open(phrase_file, 'r', encoding = 'utf-8') as f:
				sents = [x.strip() for x in f]
		except:
			logging.warning(f'Cannot open {phrase_file}, skipping...')
			continue

		# Parse Annotation
		root = parse(anno_file).getroot()
		obj_elems = root.findall('./object')

		target_bboxes_dict = {}
		entitywbox = []

		for elem in obj_elems:
			if elem.find('bndbox') == None or len(elem.find('bndbox')) == 0:
				continue
			left = int(elem.findtext('./bndbox/xmin'))
			top = int(elem.findtext('./bndbox/ymin'))
			right = int(elem.findtext('./bndbox/xmax'))
			bottom = int(elem.findtext('./bndbox/ymax'))
			assert 0 < left and 0 < top

			for name in elem.findall('name'):
				entity_id = int(name.text)
				assert 0 < entity_id
				entitywbox.append(entity_id)
				if not entity_id in target_bboxes_dict.keys():
					target_bboxes_dict[entity_id] = []
				target_bboxes_dict[entity_id].append([left, top, right, bottom])
		image_id = str(image_id)
		bboxes = obj_detection[image_id]['bboxes']
		labels = obj_detection[image_id]['classes']  # [B, 4]
		attrs = obj_detection[image_id]['attrs'] if 'attrs' in obj_detection[image_id].keys() else []

		image_size = images_size[image_id]
		image_width = image_size[0]
		image_height = image_size[1]

		assert (len(bboxes) == len(labels))

		# Parse Sentence
		for sent_id, sent in enumerate(sents):
			sentence = utils.remove_annotations(sent)
			entities = re.findall(pattern_phrase, sent)
			entity_ids = []
			entity_types = []
			entity_names = []
			entity_indices = []
			target_bboxes = []
			query = []
			head = []

			for i, entity in enumerate(entities):
				info, phrase = entity.split(' ', 1)
				entity_id = int(re.findall(pattern_no, info)[0])
				entity_type = info.split('/')[2:]
				entity_idx = utils.find_sublist(sentence.split(' '), phrase.split(' '))

				if not entity_id in target_bboxes_dict:
					if entity_id >= 0:
						missing_entity_count[entity_type[0]] = missing_entity_count.get(entity_type[0], 0) + 1
					continue

				assert 0 < entity_id

				# in entity order
				target_bboxes.append(target_bboxes_dict[entity_id])

				# correct phrase
				if do_spellchecker:
					words: List[str] = phrase.split(' ')
					words_corrected = [spell.correction(word) or word for word in words]
					phrase_corrected = ' '.join(words_corrected)
					if phrase != phrase_corrected:
						print(phrase, "->", phrase_corrected)
						phrase = phrase_corrected

				if do_head:
					doc = spacy_nlp(phrase)
					phrase_heads = [chunk.root.text for chunk in doc.noun_chunks]
					if len(phrase_heads) == 0:
						phrase_heads = [doc[-1].text]  # fallback to last word
					phrase_head = ' '.join(phrase_heads)   # we treat multiple heads as a phrase
					head.append(phrase_head)

				query.append(phrase)

				entity_names.append(phrase)
				entity_ids.append(entity_id)
				entity_types.append(entity_type)
				assert len(entity_names) == len(entity_ids)

				entity_indices.append(entity_idx)

			if 0 == len(entity_ids):
				continue

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

			entry = {
				'image': image_id,
				'target_bboxes': target_bboxes,  # in order of entities
				"detected_bboxes": bboxes,  # list, in order of labels
				'labels': labels,
				'attrs': attrs,
				'query': query,
				'head': head,
				'bert_query_input_ids': bert_query_input_ids,
				'bert_query_attention_mask': bert_query_attention_mask,
				'width': image_width,
				'height': image_height,
			}
			entries.append(entry)
	return entries


def load_dataset(name = 'train', dataroot = 'data/flickr30k/', train_fract=1.0, do_spellchecker=False, do_head=False, do_bert=False):
	obj_detection_dict = json.load(open("data/flickr30k/%s_detection_dict.json" % name, "r"))
	# n_objects = 0
	# classes = set()
	# for key, val in obj_detection_dict.items():
	# 	n_objects += len(val['bboxes'])
	# 	classes = classes.union(set(val['classes']))
	# print("Number of objects: ", n_objects)
	# print("Classes: ", len(classes), classes)
	img_id2idx = cPickle.load(open(os.path.join(dataroot, '%s_imgid2idx.pkl' % name), 'rb'))

	images_size = json.load(open(os.path.join(dataroot, '%s_images_size.json' % name), 'r'))

	# subsample dataset accordin to train_fract value only in training set
	if name == 'train' and train_fract < 1.0:
		random.seed(2022)
		n_images = len(img_id2idx)
		n_subset = train_fract * n_images
		subset_idx = random.sample([i for i in img_id2idx.keys()], int(n_subset))
		img_id2idx = {key: img_id2idx[key] for key in subset_idx}

	entries = load_train_flickr30k(dataroot, img_id2idx, obj_detection_dict, images_size, do_spellchecker=do_spellchecker, do_head=do_head, do_bert=do_bert)
	return entries, img_id2idx


# def gen_obj_dict(obj_detection):
# 	"""
# 	generate object detection dictionary
# 	"""
# 	obj_detect_dict = {}
# 	for img in obj_detection:
# 		try:
# 			img_id = int(img["image"].split('.')[0])
# 		except:
# 			continue
# 
# 		tmp = {"bboxes": [], "classes": [], "scores": [], "features": []}
# 		for dic in img['objects']:
# 			bbox = [int(i) for i in dic["bbox"][1:-1].split(',')]
# 			tmp["bboxes"].append(bbox)
# 			tmp["classes"].append(dic["class"])
# 			tmp["scores"].append(dic["score"])
# 			tmp["features"].append(dic["feature"])
# 
# 		obj_detect_dict[img_id] = tmp
# 	return obj_detect_dict

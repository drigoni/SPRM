from __future__ import print_function
from collections import defaultdict
from email.mime import image

import os
from os import listdir
from os import path
from os.path import isfile, join
import argparse
from posixpath import split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import utils
import csv
import pandas as pd 
import base64
import json
from tqdm import tqdm


def save_json(data, output_folder, split_name):
    file_name = '{}_detection_dict.json'.format(split_name)
    file_path = os.path.join(output_folder, file_name)
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print('Data saved: ', file_path)


def load_boxes_classes(file_classes, do_spellchecker=False):
	# read labels
	with open(file_classes, 'r') as f:
		data = f.readlines()
	labels = [label.strip() for label in data] # add background class

	# correct phrase
	if do_spellchecker:
		new_labels = []
		spell = SpellChecker()
		for label in labels:
			label_corrected = spell.correction(label)
			if label_corrected is not None:
				new_labels.append(label_corrected)
			else:
				new_labels.append(label)
	else:
		new_labels = labels
	return new_labels


def load_data(img_folder):
    class_labels = load_boxes_classes('data/objects_vocab.txt')
    attributes_labels = load_boxes_classes('data/attributes_vocab.txt')
    # get all file in the folder
    onlyfiles = [join(img_folder, f) for f in listdir(img_folder) if isfile(join(img_folder, f))]
    print('Number of files: ', len(onlyfiles))
    onlyfiles = [f for f in onlyfiles if f[-4:] == '.npz']
    print('Number of .npz files: ', len(onlyfiles))

    # load all data [x, bbox, num_bbox, image_h, image_w, info]
    # info = {
    # 'image_id': im_file.split('.')[0],
    # 'image_h': np.size(im, 0),
    # 'image_w': np.size(im, 1),
    # 'num_boxes': len(keep_boxes),
    # 'objects_id': image_objects,
    # 'objects_conf': image_objects_conf,
    # 'attrs_id': image_attrs,
    # 'attrs_conf': image_attrs_conf,
    # }
    all_data = dict()
    for img_file in tqdm(onlyfiles):
        img_id = img_file.split('/')[-1][:-4]   # remove ".npz"
        tmp_dict = dict()
        with np.load(img_file, allow_pickle=True) as f:
            data_info = f['info'].item()
            # print("Numero classi: ", len(data_info['cls_prob'][0]))
            best_class_idx = data_info['objects_id']
            # all_data['cls_prob'].append(data_info['cls_prob'])
            # all_data['attr_id'].append(data_info['attrs_id'])
            # all_data['attr_scores'].append(data_info['attrs_scores'])
            best_attribute_idx = data_info['attrs_id']
            # retrieve labels
            best_class_labels = [class_labels[i] for i in best_class_idx]
            best_attribute_labels = [attributes_labels[i] for i in best_attribute_idx]
            # save
            tmp_dict['bboxes'] = f['bbox'].tolist()
            tmp_dict['classes'] = best_class_labels
            tmp_dict['attrs'] = best_attribute_labels
        all_data[img_id] = tmp_dict
    return all_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_features', type=str, default='./data/referit/extracted_features/', help='Folder of extracted features.')
    parser.add_argument('--output_folder', type=str, default='./data/referit/', help='Folder where to save the detection_dict.json file.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.extracted_features):
        print('Loading all data.')
        all_data = load_data(args.extracted_features)

        print("Saving data.")
        splits = ['./data/referit/train.txt',
                    './data/referit/val.txt',
                    './data/referit/test.txt']
        for split_path in splits:
            split_name = split_path.split('/')[-1][:-4]
            with open(split_path, 'r') as f:
                split_idx = f.read().splitlines()
            split_idx = [i.strip() for i in split_idx]  # cleaning
            print('Processing split {} with {} elements. '.format(split_name, len(all_data)))
            save_json(all_data, args.output_folder, split_name)
    else:
        print("Folder not valid: ", args.extracted_features)
        exit(1)

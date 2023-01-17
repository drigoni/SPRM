"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa

Reads in a tsv file with pre-trained bottom up attention features 
of the adaptive number of boxes and stores it in HDF5 format.  
Also store {image_id: feature_idx} as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_boxes x 2048
  'image_bb': num_boxes x 4
  'spatial_features': num_boxes x 6
  'pos_boxes': num_images x 2 }
"""
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
from tqdm import tqdm

def load_data(img_folder):
    # get all file in the folder
    onlyfiles = [join(img_folder, f) for f in listdir(img_folder) if isfile(join(img_folder, f))]
    print('Number of files: ', len(onlyfiles))
    onlyfiles = [f for f in onlyfiles if f[-4:] == '.npz']
    print('Number of .npz files: ', len(onlyfiles))

    # load all data ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    all_data = defaultdict(list)
    for img_file in tqdm(onlyfiles):
        img_id = img_file.split('/')[-1][:-8]   # remove ".jpg.npz"
        with np.load(img_file, allow_pickle=True) as f:
            # get best class in order to filter the __background__ boxes
            best_class_idx = np.argmax(f['cls_prob'], axis=-1)
            selected_boxes = f['bbox'][best_class_idx > 0, :]
            selected_features = f['x'][:, best_class_idx > 0]   # this is transposed [2048, 100]
            n_selected_boxes = sum(best_class_idx > 0)
            # print(n_selected_boxes, f['num_bbox'])
            # all_data['boxes'].append(base64.b64encode(f['bbox']))
            # save data
            # NOTE: be careful with the order. It should match the make_dataset_h5py.py file.
            all_data['image_id'].append(img_id)
            all_data['image_w'].append(f['image_w'])
            all_data['image_h'].append(f['image_h'])
            all_data['num_boxes'].append(n_selected_boxes)
            all_data['boxes'].append(base64.b64encode(np.ascontiguousarray(selected_boxes)))
            all_data['features'].append(base64.b64encode(np.ascontiguousarray(selected_features)))
    return pd.DataFrame(all_data)


def save_tsv(subset_data, output_folder, split_name):
    file_name = '{}_referit_resnet101_faster_rcnn_genome.tsv'.format(split_name)
    file_path = os.path.join(output_folder, file_name)
    # with open(file_path, 'w') as tsvfile:
    #       writer = csv.writer(tsvfile, delimiter='\t', newline='\n')
    #       pd.DataFrame(np_array)
    subset_data.to_csv(file_path, sep="\t",header=False, index=False)
    print('Data saved: ', file_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_features', type=str, default='./data/referit/extracted_features/', help='Folder of extracted features')
    parser.add_argument('--output_folder', type=str, default='./data/referit/', help='Folder where to save the .tsv file.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.extracted_features):
        print('Loading all data.')
        all_data = load_data(args.extracted_features)
        
        # print("Mean number of boxes: ", np.mean(all_data['num_boxes']))
        # print("Max number of boxes: ", max(all_data['num_boxes']))
        # print("Min number of boxes: ", min(all_data['num_boxes']))
        # exit(1)

        print("Saving data.")
        splits = ['./data/referit/train.txt',
                    './data/referit/val.txt',
                    './data/referit/test.txt']
        for split_path in splits:
            split_name = split_path.split('/')[-1][:-4]
            with open(split_path, 'r') as f:
                split_idx = f.read().splitlines()
            split_idx = [i.strip() for i in split_idx]  # cleaning
            subset_data = all_data[all_data['image_id'].isin(split_idx)]
            print('Processing split {} with {}/{} idx found. '.format(split_name, len(subset_data), len(split_idx)))
            save_tsv(subset_data, args.output_folder, split_name)
    else:
        print("Folder not valid: ", args.extracted_features)
        exit(1)

# referit
# Processing split train with 8997/8998 idx found. 
# Data saved:  ./data/referit/train_referit_resnet101_faster_rcnn_genome.tsv
# Processing split val with 1000/1000 idx found. 
# Data saved:  ./data/referit/val_referit_resnet101_faster_rcnn_genome.tsv
# Processing split test with 9998/9999 idx found. 
# Data saved:  ./data/referit/test_referit_resnet101_faster_rcnn_genome.tsv
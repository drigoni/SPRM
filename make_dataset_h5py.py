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

import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import utils
import ast

csv.field_size_limit(sys.maxsize)

def extract(split, infiles, task='vqa'):
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    if task == 'flickr':
        data_file = {
            'train': 'data/flickr30k/train_features_compress.hdf5',
            'val': 'data/flickr30k/val_features_compress.hdf5',
            'test': 'data/flickr30k/test_features_compress.hdf5'}
        indices_file = {
            'train': 'data/flickr30k/train_imgid2idx.pkl',
            'val': 'data/flickr30k/val_imgid2idx.pkl',
            'test': 'data/flickr30k/test_imgid2idx.pkl'}
        ids_file = {
            'train': 'data/flickr30k/train.pkl',
            'val': 'data/flickr30k/val.pkl',
            'test': 'data/flickr30k/test.pkl'}
        path_imgs = {
            'train': 'data/flickr30k/flickr30k_images',
            'val': 'data/flickr30k/flickr30k_images',
            'test': 'data/flickr30k/flickr30k_images'}
        known_num_boxes = {
            'train': None,
            'val': None,
            'test': None,}
    elif task == 'referit':
        data_file = {
            'train': 'data/referit/train_features_compress.hdf5',
            'val': 'data/referit/val_features_compress.hdf5',
            'test': 'data/referit/test_features_compress.hdf5'}
        indices_file = {
            'train': 'data/referit/train_imgid2idx.pkl',
            'val': 'data/referit/val_imgid2idx.pkl',
            'test': 'data/referit/test_imgid2idx.pkl'}
        ids_file = {
            'train': 'data/referit/train.txt',
            'val': 'data/referit/val.txt',
            'test': 'data/referit/test.txt'}
        path_imgs = {
            'train': 'data/referit/flickr30k_images',
            'val': 'data/referit/flickr30k_images',
            'test': 'data/referit/flickr30k_images'}
        known_num_boxes = {
            'train': None,
            'val': None,
            'test': None,}

    feature_length = 2048
    min_fixed_boxes = 10
    max_fixed_boxes = 100

    if os.path.exists(ids_file[split]):
        with open(ids_file[split], 'r') as f:
            imgids = f.readlines()
            imgids = [int(l.strip()) for l in imgids]
    else:
        print("File not exist: ", ids_file[split])

    h = h5py.File(data_file[split], 'w')

    if known_num_boxes[split] is None:
        num_box_list = []
        for infile in infiles:
            print("reading tsv...%s" % infile)
            with open(infile, "r+") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    item['num_boxes'] = int(item['num_boxes'])
                    image_id = int(item['image_id'])
                    if image_id in imgids:
                        num_box_list.append(item['num_boxes'])
        num_boxes = sum(num_box_list)
        print('Number of detected boxes for split {}: {}. '.format(split, num_boxes))
        print("Max number of boxes for split {}: {}".format(split, max(num_box_list)))
        print("Min number of boxes for split {}: {}".format(split, min(num_box_list)))
    else:
        num_boxes = known_num_boxes[split]

    print('num_boxes=%d' % num_boxes)

    img_features = h.create_dataset(
        'image_features', (num_boxes, feature_length), 'f')
    img_bb = h.create_dataset(
        'image_bb', (num_boxes, 4), 'f')
    spatial_img_features = h.create_dataset(
        'spatial_features', (num_boxes, 6), 'f')
    pos_boxes = h.create_dataset(
        'pos_boxes', (len(imgids), 2), dtype='int32')

    counter = 0
    num_boxes = 0
    indices = {}

    for infile in infiles:
        unknown_ids = []
        print("reading tsv...%s" % infile)
        with open(infile, "r+") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['num_boxes'] = int(item['num_boxes'])
                item['boxes'] = ast.literal_eval(item['boxes'])
                item['features'] = ast.literal_eval(item['features'])
                image_id = int(item['image_id'])
                image_w = float(item['image_w'])
                image_h = float(item['image_h'])
                bboxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32)
                bboxes = bboxes.reshape((item['num_boxes'], -1))

                box_width = bboxes[:, 2] - bboxes[:, 0]
                box_height = bboxes[:, 3] - bboxes[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bboxes[:, 0] / image_w
                scaled_y = bboxes[:, 1] / image_h

                box_width = box_width[..., np.newaxis]
                box_height = box_height[..., np.newaxis]
                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                     scaled_y,
                     scaled_x + scaled_width,
                     scaled_y + scaled_height,
                     scaled_width,
                     scaled_height),
                    axis=1)

                if image_id in imgids:
                    imgids.remove(image_id)
                    indices[image_id] = counter
                    pos_boxes[counter,:] = np.array([num_boxes, num_boxes + item['num_boxes']])
                    img_bb[num_boxes:num_boxes+item['num_boxes'], :] = bboxes
                    img_features[num_boxes:num_boxes+item['num_boxes'], :] = np.frombuffer(
                        base64.b64decode(item['features']),
                        dtype=np.float32).reshape((item['num_boxes'], -1))
                    spatial_img_features[num_boxes:num_boxes+item['num_boxes'], :] = spatial_features
                    counter += 1
                    num_boxes += item['num_boxes']
                else:
                    unknown_ids.append(image_id)

        print('%d unknown_ids...' % len(unknown_ids))
        print('%d image_ids left...' % len(imgids))

    if len(imgids) != 0:
        print('Warning: %s_image_ids is not empty' % split)

    cPickle.dump(indices, open(indices_file[split], 'wb'))
    h.close()
    print("done!")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='referit', help='flickr or referit')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.task == 'flickr':
        infiles = ['data/flickr30k/train_flickr30k_resnet101_faster_rcnn_genome.tsv']
        extract('train', infiles, args.task)
        infiles = ['data/flickr30k/val_flickr30k_resnet101_faster_rcnn_genome.tsv']
        extract('val', infiles, args.task)
        infiles = ['data/flickr30k/test_flickr30k_resnet101_faster_rcnn_genome.tsv']
        extract('test', infiles, args.task)
    elif args.task == 'referit':
        infiles = ['data/referit/train_referit_resnet101_faster_rcnn_genome.tsv']
        extract('train', infiles, args.task)
        infiles = ['data/referit/val_referit_resnet101_faster_rcnn_genome.tsv']
        extract('val', infiles, args.task)
        infiles = ['data/referit/test_referit_resnet101_faster_rcnn_genome.tsv']
        extract('test', infiles, args.task)
    else:
        print("Task error: ", args.task)


# referit
# reading tsv...data/referit/train_referit_resnet101_faster_rcnn_genome.tsv
# Number of detected boxes for split train: 899177. 
# Max number of boxes for split train: 100
# Min number of boxes for split train: 74
# num_boxes=899177
# reading tsv...data/referit/train_referit_resnet101_faster_rcnn_genome.tsv
# 0 unknown_ids...
# 1 image_ids left...
# Warning: train_image_ids is not empty
# done!
# reading tsv...data/referit/val_referit_resnet101_faster_rcnn_genome.tsv
# Number of detected boxes for split val: 99995. 
# Max number of boxes for split val: 100
# Min number of boxes for split val: 95
# num_boxes=99995
# reading tsv...data/referit/val_referit_resnet101_faster_rcnn_genome.tsv
# 0 unknown_ids...
# 0 image_ids left...
# done!
# reading tsv...data/referit/test_referit_resnet101_faster_rcnn_genome.tsv
# Number of detected boxes for split test: 999160. 
# Max number of boxes for split test: 100
# Min number of boxes for split test: 66
# num_boxes=999160
# reading tsv...data/referit/test_referit_resnet101_faster_rcnn_genome.tsv
# 0 unknown_ids...
# 1 image_ids left...
# Warning: test_image_ids is not empty
# done!

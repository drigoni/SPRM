import argparse
import logging
import pickle
import re
from typing import List

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

model, preprocess = None, None
device = None

def main():
    global model, preprocess
    global device

    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    dataset = args.dataset
    split = args.split
    data_root = args.data_root
    clip_model = args.clip_model
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device

    logging.info(f"Using device: {device}")

    dataset, data_root = get_dataset(dataset, data_root)
    model, preprocess = clip.load(clip_model, device=device, jit=False)
    model = model.to(device).float()

    img_index_path = f"{data_root}/{split}_imgid2idx.pkl"
    img_id2idx = pickle.load(open(img_index_path, "rb"))

    skipped = []

    queries_embedding = {}

    for image_id, idx in tqdm(img_id2idx.items()):
        try:
            sentences = dataset.get_sentences(image_id)
        except:
            skipped.append(image_id)
            continue

        embedding_list = []

        for sentence in sentences:
            queries = dataset.get_queries(sentence)

            with torch.no_grad():
                text_embedding = get_text_embedding(queries)
                text_embedding = text_embedding.cpu().numpy()

            embedding_list.append(text_embedding)

        queries_embedding[image_id] = embedding_list

    logging.info(f"Skipped {len(skipped)} images over {len(img_id2idx.items())}")
    logging.debug(skipped)

    output_path = f"{data_root}/{split}_queries_embedding.pkl"
    pickle.dump(queries_embedding, open(output_path, "wb"))


def get_text_embedding(text):
    text = clip.tokenize(text).to(device)
    text_embedding = model.encode_text(text).float()

    return text_embedding


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["flickr30k", "referit"])
    parser.add_argument("--split", choices=["train", "test", "val"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--clip-model", default="RN50", choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"])
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])

    return parser.parse_args()


def get_dataset(dataset: str, data_root: str):
    data_root = f"{data_root}/{dataset}"

    if dataset == "flickr30k":
        return Flickr30k(data_root), data_root
    
    if dataset == "referit":
        raise NotImplementedError("Referit dataset is not supported yet")
        #return Referit(data_root), data_root
    
    raise ValueError(f"Dataset '{dataset}' is not supported")


class Flickr30k:
    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_sentences(self, image_id: str) -> List[str]:
        sentence_file = f"{self.data_root}/Flickr30kEntities/Sentences/{image_id}.txt"

        with open(sentence_file, 'r', encoding = 'utf-8') as f:
            sentences = [x.strip() for x in f]

        return sentences
    
    def get_queries(self, sentence) -> List[str]:
        query_pattern = r'\[(.*?)\]'
        queries = []

        entities = re.findall(query_pattern, sentence)

        for entity in entities:
            _, query = entity.split(' ', 1)
            queries.append(query)

        return queries

class Referit:
    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_image_file(self, image_id: str):
        image_id_str = str(image_id)
        image_id_str = image_id_str.zfill(5)

        image_id_part1 = image_id_str[:2]

        return f"{self.data_root}/refer/data/images/saiapr_tc-12/{image_id_part1}/images/{image_id}.jpg"


if __name__ == "__main__":
    main()

import argparse
import pickle
import torch
import clip
import logging
import numpy as np
import json
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

    detection_dict_path = f"{data_root}/{split}_detection_dict.json"
    detection_dict = json.load(open(detection_dict_path, "r"))

    images_size_path = f"{data_root}/{split}_images_size.json"
    images_size = json.load(open(images_size_path, "r"))

    skipped = []

    images_embedding = {}

    for image_id, idx in tqdm(img_id2idx.items()):
        image_file = dataset.get_image_file(image_id)

        try:
            img = load_image(image_file)
        except FileNotFoundError:
            skipped.append(image_file)
            continue

        boxes = detection_dict[str(image_id)]["bboxes"]
        size = images_size[str(image_id)]

        with torch.no_grad():
            try:
                patches = get_patches(img, boxes, size=size)
                image_embedding = get_image_embedding(patches)
            except:
                print(f"Error with image {image_id} -> {image_file}")
                raise
            images_embedding[image_id] = image_embedding.cpu().numpy()

    logging.info(f"Skipped {len(skipped)} images over {len(img_id2idx.items())}")
    logging.debug(skipped)

    output_path = f"{data_root}/{split}_images_embedding.pkl"
    pickle.dump(images_embedding, open(output_path, "wb"))


def get_patches(img: np.array, boxes: np.array, size: np.array) -> torch.Tensor:
    patches = []

    iw, ih = size

    for box in boxes:
        x1, y1, x2, y2 = box
        x, y, w, h = max(x1, 0), max(y1, 0), min(x2 - x1, iw) , min(y2 - y1, ih)

        # workaround for empty boxes (bug in detection)
        # occurs 2 times in training set
        if w <= 0 or h <= 0:
            patch = torch.zeros(1, 3, 224, 224)
        else:
            patch = img[y:y+h, x:x+w]
            patch = to_pil(patch)
            patch = preprocess(patch).unsqueeze(0)

        patches.append(patch)
    
    return torch.cat(patches, dim=0).to(device)
    

def get_image_embedding(patches):
    return model.encode_image(patches).float()


def to_pil(img):
    patch = (img * 255.).astype(np.uint8)
    return Image.fromarray(patch).convert("RGB")


def load_image(img_path, resize: tuple=None, pil=False):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    if pil:
        return image
    image = np.asarray(image).astype(np.float32) / 255. 
    return image


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
        return Referit(data_root), data_root
    
    raise ValueError(f"Dataset '{dataset}' is not supported")


class Flickr30k:
    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_image_file(self, image_id: str):
        return f"{self.data_root}/flickr30k_images/{image_id}.jpg"


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

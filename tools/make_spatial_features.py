import argparse
import json
import logging
import pickle

from PIL import Image
from torchvision.transforms.functional import get_image_size
from tqdm import tqdm

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["flickr30k", "referit"])
    parser.add_argument("--split", choices=["train", "test", "val"])
    parser.add_argument("--data-root", default="data")

    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    data_root = args.data_root

    if not dataset in ["flickr30k", "referit"]:
        raise ValueError(f"Dataset '{dataset}' is not supported")

    data_root = f"{data_root}/{dataset}"

    img_index_path = f"{data_root}/{split}_imgid2idx.pkl"
    img_id2idx = pickle.load(open(img_index_path, "rb"))

    skipped = []
    sizes = {}

    def get_image_file(image_id: str):
        if dataset == "flickr30k":
            return f"{data_root}/flickr30k_images/{image_id}.jpg"
        if dataset == "referit":
            image_id_str = str(image_id)
            image_id_str = image_id_str.zfill(5)

            image_id_part1 = image_id_str[:2]

            return f"{data_root}/refer/data/images/saiapr_tc-12/{image_id_part1}/images/{image_id}.jpg"
        return f"{image_id}.jpg"

    for image_id, idx in tqdm(img_id2idx.items()):
        image_file = get_image_file(image_id)

        try:
            img = Image.open(image_file)
        except FileNotFoundError:
            skipped.append(image_file)
            continue

        sizes[image_id] = get_image_size(img)

    logging.info(f"Skipped {len(skipped)} images over {len(img_id2idx.items())}")
    logging.debug(skipped)

    print(json.dumps(sizes, indent=2))

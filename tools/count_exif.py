from PIL import Image, ExifTags
import pickle
from tqdm import tqdm

def main():
    data_root = "data"
    split = "val"
    dataset = "flickr30k"
    data_path = f"{data_root}/{dataset}"

    img_index_path = f"{data_path}/{split}_imgid2idx.pkl"
    img_id2idx = pickle.load(open(img_index_path, "rb"))

    dataset = Flickr30k(data_path)

    exc = 0
    tot = 0

    for image_id, idx in tqdm(img_id2idx.items()):
        tot += 1

        image_file = dataset.get_image_file(image_id)
        try:
            image=Image.open(image_file)

            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            
            exif = image._getexif()

            if exif[orientation] == 3:
                image=image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image=image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image=image.rotate(90, expand=True)

            #image.save(filepath)
            image.close()
        except (AttributeError, KeyError, IndexError, TypeError):
            # cases: image don't have getexif
            exc += 1

    print(f"Total: {tot}")
    print(f"Exceptions: {exc}")

class Referit:
    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_image_file(self, image_id: str):
        image_id_str = str(image_id)
        image_id_str = image_id_str.zfill(5)

        image_id_part1 = image_id_str[:2]

        return f"{self.data_root}/refer/data/images/saiapr_tc-12/{image_id_part1}/images/{image_id}.jpg"


class Flickr30k:
    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_image_file(self, image_id: str):
        return f"{self.data_root}/flickr30k_images/{image_id}.jpg"

if __name__ == "__main__":
    main()
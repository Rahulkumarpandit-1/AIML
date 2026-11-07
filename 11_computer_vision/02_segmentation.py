import json
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class COCODetectionDataset(Dataset):

    def __init__(self, images_dir, annotation_file):

        self.images_dir = Path(images_dir)

        with open(annotation_file, "r") as file:
            self.coco = json.load(file)


        self.images = self.coco["images"]


        self.annotations = {}

        for annotation in self.coco["annotations"]:
            image_id = annotation["image_id"]

            if image_id not in self.annotations:
                self.annotations[image_id] = []

            self.annotations[image_id].append(annotation)
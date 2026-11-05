from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset        
import matplotlib.pyplot as plt
import matplotlib.patches as patches
DATA_DIR = Path("11_computer_vision/data/PennFudanPed")

image_path = DATA_DIR / "PNGImages" / "FudanPed00001.png"
mask_path = DATA_DIR / "PedMasks" / "FudanPed00001_mask.png"

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path)

mask_array = np.array(mask)

print("Image shape:", np.array(image).shape)
print("Mask shape:", mask_array.shape)
print("Unique mask values:", np.unique(mask_array))

object_ids=np.unique(mask_array)
object_ids=object_ids[object_ids!=0]

boxes=[]
for object_id in object_ids:
    binary_mask=mask_array==object_id   
    y_coordinates, x_coordinates = np.where(binary_mask)
    x_min = x_coordinates.min()
    x_max = x_coordinates.max()

    y_min = y_coordinates.min()
    y_max = y_coordinates.max()
    
    boxes.append([x_min, y_min, x_max, y_max])
    
print("Bounding boxes:")
print(boxes)    

# fig, ax = plt.subplots(figsize=(10, 8))
# ax.imshow(image)

# for i, box in enumerate(boxes):

#     x_min, y_min, x_max, y_max = box

#     width = x_max - x_min
#     height = y_max - y_min

#     rectangle = patches.Rectangle(
#         (x_min, y_min),
#         width,
#         height,
#         linewidth=2,
#         edgecolor="red",
#         facecolor="none"
#     )

#     ax.add_patch(rectangle)

#     ax.text(
#         x_min,
#         y_min - 5,
#         f"Person {i + 1}",
#         color="red",
#         fontsize=12
#     )

# ax.set_title("Detected Pedestrian Bounding Boxes")
# ax.axis("off")

# plt.show()

boxes=torch.tensor(boxes,dtype=torch.float32)
print("Boxes",boxes)
print("Boxes shape",boxes.shape )
labels=torch.ones((len(boxes),),dtype=torch.int64)
# print("labels",labels)
# print("labels shape",labels.shape   )

target={
    "boxes":boxes,
    "labels":labels     
}
print("target:",target)

class PennFudanDataset(Dataset):
    def __init__(self,root):
        self.root=Path(root)
        
        self.image_dir=self.root/"PNGImages"
        self.mask_dir=self.root/"PedMasks"
        
        self.images=sorted(self.image_dir.glob("*.png"))


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        mask_path = self.masks_dir / (
        image_path.stem + "_mask.png"
    )
        # Load the image as RGB
        image = Image.open(image_path).convert("RGB")

# Load the mask without converting it to RGB
        mask = np.array(Image.open(mask_path))
        print("Image:", image_path.name)
        print("Mask:", mask_path.name)
dataset = PennFudanDataset("data/PennFudanPed")

print("Total images:", len(dataset))    



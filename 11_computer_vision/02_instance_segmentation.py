from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


DATA_DIR = Path(__file__).resolve().parent / "data" / "PennFudanPed"\
    
class PennFudanDataset(Dataset):
    
    def __init__(self,root,transform=None):
        self.root=Path(root)
        self.transform=transform
        
        self.images=sorted((self.root/"PNGImages").glob("*png")) 
        self.masks = sorted(
            (self.root / "PedMasks").glob("*_mask.png")
        )

        if len(self.images) != len(self.masks):
            raise ValueError("Image and mask counts do not match")
        
        
    def __len__(self):
        return len(self.images)    
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        mask_array = np.array(mask)
        if mask_array.ndim !=2:
            raise ValueError("Expected 2 dimesional array   ")
        
        object_ids=np.unique(mask_array)
        object_ids=object_ids[object_ids!=0]
        
        boxes=[]
        instance_masks=[]
        for object_id in object_ids:
            binary_mask=mask_array==object_id
            

            y_coordinates, x_coordinates = np.where(binary_mask)

            x_min = x_coordinates.min()
            x_max = x_coordinates.max()
            y_min = y_coordinates.min()
            y_max = y_coordinates.max()
            
            boxes.append([x_min,y_min,x_max,y_max])
            instance_masks.append(binary_mask)
            
        boxes=torch.as_tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1,4)   
        
        labels = torch.ones(
            (len(boxes),),
            dtype=torch.int64
        )

        masks = torch.as_tensor(
            np.array(instance_masks),
            dtype=torch.uint8
        )

        if len(instance_masks) == 0:
            masks = torch.zeros(
                (0, mask_array.shape[0], mask_array.shape[1]),
                dtype=torch.uint8
            )

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target
         
transform = transforms.ToTensor()

dataset = PennFudanDataset(
    root=DATA_DIR,
    transform=transform
)
image, target = dataset[0]

print("Total images:", len(dataset))
print("Image shape:", image.shape)
print("Boxes shape:", target["boxes"].shape)
print("Labels shape:", target["labels"].shape)
print("Masks shape:", target["masks"].shape)
print("Unique mask values:", torch.unique(target["masks"]))         
        
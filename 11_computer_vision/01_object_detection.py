from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset       
from torchvision import transforms 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import matplotlib.patches as patches
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
DATA_DIR = Path("11_computer_vision/data/PennFudanPed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
image_path = DATA_DIR / "PNGImages" / "FudanPed00001.png"
mask_path = DATA_DIR / "PedMasks" / "FudanPed00001_mask.png"

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path)

mask_array = np.array(mask)

class PennFudanDataset(Dataset):
    def __init__(self,root):
        self.root=Path(root)
        
        self.image_dir=self.root/"PNGImages"
        self.masks_dir=self.root/"PedMasks"
        
        self.images=sorted(self.image_dir.glob("*.png"))


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]   
        
        mask_path = self.masks_dir / (
        image_path.stem + "_mask.png"
    )
        # Load the image as RGB
        image = Image.open(image_path).convert("RGB")
        image=transforms.ToTensor()(image)
# Load the mask without converting it to RGB
        mask = np.array(Image.open(mask_path))
        object_ids=np.unique(mask)
        object_ids=object_ids[object_ids!=0]
        
        boxes=[]
        for object_id in object_ids:
            binary_mask=mask==object_id
            
            y_coordinates,x_coordinates=np.where(binary_mask)
            x_min = x_coordinates.min()
            x_max = x_coordinates.max()

            y_min = y_coordinates.min()
            y_max = y_coordinates.max()

            boxes.append([x_min, y_min, x_max, y_max])
            
        boxes=torch.as_tensor(boxes,dtype=torch.float32)    
        labels=torch.ones((len(boxes),),dtype=torch.int64)
        
        target = {
       "boxes": boxes,
       "labels": labels
        }
        return image, target
dataset = PennFudanDataset("11_computer_vision/data/PennFudanPed")
generator=torch.Generator().manual_seed(42)
def collate_fn(batch):
    return tuple(zip(*batch))
train_dataset, test_dataset = random_split(
    dataset,
    [136, 34],
    generator=generator
)

train_Loader=DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

test_Loader=DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn               
)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


images, targets = next(iter(train_Loader))



weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model=fasterrcnn_resnet50_fpn(weights=weights)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    2
)
model = model.to(device)
params=[p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

epochs=5

for epoch in range(epochs):
    model.train()
    total_loss=0
    
    for images,targets in train_Loader:
        images=[image.to(device) for image in images]

        
        targets = [
    {key: value.to(device) for key, value in target.items()}
    for target in targets
]
        loss_dict=model(images,targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item() 
        
    average_loss=total_loss/len(train_Loader)
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Average Loss: {average_loss:.4f}"
    )         
def calculate_iou(box1, box2):

    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )
    

    # Calculate individual box areas
    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    # Calculate union
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0    


image,target=test_dataset[0]    
image_tensor = image.to(device)

model.eval()

true_positives = 0
false_positives = 0
false_negatives = 0

confidence_threshold = 0.5
iou_threshold = 0.5

with torch.no_grad():

    for images, targets in test_Loader:

        images = [image.to(device) for image in images]

        predictions = model(images)

        for prediction, target in zip(predictions, targets):

            pred_boxes = prediction["boxes"].cpu()
            pred_scores = prediction["scores"].cpu()

            gt_boxes = target["boxes"].cpu()

            # Keep confident predictions
            selected = pred_scores >= confidence_threshold
            pred_boxes = pred_boxes[selected]

            matched_gt = set()

            # Match each prediction with a ground-truth box
            for pred_box in pred_boxes:

                best_iou = 0
                best_gt_index = -1

                for gt_index, gt_box in enumerate(gt_boxes):

                    if gt_index in matched_gt:
                        continue

                    iou = calculate_iou(
                        pred_box.tolist(),
                        gt_box.tolist()
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_index = gt_index

                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_index)
                else:
                    false_positives += 1

            false_negatives += len(gt_boxes) - len(matched_gt)


# Calculate metrics
precision = (
    true_positives /
    (true_positives + false_positives)
    if true_positives + false_positives > 0 else 0
)

recall = (
    true_positives /
    (true_positives + false_negatives)
    if true_positives + false_negatives > 0 else 0
)

print("\nTest Results")
print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
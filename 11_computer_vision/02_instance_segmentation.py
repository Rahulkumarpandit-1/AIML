from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights   
)
import  matplotlib as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor                    
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.detection import MeanAveragePrecision

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
     
def collate_fn(batch):
    return tuple(zip(*batch))         
transform = transforms.ToTensor()

dataset = PennFudanDataset(
    root=DATA_DIR,
    transform=transform
)
weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model=maskrcnn_resnet50_fpn(weights=weights)    


in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    2
)
in_channels=model.roi_heads.mask_predictor.conv5_mask.in_channels                    
hidden_layer=256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_channels,
    hidden_layer,
    2
)

train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_Loader=DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
)

test_Loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
# Get one batch
images, targets = next(iter(train_Loader))

# Move images to GPU
images = [image.to(device) for image in images]

# Move targets to GPU
targets = [
    {key: value.to(device) for key, value in target.items()}
    for target in targets
]


params = [
    parameter for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

epochs=5

for epoch in range(epochs):
    model.train()
    running_loss=0.0
    
    for images,targets in train_Loader:
        images=[image.to(device) for image in images]
        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]
        
        optimizer.zero_grad()
        loss_dict=model(images,targets)
        
        loss = sum(
            individual_loss
            for individual_loss in loss_dict.values()
        )

        
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        
        average_loss=running_loss/len(train_Loader)
        
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Training Loss: {average_loss:.4f}"
    )
    

model.eval()

images, targets = next(iter(test_Loader))

image = images[0].to(device)

with torch.no_grad():
    prediction = model([image])[0]

scores=prediction["scores"]
masks=prediction["masks"]

threshold=0.5
confidence_threshold=0.5
keep=scores>=confidence_threshold
filtered_masks=masks[keep]

binary_masks = filtered_masks[:, 0] > threshold

print("Binary masks shape:", binary_masks.shape)


print("Predicted boxes:", prediction["boxes"].shape)
print("Predicted labels:", prediction["labels"])
print("Confidence scores:", prediction["scores"])
print("Predicted masks:", prediction["masks"].shape)

metric=MeanAveragePrecision(
    iou_type="segm"
)
model.eval()

metric = MeanAveragePrecision(
    iou_type="segm"
)

with torch.no_grad():

    for images, targets in test_Loader:

        images = [
            image.to(device)
            for image in images
        ]

        predictions = model(images)

        predictions_for_metric = []
        targets_for_metric = []

        for prediction, target in zip(predictions, targets):

            predicted_masks = (
                prediction["masks"][:, 0] > 0.5
            ).to(torch.uint8)

            predicted_labels = prediction["labels"] - 1

            predictions_for_metric.append({
                "masks": predicted_masks.cpu(),
                "scores": prediction["scores"].cpu(),
                "labels": predicted_labels.cpu()
            })

            actual_labels = target["labels"] - 1

            targets_for_metric.append({
                "masks": target["masks"].cpu(),
                "labels": actual_labels.cpu()
            })

        metric.update(
            predictions_for_metric,
            targets_for_metric
        )

results = metric.compute()

print("\nMask R-CNN Segmentation Results")

for key, value in results.items():
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        print(f"{key}: {value.item():.4f}")






# image_np = image.cpu().permute(1, 2, 0).numpy()

# plt.figure(figsize=(10, 6))
# plt.imshow(image_np)

# for index, mask in enumerate(binary_masks):
#     mask_np = mask.cpu().numpy()

#     plt.imshow(
#         np.ma.masked_where(~mask_np, mask_np),
#         alpha=0.5
#     )

#     x1, y1, x2, y2 = prediction["boxes"][index].cpu().numpy()

#     plt.gca().add_patch(
#         plt.Rectangle(
#             (x1, y1),
#             x2 - x1,
#             y2 - y1,
#             fill=False,
#             edgecolor="red",
#             linewidth=2
#         )
#     )

#     score = scores[index].item()

#     plt.text(
#         x1,
#         y1,
#         f"Person: {score:.2f}",
#         color="white",
#         backgroundcolor="red"
#     )

# plt.title("Mask R-CNN Instance Segmentation")
# plt.axis("off")
# plt.show()
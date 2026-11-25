
import torch
from pathlib import Path
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ViT_B_16_Weights.DEFAULT

model = vit_b_16(weights=weights)

model = model.to(device)

model.eval()

script_dir = Path(__file__).resolve().parent
image = Image.open(script_dir / "test_image1.png").convert("RGB")
transform=weights.transforms()
image_tensor=transform(image)

image_tensor=image_tensor.unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    outputs = model(image_tensor)

print("Output shape:", outputs.shape)


probabilities=torch.softmax(outputs,1)

top_probabilities,top_indices=torch.topk(probabilities[0],
                                         k=5)

categories=weights.meta["categories"]

for probability,index in zip(top_probabilities,top_indices):
    class_name=categories[index.item()]
    
    print(f"{class_name}:{probability.item()*100:.2f}")


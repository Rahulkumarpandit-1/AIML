from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms

# 1. Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. Recreate the same CNN architecture
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 10)
).to(device)

# 3. Find and load the saved weights
model_path = Path(__file__).resolve().parent / "cnn_cifar10.pth"

model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)

model.eval()

print("Saved model loaded successfully!")

# 4. Prepare test images exactly as during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        shuffle=False   
)


correct = 0
total = 0

# 7. Make prediction without calculating gradients
with torch.no_grad():
    for images,labels in test_loader:
        
        images=images.to(device)
        labels=labels.to(device)
        logits=model(images)
        
        predictions=logits.argmax(dim=1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
   
accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
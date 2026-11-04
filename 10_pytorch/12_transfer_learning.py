import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


weights=models.ResNet18_Weights.DEFAULT

model=models.resnet18(weights=weights)

model=model.to(device)

num_feature=model.fc.in_features
model.fc=nn.Linear(num_feature,10).to(device)

for param in model.parameters():
    param.requires_grad=False           
    
for param in model.fc.parameters():                     
    param.requires_grad=True        
    

for param in model.layer4.parameters():
    param.requires_grad = True   
        
transform=weights.transforms()
train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=128,
    shuffle=False,
    num_workers=0
)
print("Classes:", train_data.classes)
print("Class mapping:", train_data.class_to_idx)



loss_fn=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4}
])

epochs=5
for epoch in range(epochs):
    model.train()


    for module in model.modules():
     if isinstance(module, nn.BatchNorm2d):
        module.eval()
    
    
    total_loss=0
    correct=0
    total=0
    
    for images,labels in train_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_fn(logits, labels)

        loss.backward()

        optimizer.step()

        predictions = logits.argmax(dim=1)
        
        correct+=(predictions==labels).sum().item()
        total+=labels.size(0)
        total_loss += loss.item() * labels.size(0)
        
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {total_loss / total:.4f} | "
        f"Accuracy: {100 * correct / total:.2f}%"
    )
    
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")    
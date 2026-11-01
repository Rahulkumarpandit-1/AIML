import torch 
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from pathlib import Path


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

train_transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data=datasets.CIFAR10(
    root="./data",  
    train=True,
    download=True,
    transform=train_transform        
)
test_data=datasets.CIFAR10(
    root="./data",  
    train=False,
    download=True,
    transform=test_transform     
)

train_loader=DataLoader(
    train_data,batch_size=64,shuffle=True,num_workers=0  
)
  #num_worker=loads data in the main process, which is a simple choice for learning on Windows.

test_loader=DataLoader(
    test_data,batch_size=64,shuffle=False,num_workers=0
)                                           

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Dropout(0.1),
    nn.Linear(32 * 8 * 8, 10)
).to(device)

loss_fn=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)
epochs=15
for epoch in range(epochs):
    model.train()
    total_loss=0
    correct=0
    total=0
    
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()   
        
        logits=model(images)
        loss=loss_fn(logits,labels)
        
        loss.backward()
         
        optimizer.step()
        
      
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    scheduler.step()    

    print(
          f"Epoch {epoch + 1}/{epochs} | "
          f"Loss: {total_loss / total:.4f} | "
          f"Train accuracy: {100 * correct / total:.2f}% | "
          f"Learning rate: {scheduler.get_last_lr()[0]:.6f}"
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

print(f"Test accuracy: {100 * correct / total:.2f}%")    
model_dir= Path(__file__).resolve().parent / "cnn_cifar10.pth"
torch.save(
    model.state_dict(),
    model_dir   
)

print("Model saved successfully!")
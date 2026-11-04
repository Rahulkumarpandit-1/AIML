import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
transformer=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_data=datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transformer
)
test_data=datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transformer
)

train_Loader=DataLoader(
    train_data,batch_size=64,shuffle=True
)
test_Loader=DataLoader(
    test_data,batch_size=64,shuffle=False
)
model=nn.Sequential(
    nn.Conv2d(3,16,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    nn.Conv2d(16,32,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
 
    nn.Flatten(),
    nn.Linear(2048,10)
).to(device)    
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)             
print("Training size",len(train_data))
print("Testing size:", len(test_data))

images,labels=next(iter(train_Loader))

for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print(name, parameter.grad.shape)
        break

        
model.train()    
correct=0
total=0
total_loss=0        
epochs = 5

for epoch in range(epochs):
    correct = 0
    total = 0
    total_loss = 0

    model.train()

    for images, labels in train_Loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, labels)

        predictions = logits.argmax(dim=1)

        loss.backward()
        optimizer.step()

        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss/total:.4f} | "
        f"Accuracy: {100*correct/total:.2f}%"
    )
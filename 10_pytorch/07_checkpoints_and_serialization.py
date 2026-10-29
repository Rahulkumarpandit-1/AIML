import torch
from torch import nn
from pathlib import Path


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device',device)

X=torch.linspace(-1,1,100).reshape(-1,1).to(device) 
y=2*X+1

model=nn.Linear(1,1).to(device)     

loss_fn=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)          

for epoch in range(100):
    model.train()
    
    optimizer.zero_grad()
    prediction=model(X)
    loss=loss_fn(prediction,y)
    loss.backward()
    optimizer.step()
    
if(epoch+1)%10==0:
       print(f'Epoch {epoch+1}/{100}, Loss: {loss.item():.4f}')     
        
        
checkpoint={
    "epoch":epoch,
    "model_state_dict":model.state_dict(),
    "optimizer_state_dict":optimizer.state_dict()
}        
CHECKPOINT_PATH = Path(__file__).resolve().parent / "training_checkpoint.pth"

torch.save(checkpoint, CHECKPOINT_PATH)

print("Checkpoint saved at:", CHECKPOINT_PATH)
print("Checkpoint saved successfully!")        
print("\nTrained weight:", model.weight.item())
print("Trained bias:", model.bias.item())    

loaded_model=nn.Linear(1,1).to(device)      
loaded_optimizer=torch.optim.Adam(
    loaded_model.parameters(),
    lr=0.01 
)       

checkpoint=torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  
start_epoch = checkpoint["epoch"] + 1

print("Checkpoint loaded successfully!")
print("Resume from epoch:", start_epoch)
print("Loaded weight:", loaded_model.weight.item())
print("Loaded bias:", loaded_model.bias.item())
# Resume training from the loaded checkpoint
for epoch in range(start_epoch, start_epoch + 50):
    loaded_model.train()
    loaded_optimizer.zero_grad()

    predictions = loaded_model(X)
    loss = loss_fn(predictions, y)

    loss.backward()
    loaded_optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

print("\nResumed weight:", loaded_model.weight.item())
print("Resumed bias:", loaded_model.bias.item())
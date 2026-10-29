import torch 
import torch.nn as nn

torch.manual_seed(42)
X = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * X + 1
lr=0.01

weight_decay=0.001

def train_model(optimizer_name):
    model=nn.Linear(1,1)
    loss_fn=nn.MSELoss()    
    
    if optimizer_name== 'Adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
        
    else:
        optimizer=torch.optim.SGD(
            model.parameters()
            ,lr=0.01,
            weight_decay=0.01
            )       
        
        
    scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,
    gamma=0.5
)
    
    for epoch in range(500):
      model.train()
      
      optimizer.zero_grad()
      prediction=model(X)
      loss=loss_fn(prediction,y)
      
      loss.backward()   
      optimizer.step()
      scheduler.step()
      
      
    if (epoch + 1) % 20 == 0:
            print(
                f"{optimizer_name} | Epoch {epoch + 1} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.5f}"
            )
            
    return model

adam_model=train_model('Adam')
adamw_model=train_model('AdamW')

print("\nAdam weight:", adam_model.weight.item())
print("Adam bias:", adam_model.bias.item())
print("AdamW weight:", adamw_model.weight.item())
print("AdamW bias:", adamw_model.bias.item())
    
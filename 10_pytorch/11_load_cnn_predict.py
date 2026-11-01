from pathlib import Path
from sklearn.metrics import classification_report       
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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
    nn.Dropout(0.1),
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

# 5. CIFAR-10 class names
classes = test_data.classes

# 6. Select one image
image, true_label = test_data[0]

# Add batch dimension: [3, 32, 32] -> [1, 3, 32, 32]
image = image.unsqueeze(0).to(device)


# Store all actual and predicted labels
all_predictions = []
all_labels = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        # Move predictions and labels to CPU
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

# fig, ax = plt.subplots(figsize=(10, 10))
# disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
# plt.title("CIFAR-10 Confusion Matrix")
# plt.tight_layout()
# plt.show()# Calculate accuracy for each class

class_correct = cm.diagonal()
class_total = cm.sum(axis=1)

class_accuracy = class_correct / class_total * 100

for i, class_name in enumerate(classes):
    print(
        f"{class_name}: "
        f"{class_accuracy[i]:.2f}%"
    )
    
    
print("\nClassification report:")
print(classification_report(all_labels, all_predictions,target_names=classes))   
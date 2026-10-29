
import torch
from torch.utils.data import Dataset, DataLoader


class StudyDataset(Dataset):
    def __init__(self):
        self.hours = [1, 2, 3, 4, 5]
        self.scores = [35, 42, 55, 68, 80]

    def __len__(self):
        return len(self.hours)

    def __getitem__(self, index):
        x = torch.tensor([self.hours[index]], dtype=torch.float32)
        y = torch.tensor([self.scores[index]], dtype=torch.float32)
        return x, y


dataset = StudyDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

for batch_number, (inputs, targets) in enumerate(loader, start=1):
    print(f"Batch {batch_number}")
    print("Inputs:", inputs.flatten().tolist())
    print("Targets:", targets.flatten().tolist())
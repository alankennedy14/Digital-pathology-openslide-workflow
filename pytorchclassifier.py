import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

# Dataset folder
data_dir = r"C:\Users\alan\Desktop\example_output"

# Image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(
    data_dir,
    transform=transform
)

# Data loader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# Simple neural network
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 128 * 3, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Loss + optimiser
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CPU/GPU selection
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print("Using device:", device)
print("Classes:", dataset.classes)

# Training loop
for epoch in range(5):

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
import torch
from torchvision.datasets import MNIST

# Download the dataset
train_dataset = MNIST(root="./data", train=True, download=True)
test_dataset = MNIST(root="./data", train=False, download=True)

# Convert images and labels to tensors
train_images = train_dataset.data          # (60000, 28, 28), uint8
train_labels = train_dataset.targets       # (60000,)

test_images = test_dataset.data            # (10000, 28, 28), uint8
test_labels = test_dataset.targets         # (10000,)

# Save everything to a single .pt file
torch.save(
    {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    },
    "mnist.pt",
)

print("Saved MNIST to mnist.pt")

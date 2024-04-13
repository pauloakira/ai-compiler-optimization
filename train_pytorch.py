# Python libs
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

# Custom libs
import mnist_repository

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

        


if __name__ == "__main__":
    try:
        train_images, train_labels, test_images, test_labels = mnist_repository.load_and_proc_mnist()
    except:
        mnist_repository.download_and_save_mnist(mnist_repository.filename)
        train_images, train_labels, test_images, test_labels = mnist_repository.load_and_proc_mnist()

    train_images = torch.tensor(train_images)
    train_labels = torch.tensor(train_labels)
    test_images = torch.tensor(test_images)
    test_labels = torch.tensor(test_labels)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
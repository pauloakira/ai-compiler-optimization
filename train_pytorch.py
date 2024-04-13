# Python libs
import torch
import numpy as np
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
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
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
    print(f"debug :: len(train_images) = {len(train_images)}")
    train_images = torch.tensor(np.array(train_images))
    train_labels = torch.tensor(np.array(train_labels), dtype=torch.long)
    test_images = torch.tensor(np.array(test_images))
    test_labels = torch.tensor(np.array(test_labels), dtype=torch.long)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Hyperparameters
    num_epochs = 10

    # Initialize model, loss and optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Train the model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}, Loss: {loss.item()}")

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}%")

    # Save model weights to NPZ
    params_np = {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}
    np.savez("assets/model_pytorch.npz", **params_np)
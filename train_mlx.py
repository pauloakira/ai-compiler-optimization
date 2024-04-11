# Python libs
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

# Custom libs
import mnist_repository

class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)  # Flatten the input
        for l in self.layers[:-1]:
            x = nn.relu(l(x))
        return self.layers[-1](x)
    
def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")

def batch_it(batch_size: int, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


if __name__ == "__main__":
    try:
        train_images, train_labels, test_images, test_labels = mnist_repository.load_and_proc_mnist()
    except:
        mnist_repository.download_and_save_mnist(mnist_repository.filename)
        train_images, train_labels, test_images, test_labels = mnist_repository.load_and_proc_mnist()

    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())
    
    # Get a function which gives the loss and gradient of the
    # loss with respect to the model's trainable parameters
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Instantiate the optimizer
    optimizer = optim.SGD(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        for X_batch, y_batch in batch_it(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X_batch, y_batch)
            optimizer.update(model, grads)
            # Force a graph evaluation
            mx.eval(model.parameters(), optimizer.state)
        print(f"Epoch {epoch} loss: {loss}")
    
        accuracy = eval_fn(model, test_images, test_labels)
        print(f"Epoch {epoch}: Test accuracy {accuracy.item():.3f}")

    print("Training complete. Saving model...")
    model.save_weights("assets/model.npz")
    print("Model saved.")
    
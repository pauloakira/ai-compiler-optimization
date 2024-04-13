# # Python libs
import tvm
import torch
import torchvision
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor

# Custom libs
from train_pytorch import MLP, mnistLoaders


# Assuming you have already defined MLP and mnistLoaders elsewhere
if __name__ == "__main__":
    # Load the MNIST dataset
    _, test_loader = mnistLoaders()

    # Load the model's state dict from a .pth file
    model = MLP()  # Initialize the model architecture
    model.load_state_dict(torch.load("assets/model_pytorch.pth"))  # Load weights
    model.eval()  # Set the model to evaluation mode

    # Prepare a dummy input for the conversion
    dummy_input = torch.randn(16, 1, 28, 28)
    # Use torch.jit.trace to convert the model to TorchScript
    scripted_model = torch.jit.trace(model, dummy_input)

    # Convert the PyTorch model to TVM Relay format
    shape_dict = [("x", dummy_input.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict, default_dtype="float32")

    # Set a target and compile the model
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)

    # Inference with TVM
    ctx = tvm.cpu(0)    
    module = graph_executor.GraphModule(lib["default"](ctx))
    
    correct = 0
    total = 0

    for images, labels in test_loader:
        # Convert PyTorch tensor to numpy
        images_np = images.numpy()
        batch_size = images_np.shape[0]
        images_np = images.reshape(batch_size, 1, 28, 28).numpy()  # Ensure correct shape

        # Set the input tensor and execute the model 
        module.set_input("x", tvm.nd.array(images_np))
        module.run()

        # Get the output and post-process
        tvm_output = module.get_output(0).asnumpy()
        predicted_labels = np.argmax(tvm_output, axis=1)

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted_labels == labels.numpy()).sum()

    print(f"Accuracy: {100 * correct / total}%")




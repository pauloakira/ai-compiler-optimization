# Python libs
import tvm
import torch
import torchvision
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor

# Custom libs
from train_pytorch import MLP, mnistLoaders

if __name__ == "__main__":
    # Load the MNIST dataset
    _, test_loader = mnistLoaders()

    # Load the npz file with the weights
    weight_dict = np.load("assets/model_pytorch.npz")
    # Convert numpy arrays to TVM NDArray
    params = {k: tvm.nd.array(v) for k, v in weight_dict.items()}

    # Prepare a dummy input for the conversion
    dummy_input = torch.randn(256,1,28,28)
    # Trace the model architecture using an untrained model
    shape_dict = [("x", dummy_input.shape)]
    print(shape_dict)
    model = MLP()
    # model.eval()
    # Use torch.jit.trace to convert the model to TorchScript
    scripted_model = torch.jit.trace(model, dummy_input)
    print(scripted_model(dummy_input))
    print(scripted_model.graph)
    mod, _ = relay.frontend.from_pytorch(scripted_model, shape_dict,  default_dtype="float32")

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
        print(f"Input shape: {images_np.shape[0]}")
        batch_size = images_np.shape[0]
        images_np = images.numpy().reshape(batch_size, 1, 28, 28)
        if batch_size == 16:
            break

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



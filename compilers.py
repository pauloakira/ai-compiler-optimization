import tvm
import torch
import torchvision
from tvm import relay

if __name__ == "__main__":
    # Example model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    # Dummy input in the shape the model expects
    dummy_input = torch.randn(1, 3, 224, 224)

    # Convert the PyTorch model to Relay
    scripted_model = torch.jit.trace(model, dummy_input).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, [("input", dummy_input.shape)])
    print(f"debug :: mod: {mod}")
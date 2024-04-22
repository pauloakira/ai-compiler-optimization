import torch

if __name__ == "__main__":
    '''
    PyTorch uses the new Metal Performance Shaders (MPS) backend for GPU training acceleration. 
    This MPS backend extends the PyTorch framework, providing scripts and capabilities to set up and 
    run operations on Mac. The MPS framework optimizes compute performance with kernels that are 
    fine-tuned for the unique characteristics of each Metal GPU family. The new mps device maps machine 
    learning computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS.
    '''
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS is not available on this device.")
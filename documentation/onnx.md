# Open Neural Network Exchange (ONNX)

ONNX, which stands for Open Neural Network Exchange, is an open-source format created to represent machine learning models. It was developed to facilitate model interoperability across different software frameworks, allowing models to be easily shared between platforms without locking users into a specific tool or library. It aims at providing a common language any machine learning framework can use to describe its models. The first scenario is to make it easier to deploy a machine learning model in production. An ONNX interpreter (or runtime) can be specifically implemented and optimized for this task in the environment where it is deployed. With ONNX, it is possible to build a unique process to deploy a model in production and independent from the learning framework used to build the model.

The deployment of a machine-learned model into production usually requires replicating the entire ecosystem used to train the model, most of the time with a docker. Once a model is converted into ONNX, the production environment only needs a runtime to execute the graph defined with ONNX operators.

## Using ONNX with TVM
Combining ONNX with TVM can further enhance performance, especially on diverse hardware platforms where TVM provides specific optimizations:

- **Cross-Platform Deployments:** TVM allows you to compile ONNX models to be run on a wide range of devices from mobile phones to servers. TVM performs additional graph-level optimizations and compiles the computational graph to machine-level code specific to your deployment hardware.

- **Auto-Tuning:** TVM’s auto-tuning feature can optimize model performance further by finding the best model configurations and tuning the kernels specifically for your model and hardware. This can lead to superior performance compared to running ONNX models directly through ONNX Runtime.

- **Flexible Backend Support:** While ONNX provides a pathway for standardizing model formats, TVM provides the flexibility in backend execution, allowing selection between various options optimized for different hardware, ensuring the best possible performance.
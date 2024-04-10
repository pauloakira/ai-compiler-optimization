# TVM

TVM is an open-source machine learning (ML) **compiler framework** that aims to enable efficient deployment of deep learning models on various hardware platforms. It addresses a critical challenge in the field of deep learning: the diversity of hardware accelerators and the need for optimization to achieve high performance across different devices. TVM acts as both a compiler and an optimizer for deep learning models. Its primary role is to bridge the gap between the diverse world of deep learning frameworks and the wide array of hardware platforms available for deploying AI models.

- **Hardware Agnostic**: TVM can target a broad range of hardware platforms, including CPUs, GPUs, TPUs, and specialized accelerators. It abstracts away the complexity of optimizing for different hardware, making it easier to deploy models across diverse devices.

- **Framework Agnostic**: It provides compatibility with models from various deep learning frameworks, such as TensorFlow, PyTorch, MXNet, and Keras, by converting these models into its optimized intermediate representation (IR).

- **Optimization**:  TVM uses advanced optimization techniques, including operator fusion, automatic layout transformation, and efficient memory management, to enhance the performance of deep learning models. These optimizations are performed at both the graph level and the operator level.

- **AutoTVM and AutoSchedule**: These are automated optimization tools within TVM that use machine learning to find the most efficient way to execute models on a given target hardware. They dramatically reduce the manual effort required for optimizing model performance.

- **Cross-Platform Deployment**: With TVM, developers can compile and optimize models on a development machine and then deploy them on different target devices, significantly simplifying the deployment process.

The TVM is primarily developed in C++ for its core components, ensuring efficient execution and flexibility in handling low-level operations. Alongside C++, TVM also heavily utilizes Python for its front-end interface, making it accessible and user-friendly for the machine learning community. Python is used for defining models, setting up compilation and optimization processes, and interfacing with different deep learning frameworks. The combination of C++ and Python leverages the strengths of both languages: C++ provides performance and efficiency critical for compiler operations and execution on various hardware, while Python offers ease of use, readability, and the ability to quickly integrate with other machine learning tools and libraries.

![TVM Schema](../assets/tvm_overview.png)

# How TVM uses LLVM?

Once TVM has performed high-level optimizations on the deep learning model's computation graph and transformed it into a low-level intermediate representation (IR), it uses LLVM to generate efficient machine code for the target hardware platform. The main benefits of using the LLVM are:

- **Portability**: LLVM's wide support for different hardware architectures enhances TVM's goal of enabling machine learning models to run on diverse platforms. This support simplifies the process of deploying AI applications across various devices and systems.

- **Performance**: LLVM's optimizations are crucial for achieving high performance. They ensure that the generated machine code is not only correct but also as efficient as possible, taking advantage of specific features of the target hardware.

TVM's integration with LLVM is a key factor in its ability to compile and optimize deep learning models for a broad range of hardware platforms efficiently. This collaboration allows TVM to focus on higher-level optimizations specific to machine learning workloads while relying on LLVM for general-purpose code generation and optimization capabilities.


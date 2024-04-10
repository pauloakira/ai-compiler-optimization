# ai-compiler-optimization
Testing different optimize compilers for linear algebra/ML.

## Installing TVM on MacOS

1. Create a conda environment:
```
conda create --name ml-env python=3.8
```

2. Using Homebrew, install the following packages:
```
brew install gcc git cmake
brew install llvm
```

3. By default brew will not link your llvm installation to correct path. TO fix this:
```
brew link llvm --force
```

4. Clone the TVM repository:
```
git clone --recursive https://github.com/apache/tvm tvm
```
5. Build:
```
cd tvm
mkdir build
cp cmake/config.cmake build
```

6. Edit build/config.cmake file to customize the compilation options, so we can directly add `set(USE_LLVM ON)` and let cmake search for a usable version of LLVM. After that, build and make TVM:
```
cd build
cmake ..
make -j4
```

7. Install required Python packages:
```
pip install -r requirements.txt
```

8. Install TVM python package:
```
export MACOSX_DEPLOYMENT_TARGET=10.9 
cd python
python setup.py install --user
```


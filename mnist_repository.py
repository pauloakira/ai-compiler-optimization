# Python libs
import os
import gzip
import pickle
import numpy as np
from urllib import request

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_and_save_mnist(filename):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    save_path = "data"
    mnist_data = {}

    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Download part
    for name in filename:
        print("Downloading " + name[1] + "...")
        file_path = os.path.join(save_path, name[1])
        request.urlretrieve(base_url + name[1], file_path)

        # Load and save the data into a dictionary as arrays
        if 'images' in name[0]:
            # For image files
            with gzip.open(file_path, 'rb') as f:
                mnist_data[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        else:
            # For label files
            with gzip.open(file_path, 'rb') as f:
                mnist_data[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    # Save the numpy arrays into a single .pkl file
    with open(os.path.join(save_path, "mnist.pkl"), 'wb') as f:
        pickle.dump(mnist_data, f)

    print("Download and save complete.")

def process_mnist(x: np.array)->np.array:
    x = x.astype(np.float32)
    x /= 255.0
    return x

def load_and_proc_mnist():
    with open("data/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    # Normalize the images
    mnist["training_images"] = process_mnist(mnist["training_images"])
    mnist["test_images"] = process_mnist(mnist["test_images"])
    # Typecast the labels
    mnist["training_labels"] = mnist["training_labels"].astype(np.uint32)
    mnist["test_labels"] = mnist["test_labels"].astype(np.uint32)

    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == "__main__":
    download_and_save_mnist(filename)
    mnist = load_and_proc_mnist()
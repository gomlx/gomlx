# MNIST

MNIST is a simple computer vision dataset that consists of images of handwritten digits.

Some examples:

![MNIST digits sample](https://github.com/user-attachments/assets/996c11e0-47f9-4b21-8e23-3867b8942e64)

It also includes labels for each image, which we use to train our example models.

## The `mnist` library

This package includes the following functionality:

  - Download the dataset from [storage.googleapis.com/cvdf-datasets/mnist](https://storage.googleapis.com/cvdf-datasets/mnist),
  - Create a `Dataset` object to iterate over it, use for training and evaluation.
  - A linear and a CNN model demo.
  - A command-line demo (in the `demo` sub-directory).

## Reference

* [Tensorflow "Deep MNIST for Experts"](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/tutorials/mnist/pros/index.md)
* [Exploring Different Loss Functions on MNIST (Kaggle)](https://www.kaggle.com/code/bkhmsi/exploring-different-loss-functions-on-mnist)
* [MNIST in Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)

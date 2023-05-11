## Oxford Flowers 102 Dataset

https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

102 category dataset, consisting of 102 flower categories. The flowers chosen to be flower commonly
occuring in the United Kingdom. Each class consists of between 40 and 258 images. The images have 
large scale, pose and light variations. In addition, there are categories that have large variations
within the category and several very similar categories.

The dataset is divided into a training set, a validation set and a test set. The training set and
validation set each consist of 10 images per class (totalling 1020 images each). The test set 
consists of the remaining 6149 images (minimum 20 per class). Total download in ~330Mb.

More information in the TensorFlow Datasets page:

https://www.tensorflow.org/datasets/catalog/oxford_flowers102


This package provides a `train.Dataset` with the images. 


Under it you will also find a `diffusion`
demo model trains a diffusion model, following the Keras example in:

https://keras.io/examples/generative/ddim/


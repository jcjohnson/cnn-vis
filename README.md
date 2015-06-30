# cnn-vis

Inspired by Google's recent [Inceptionism](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html) blog post, cnn-vis is an open-source tool that lets you use convolutional neural networks to generate images. Here's an example:

<img src="https://github.com/jcjohnson/cnn-vis/blob/master/examples/example12.png?raw=true" width=800px>

CNNs have become very popular in recent years for many tasks in computer vision, but most especially for image classification. A CNN takes an image (in the form of a pixel grid) as input, and transforms the image through several **layers** of nonlinear functions. In a classification setup, the final layer encodes the contents of the image in the form of a probability distribution over a set of classes. The lower layers tend to capture low-level image features such as oriented edges or corners, while the higher layers are thought to encode more semantically meaningful features such as object parts.

In order to use a CNN for a classification task, it needs to be **trained**. We initialize the weights of the network randomly, then show it many examples of images whose labels are known. Based on the errors that the network makes in classifying these known images, we gradually adjust the weights of the network so that it correctly classifies these images. Two popular datasets for training CNNs are ImageNet [4] and MIT Places [9]. ImageNet contains 1000 categories of objects, such as dogs, birds, and other animals, while MIT Places contains 205 types of scenes such as bedrooms, kitchens, and forests.

Although CNNs perform well on a variety of tasks, it can be difficult to understand exactly what types of image features a CNN is using to work its magic. One trick for demystifying a CNN is to choose a neuron in a trained CNN, and attempt to generate an image that causes the neuron to activate strongly. We initialize the image with random noise, propagate the image forward through the network to compute the activation of the target neuron, then propagate the activation of the neuron backward through the network to compute an update direction for the image. We use this information to update the image, and repeat the process until convergence. This general strategy has been used to visualize the activations of individual neurons [7, 8], to generate images of particular object classes [5], to invert CNN features [1, 2], and to generate images to fool CNNs [3, 6].

# Setup
## Caffe
You will need to install [Caffe](http://caffe.berkeleyvision.org/), an excellent open-source CNN implementation from Berkeley. Follow the [installation instructions](http://caffe.berkeleyvision.org/installation.html) and make sure to install the Python bindings.

# Usage
cnn-vis is a standalone Python script; you can control its behavior by passing various command-line arguments.

# References
[1] A. Dosovitskiy and T. Brox. "Inverting Convolutional Networks with Convolutional Networks", arXiv preprint arXiv:1506.02753 (2015).

[2] A. Mahendran and A. Vedaldi, "Understanding Deep Image Representations by Inverting Them", CVPR 2015

[3] A. Nguyen, J. Yosinski, J. Clune. "Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images", CVPR 2015

[4] O. Russakovsky, et al. "Imagenet large scale visual recognition challenge", IJCV 2014.

[5] K. Simonyan and A. Vedaldi and A. Zisserman, "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR 2014

[6] C. Szegedy, et al. "Intriguing properties of neural networks." arXiv preprint arXiv:1312.6199 (2013).

[7] J. Yosinski, J. Clune, A. Nguyen, T. Fuchs, H. Lipson H (2015) "Understanding Neural Networks Through Deep Visualization", ICML 2015 Deep Learning workshop.

[8] M. D. Zeiler and R. Fergus. "Visualizing and understanding convolutional networks", ECCV 2014.

[9] B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. "Learning Deep Features for Scene Recognition using Places Database", NIPS 2014.

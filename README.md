# cnn-vis

Inspired by Google's recent [Inceptionism](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html) blog post, cnn-vis is an open-source tool that lets you use convolutional neural networks to generate images. Here's an example:

<img src="https://github.com/jcjohnson/cnn-vis/blob/master/examples/example12.png?raw=true" width=800px>

Convolutional neural networks (CNNs) have become very popular in recent years for many tasks in computer vision, but most especially for image classification. A CNN takes an image (in the form of a pixel grid) as input, and transforms the image through several **layers** of nonlinear functions. In a classification setup, the final layer encodes the contents of the image in the form of a probability distribution over a set of classes. The lower layers tend to capture low-level image features such as oriented edges or corners, while the higher layers are thought to encode more semantically meaningful features such as object parts.

In order to use a CNN for a classification task, it needs to be **trained**. We initialize the weights of the network randomly, then show it many examples of images whose labels are known. Based on the errors that the network makes in classifying these known images, we gradually adjust the weights of the network so that it correctly classifies these images. Two popular datasets for training CNNs are ImageNet [4] and MIT Places [10]. ImageNet contains 1000 categories of objects, such as dogs, birds, and other animals, while MIT Places contains 205 types of scenes such as bedrooms, kitchens, and forests.

Although CNNs perform well on a variety of tasks, it can be difficult to understand exactly what types of image features a CNN is using to work its magic. One trick for demystifying a CNN is to choose a neuron in a trained CNN, and attempt to generate an image that causes the neuron to activate strongly. We initialize the image with random noise, propagate the image forward through the network to compute the activation of the target neuron, then propagate the activation of the neuron backward through the network to compute an update direction for the image. We use this information to update the image, and repeat the process until convergence. This general strategy has been used to visualize the activations of individual neurons [8, 9], to generate images of particular object classes [5], to invert CNN features [1, 2], and to generate images to fool CNNs [3, 6].

Inceptionism builds on this line of work, adding three unique twists:

* **Layer amplification**: Instead of choosing a neuron and generating an image to maximize it, we instead choose a layer of the network and attempt to amplify the neurons in that layer that are already activated by the image. This leads to a feedback loop, causing the network to emphasize features of the image that are already present. Google's blog post doesn't specify many technical details around this point; in cnn-vis we achieve this by maximizing the sum of the absolute value and the square of each neuron in the chosen layer, but in principle any superlinear function of the chosen layer should work.
* **Multiscale, high-res images**: Because they are so computationally expensive, CNNs tend to work on relatively low-resolution images. The state-of-the-art GoogLeNet network [7], for example, works on input images that are 224x224 pixels. To generate higher-resolution images, Inceptionism "appl[ies] the algorithm iteratively on its own outputs and appl[ies] some zooming after each iteration". In addition to giving high resolution images, this trick also causes the generated images to have structure at multiple scales, giving them a fractal-like appearance. In cnn-vis we implement this idea by tiling the image with overlapping 224x224 patches, and interleaving updates on each patch. After some number of iterations we upsample the image, retile it, and continue.
* **Non-random initialization**: Instead of initializing our generated image with random noise or zero as is common in the literature, Inceptionism (and cnn-vis!) allow you to start from a user-specified image. I'm not sure how much scientific value this has, but it sure does look cool!

# Setup
## Caffe
cnn-vis is built on top of [Caffe](http://caffe.berkeleyvision.org/), an excellent open-source CNN implementation from Berkeley. You'll need to do the following:
* Install Caffe; follow the official [installation instructions](http://caffe.berkeleyvision.org/installation.html).
* Build the Python bindings for Caffe
* If you have an NVIDIA GPU, you can optionally install [cuDNN](https://developer.nvidia.com/cuDNN) to make Caffe even faster
* Set the environment variable `$CAFFE_ROOT` to point to the root directory of your Caffe installation
* Download the official Caffe pretrained GoogLeNet model; from `$CAFFE_ROOT` run the command
```
./scripts/download_model_binary.py models/bvlc_googlenet/
```
* Download a version of GoogLeNet pretrained on the MIT Places dataset [here](http://places.csail.mit.edu/downloadCNN.html); place it in `$CAFFE_ROOT/models/googlenet_places`.

## cnn-vis
Clone the repo, create a virtual environment, install requirements, and add the Caffe Python library to the virtualenv:
```
git clone https://github.com/jcjohnson/cnn-vis.git
cd cnn-vis
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
echo $CAFFE_ROOT/python > .env/lib/python2.7/site-packages/caffe.pth
```

# Usage
cnn-vis is a standalone Python script; you can control its behavior by passing various command-line arguments.

# References
[1] A. Dosovitskiy and T. Brox. "Inverting Convolutional Networks with Convolutional Networks", arXiv preprint arXiv:1506.02753 (2015).

[2] A. Mahendran and A. Vedaldi, "Understanding Deep Image Representations by Inverting Them", CVPR 2015

[3] A. Nguyen, J. Yosinski, J. Clune. "Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images", CVPR 2015

[4] O. Russakovsky, et al. "Imagenet large scale visual recognition challenge", IJCV 2014.

[5] K. Simonyan and A. Vedaldi and A. Zisserman, "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR 2014

[6] C. Szegedy, et al. "Intriguing properties of neural networks." arXiv preprint arXiv:1312.6199 (2013).

[7] C. Szegedy, et al. "Going Deeper with Convolutions", CVPR 2015.

[8] J. Yosinski, J. Clune, A. Nguyen, T. Fuchs, H. Lipson H (2015) "Understanding Neural Networks Through Deep Visualization", ICML 2015 Deep Learning workshop.

[9] M. D. Zeiler and R. Fergus. "Visualizing and understanding convolutional networks", ECCV 2014.

[10] B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. "Learning Deep Features for Scene Recognition using Places Database", NIPS 2014.

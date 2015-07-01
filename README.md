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

## CNN options
These options control the CNN that will be used to generate images.
* `--deploy_txt`: Path to the Caffe .prototxt file that defines the CNN model to be used. cnn-vis expects that this model's input comes from a blob named `data`. Default is the BVLC reference GoogLeNet.
* `--caffe_model`: Path to the `.caffemodel` file giving the weights of the CNN model to be used. Default is the BVLC reference GoogLeNet.
* `--batch_size`: The number of image patches to be computed in parallel. Larger values will take more GPU memory, but may be more efficient for larger images. Default is 1.
* `--mean_image`: By convention, most Caffe pretrained models do not work on raw images, but instead work on the residual after subtracting some average image. This parameter gives the path to a `.npy` file giving the mean image; the default is the ImageNet mean image provided by Caffe.
* `--gpu`: Which GPU to use for optimization. Setting this to a negative value will run the model in CPU mode.

## Image options
These options define the objective that will be optimized to generate an image
* `--image_type`: The type of image to generate. If this is `amplify_neuron` then we will attempt to maximize a single neuron in the network, similar to [5]. If this is `amplify_layer` then this will produce images in the style of Inceptionism.
* `--target_layer`: The name of the layer to target in the network. Earlier layers tend to encode lower level features like edges or blobs, while later layers tend to encode higher-level features like object parts. For convenience, a complete list of layers in order for GoogLeNet is given in the file `googlenet_layers.txt`.
* `--target_neuron`: If `image_type` is `amplify_neuron`, then `target_neuron` gives the index of the neuron to amplify.

## Initialization options
Options for setting the initial image. You can either seed the initial image from an existing image, or use random noise. In the case of random noise, we generate Gaussian white noise, then smooth it using Gaussian blur to prevent TV regularization from dominating the first few steps of optimization.
* `--initial_image:` Path to an image file to use to start optimization. If this flag is not set, then the image will be initialized from smoothed Gaussian white noise instead.
* `--initialization_scale`: If `initial_image` is not set, then this gives the standard deviation of the Gaussian white noise used to initialize the image. Default is 1.
* `--initialization_blur`: If `initial_image` is not set, this gives the standard deviation for the Gaussian kernel used to smooth the white noise image. Default is 0, corresponding to no smoothing.

## Resize options
Options for configuring multiscale zooming used to generate high-resolution images. To generate nice images, we want to start with a small initial size that is ideally not much bigger than the base resolution of the CNN, then gradually grow to larger images. 

Sizes may be specified as multiples of a **base size**; for noise initializations the base size is the input size of the CNN, and for image initializations the base size is the original size of the initial image.
* `--initial_size`: The initial size. Can be one of the following:
  * If not set, then the initial size is the base size.
  * `xF` where `F` is a floating point number, such as `x0.5`. The initial size will be a multiple of the base size.
  * `HxW` where `H` and `W` are integers, such as `600x800`. The initial image will have height and width `H` and `W` pixels respectively.
* `--final_size`: The final size, in the same format as `--initial_size`.
* `--num_sizes`: The number of sizes to use. Default is 1.
* `--resize_type`: How to space the intermediate sizes between the initial and final sizes. Choices are `geometric` or `linear`; default is `geometric`.

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

# cnn-vis

TODOs:
* Make an example for amplify_neuron
* Make an example gallery
* Add documentation for l1_weight, l2_weight, grad_clip

Inspired by Google's recent [Inceptionism](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html) blog post, cnn-vis is an open-source tool that lets you use convolutional neural networks to generate images. Here's an example:

<img src="http://cs.stanford.edu/people/jcjohns/cnn-vis-examples/example12.png" width=800px>

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

## Optimization options
We optimize using gradient descent, and use RMSProp to compute per-parameter adaptive learning rates.
* `--learning_rate`: The learning rate to use. Default is 1.0.
* `--decay_rate`: Decay rate for RMSProp. Default is 0.95. Usually when RMSProp is used for stochastic gradient descent, it is common to use values greater than 0.9 for the decay rate; however in this application our gradients are not stochastic, so lower decay rate values sometimes work well.
* `--num_steps`: The number of optimization steps to take at each size.
* `--use_pixel_learning_rates`: Because the image is tiled with overlapping windows of input size to the CNN, each pixel will be contained in either 1, 2, or 4 windows; this can cause ugly artifacts near the borders of window regions, especially for high learning rates. If this flag is passed, divide the learning rate for each pixel by the number of windows that the pixel is contained in; this can sometimes help alleviate this problem.
 
## P-norm regularization options
P-norm regularization prevents individual pixels from getting too large. For noise initializations, p-norm regularization pulls each pixel toward zero (corresponding to the mean ImageNet color) and for image initializations, p-norm regularization will pull each pixel toward the value of that pixel in the initial image. For noise initializations, relatively weak p-norm regularization tends to work well; for image initializations, p-norm regularization is the only term enforcing visual consistency with the initial image, so p-norm regularization should be stronger.
* `--alpha`: The exponent of the p-norm. Note that [5] uses L2 regularization, corresponding to `alpha=2.0` while [2] suggests using `alpha=6.0`. Default is 6.0.
* `--p_reg`: Regularization constant for p-norm regularization. Larger values will cause the p-norm constraint to be enforced more strongly. Default is 1e-4.
* `--p_scale`: Scaling constant; divide pixels by this value before computing the p-norm regularizer. Note that a non-unit value for `p_scale` can be absorbed into `p_reg`, so this is technically redudent; however it can be useful for both numeric stability and to make it easier to compare values of `p_reg` across different values of `alpha`.

## Auxiliary p-norm regularization options
Parameters for a second p-norm regularizer; however the second p-norm regularizer always pulls towards zero, while the first p-norm regularizer pulls toward the initial image if it is given. If the initial image contains very saturated regions (either very white or very black) then even small deviations around the initial value can result in pixel values outside the [0, 255] range. A trick for getting around this problem is adding a second p-norm regularizer with a high exponent (maybe 11) and very low regularization constant (maybe 1e-11). This regularizer will have little effect on pixels near the center of the [0, 255] range, but will push pixels outside this range back toward zero.
* `--alpha_aux`: Exponent for auxiliary p-norm regularization. Default is 6.
* `--p_reg_aux`: Regularization strength for auxiliary p-norm regularization. Default is 0 (no auxiliary p-norm regularizer).
* `--p_scale_aux`: Scaling constant for auxiliary p-norm regularizer, analogous to `p_scale`.

## Total Variation regularization options
Total Variation (TV) regularization encourages neighboring pixels to have similar values. For noise initializations this regularizer is critical; without it the generated image will exhibit large amounts of high-frequency noise. For image initializations it is less critical; strong p-regularization will keep the pixels close to the initial image, and this will be sufficient to prevent high-frequency noise.

As defined in [2], we compute the TV-norm of an image by approximating the magnitude of the image gradient using neighboring pixels, raising the image gradient to the power of beta, and summing over the image.

[2] suggests that starting with a low TV-norm regularization strength and increasing it over time gives good results. In cnn-vis we implement this idea by increasing the TV-norm regularization strength by a constant amount after a fixed number of iterations.
* `--beta`: Exponent for TV-regularization. As discussed in [2], values less than 1 will give rise to patches of solid color; setting beta to 2 or 2.5 tends to give good results. Default is 2.0.
* `--tv_reg`: Regularization strength for TV-regularization. Higher values will more strongly encourage the image to have a small TV-norm.
* `--tv_reg_scale`: Similar to `p_scale`, a scaling factor that the image is divided by prior to computing the TV-norm. As with `p_scale` this is technically redudent and can be absorbed into `tv_reg`.
* `--tv_reg_step_iter`: TV-norm regularization strength will be increased every `tv_reg_step_iter` steps. Default is 50.
* `--tv_reg_step`: Every `tv_reg_step_iter` steps, TV-norm regularization strength will increase by this amount. Default is 0, corresponding to a fixed TV-norm regularization strength.

## Output options
Options for controlling the output.
`--output_file`: Filename where the final image will be saved. Default is `out.png`.
`--rescale_image`: If this flag is given, then the image colors are rescaled to [0, 255] linearly; the minimum value of the image will be mapped to 0, and the maximum image value will map to 255. If this flag is not given, the image wil be clipped to the range [0, 255] for output. Rescaling the image values can reveal detail in highly saturated or desaturated image regions, but can lead to color distortion.
`--output_iter`: After every `output_iter` steps of optimization, some outputs will be produced. Exactly what is produced is controlled by `iter_behavior`.
`--iter_behavior`: What should happen every `output_iter` steps. The allowed options are shown below. Options can be combined with `+` to have multiple types of behavior; for example `show+print+save` will do all three every `output_iter` steps.
  * `show`: Show the current image using matplotlib
  * `save`: Save the current image; the filename will append the size number and iteration number to `output_file`.
  * `print`: Print the current iteration number, along with some statistics about the image and the gradients from the different regularizers.
  * `plot_pix`: Plot the values of a few image pixels over time using matplotlib; this can give you a rough sense for how the optimization process is proceeding. For example very oscillatory behavior indicates that something bad is happening, and the learning rate should be decreased.

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

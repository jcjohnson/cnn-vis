import argparse, os

import numpy as np
from scipy.misc import imresize, imsave, imread

import caffe

"""

References:
  [1] A. Mahendran and A. Vedaldi, "Understanding Deep Image Representations by Inverting Them", CVPR 2015
  [2] K. Simonyan and A. Vedaldi and A. Zisserman, "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR 2014
"""



def tv_norm(x, beta=2.0):
  """
  Compute the total variation norm and its gradient.
  
  The total variation norm is the sum of the image gradient
  raised to the power of beta, summed over the image.
  We approximate the image gradient using finite differences.
  We use the total variation norm as a regularizer to encourage
  smoother images.

  Inputs:
  - x: numpy array of shape (1, C, H, W)

  Returns a tuple of:
  - loss: Scalar giving the value of the norm
  - dx: numpy array of shape (1, C, H, W) giving gradient of the loss
        with respect to the input x.
  """
  assert x.shape[0] == 1
  x_diff = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
  y_diff = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
  grad_norm2 = x_diff ** 2.0 + y_diff ** 2.0
  grad_norm_beta = grad_norm2 ** (beta / 2.0)
  loss = np.sum(grad_norm_beta)
  dgrad_norm2 = (beta / 2.0) * grad_norm2 ** (beta / 2.0 - 1.0)
  dx_diff = 2.0 * x_diff * dgrad_norm2
  dy_diff = 2.0 * y_diff * dgrad_norm2
  dx = np.zeros_like(x)
  dx[:, :, :-1, :-1] += dx_diff + dy_diff
  dx[:, :, :-1, 1:] -= dx_diff
  dx[:, :, 1:, :-1] -= dy_diff
  return loss, dx


def p_norm(x, p=6.0, scale=10.0):
  """
  Compute the p-norm for an image and its gradient.
  
  The p-norm is defined as

  |x|_p = (\sum_i |x_i|^p)^(1/p)

  so strictly speaking this fucntion actually computes the pth power of the
  p-norm.

  We use it as a regularizer to prevent individual pixels from getting too big.
  We don't actually want to drive pixels toward zero; we are more interested in
  making sure they stay within a reasonable range. This suggests that we divide
  the pixels by a scaling factor and use a high value of p; as suggested by
  [1] p=6 tends to work well.

  Inputs:
  - x: numpy array of any shape
  - p: Power for p-norm
  - scale: Scale for p-norm.

  Returns a tuple of:
  - loss: Value of the p-norm 
  """
  loss = (np.abs(x / scale) ** p).sum()
  grad = p / scale * np.sign(x / scale) * np.abs(x / scale) ** (p - 1)
  return loss, grad


def rmsprop(dx, cache=None, decay_rate=0.95):
  """
  Use RMSProp to compute a step from gradients.

  Inputs:
  - dx: numpy array of gradients.
  - cache: numpy array of same shape as dx giving RMSProp cache
  - decay_rate: How fast to decay cache

  Returns a tuple of:
  - step: numpy array of the same shape as dx giving the step. Note that this
    does not yet take the learning rate into account.
  - cache: Updated RMSProp cache.
  """
  if cache is None:
    cache = np.zeros_like(dx)
  cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
  step = -dx / np.sqrt(cache + 1e-8)
  return step, cache


def get_cnn_grads(cur_img, regions, net, target_layer, step_type='amplify_layer', **kwargs):
  """
  Inputs:
  - cur_img: 3 x H x W
  - regions: Array of (y0, y1, x0, x1); must all have same shape as input to CNN
  - target_layer: String
  
  Returns:
  - grads: N x 3 x h x w array where grads[i] is the image gradient for regions[i] of cur_img
  """
  cur_batch = np.zeros_like(net.blobs['data'].data)
  batch_size = cur_batch.shape[0]
  next_idx = 0
  
  def run_cnn(data):
    net.forward(data=data)
    if step_type == 'amplify_layer':
      l1_weight = kwargs.get('L1_weight', 1.0)
      l2_weight = kwargs.get('L2_weight', 1.0)
      grad_clip = kwargs.get('grad_clip', 5)
      target_data = net.blobs[target_layer].data.copy()
      target_diff = -l1_weight * np.abs(target_data)
      target_diff -= l2_weight * np.clip(target_data, -grad_clip, grad_clip)
      net.blobs[target_layer].diff[...] = target_diff
    elif step_type == 'maximize_class':
      if 'target_idx' not in kwargs:
        raise ValueError('Must specify target_idx for step_type=maximize_class')
      target_blob.diff[...] = 0.0
      target_blob.diff[0, target_idx] = -1.0
    net.backward(start=target_layer)
    return net.blobs['data'].diff.copy()
  
  grads = []
  for region in regions:
    y0, y1, x0, x1 = region
    cur_batch[next_idx] = cur_img[:, y0:y1, x0:x1]
    next_idx += 1
    if next_idx == batch_size:
      grads.append(run_cnn(cur_batch))
      next_idx = 0
  if next_idx > 0:
    grad = run_cnn(cur_batch)
    grads.append(grad[:next_idx])
  
  return np.vstack(grads)


def img_to_uint(img, mean_img=None):
  """
  Do post-processing to convert images from caffe format to something more reasonable.

  Inputs:
  - img: numpy array of shape (1, C, H, W)
  - mean_img: numpy array giving a mean image to add in

  Returns:
  A version of img that can be saved to disk or shown with matplotlib
  """
  if mean_img is not None:
    # Be lazy and just add the mean color
    img = img + mean_img.mean()

  # Renormalize so everything is in the range [0, 255]
  # low, high = img.min(), img.max()
  low = max(img.mean() - 2.5 * img.std(axis=None), img.min())
  high = min(img.mean() + 2.5 * img.std(axis=None), img.max())
  img = np.clip(img, low, high)
  img = 255.0 * (img - low) / (high - low)

  # Squeeze out extra dimensions and flip from (C, H, W) to (H, W, C)
  img = img.squeeze().transpose(1, 2, 0)

  # Caffe models are trained with BGR; flip to RGB
  img = img[:, :, [2, 1, 0]]

  # finally convert to uint8
  return img.astype('uint8')


def build_parser():
  parser = argparse.ArgumentParser()
  
  # Model options
  parser.add_argument('--deploy_txt', default='$CAFFE_ROOT/models/bvlc_googlenet/deploy_1.prototxt')
  parser.add_argument('--caffe_model', default='$CAFFE_ROOT/models/bvlc_googlenet/bvlc_googlenet.caffemodel')
  parser.add_argument('--target_layer', default='inception_4d/3x3_reduce')
  parser.add_argument('--mean_image', default='$CAFFE_ROOT/python/caffe/imagenet/ilsvrc_2012_mean.npy')
  parser.add_argument('--image_type', default='amplify_layer')
  parser.add_argument('--gpu', type=int, default=0)

  # Optimization options
  parser.add_argument('--learning_rate', type=float, default=1.0)
  parser.add_argument('--decay_rate', type=float, default=0.95)
  parser.add_argument('--num_steps', type=int, default=1000)
  
  # Options for layer amplification
  parser.add_argument('--amplify_l1_weight', type=float, default=1.0)
  parser.add_argument('--amplify_l2_weight', type=float, default=1.0)
  parser.add_argument('--amplify_grad_clip', type=float, default=5.0)

  # P-norm regularization options
  parser.add_argument('--alpha', type=float, default=6.0)
  parser.add_argument('--p_scale', type=float, default=50)
  parser.add_argument('--p_reg', type=float, default=1e-4)

  # TV regularization options
  parser.add_argument('--beta', type=float, default=2.0)
  parser.add_argument('--tv_reg', type=float, default=0.5)
  parser.add_argument('--tv_reg_step', type=float, default=0.25)
  parser.add_argument('--tv_reg_step_iter', type=int, default=50)

  # Output options
  parser.add_argument('--output_file', default='out.png')
  parser.add_argument('--output_iter', default=50, type=int)
  
  return parser


def main():
  parser = build_parser()
  args = parser.parse_args()
  
  if args.gpu < 0:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

  # Build the net; paths may have CAFFE_ROOT
  proto_file = os.path.expandvars(args.deploy_txt)
  caffe_model_file = os.path.expandvars(args.caffe_model)
  net = caffe.Net(proto_file, caffe_model_file, caffe.TEST)

  C, H, W = net.blobs['data'].data.shape[1:]
  
  # Initialize image
  mean_img = np.load(os.path.expandvars(args.mean_image))
  img = np.random.randn(1, C, H, W)
  cache = None

  # Run optimization
  tv_reg = args.tv_reg
  for t in xrange(args.num_steps):
    regions = [(0, H, 0, W)]
    cnn_grad = get_cnn_grads(img, regions, net, args.target_layer,
                   step_type=args.image_type,
                   L1_weight=args.amplify_l1_weight,
                   L2_weight=args.amplify_l2_weight,
                   grad_clip=args.amplify_grad_clip)    
    p_loss, p_grad = p_norm(img, p=args.alpha, scale=args.p_scale)
    tv_loss, tv_grad = tv_norm(img, beta=args.beta)

    dimg = cnn_grad + args.p_reg * p_grad + tv_reg * tv_grad

    step, cache = rmsprop(dimg, cache=cache, decay_rate=args.decay_rate)
    step *= args.learning_rate
    img += step

    if (t + 1) % args.tv_reg_step_iter == 0:
      tv_reg += args.tv_reg_step

    if (t + 1) % args.output_iter == 0:
      print 'Finished iteration %d / %d' % (t + 1, args.num_steps)
      print 'p_loss: ', p_loss
      print 'tv_loss: ', tv_loss
      print 'mean p_grad: ', np.abs(p_grad).mean()
      print 'mean tv_grad: ', np.abs(tv_grad).mean()
      print 'mean cnn_grad: ', np.abs(cnn_grad).mean()
      print 'image mean / std: ', img.mean(), img.std()
      print 'mean step / val: ', np.mean(np.abs(step) / np.abs(img))
      name, ext = os.path.splitext(args.output_file)
      filename = '%s_%d%s' % (name, t + 1, ext)
      img_uint = img_to_uint(img, mean_img)
      imsave(filename, img_uint)


if __name__ == '__main__':
  main()

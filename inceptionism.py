import argparse, os, tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize, imsave, imread
from scipy.ndimage.filters import gaussian_filter

import caffe

"""

References:
  [1] A. Mahendran and A. Vedaldi, "Understanding Deep Image Representations by Inverting Them", CVPR 2015
  [2] K. Simonyan and A. Vedaldi and A. Zisserman, "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR 2014
"""

def tv_norm(x, beta=2.0, verbose=False):
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
  grad_norm2[grad_norm2 < 1e-3] = 1e-3
  grad_norm_beta = grad_norm2 ** (beta / 2.0)
  loss = np.sum(grad_norm_beta)
  dgrad_norm2 = (beta / 2.0) * grad_norm2 ** (beta / 2.0 - 1.0)
  dx_diff = 2.0 * x_diff * dgrad_norm2
  dy_diff = 2.0 * y_diff * dgrad_norm2
  dx = np.zeros_like(x)
  dx[:, :, :-1, :-1] += dx_diff + dy_diff
  dx[:, :, :-1, 1:] -= dx_diff
  dx[:, :, 1:, :-1] -= dy_diff
  
  def helper(name, x):
    num_nan = np.isnan(x).sum()
    num_inf = np.isinf(x).sum()
    num_zero = (x == 0).sum()
    print '%s: NaNs: %d infs: %d zeros: %d' % (name, num_nan, num_inf, num_zero)
  
  if verbose:
    print '-' * 40
    print 'tv_norm debug output'
    helper('x', x)
    helper('x_diff', x_diff)
    helper('y_diff', y_diff)
    helper('grad_norm2', grad_norm2)
    helper('grad_norm_beta', grad_norm_beta)
    helper('dgrad_norm2', dgrad_norm2)
    helper('dx_diff', dx_diff)
    helper('dy_diff', dy_diff)
    helper('dx', dx)
    print
  
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
    elif step_type == 'amplify_neuron':
      if 'target_neuron' not in kwargs:
        raise ValueError('Must specify target_neuron for step_type=amplify_neuron')
      target_idx = kwargs['target_neuron']
      net.blobs[target_layer].diff[...] = 0.0
      net.blobs[target_layer].diff[:, target_idx] = -1.0
    else:
      raise ValueError('Unrecognized step_type "%s"' % step_type)
    net.backward(start=target_layer)
    return net.blobs['data'].diff.copy()
  
  grads = []
  for region in regions:
    y0, y1, x0, x1 = region
    cur_batch[next_idx] = cur_img[0, :, y0:y1, x0:x1]
    next_idx += 1
    if next_idx == batch_size:
      grads.append(run_cnn(cur_batch))
      next_idx = 0
  if next_idx > 0:
    grad = run_cnn(cur_batch)
    grads.append(grad[:next_idx])
  
  vgrads = np.vstack(grads)
  return vgrads


def img_to_uint(img, mean_img=None, rescale=False):
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
    img = 1.2 * img + mean_img.mean()

  # Renormalize so everything is in the range [0, 255]
  if rescale:
    low, high = img.min(), img.max()
  else:
    low, high = 0, 255
  # low = max(img.mean() - 2.5 * img.std(axis=None), img.min())
  # high = min(img.mean() + 2.5 * img.std(axis=None), img.max())
  img = np.clip(img, low, high)
  img = 255.0 * (img - low) / (high - low)

  # Squeeze out extra dimensions and flip from (C, H, W) to (H, W, C)
  img = img.squeeze().transpose(1, 2, 0)

  # Caffe models are trained with BGR; flip to RGB
  img = img[:, :, [2, 1, 0]]

  # finally convert to uint8
  return img.astype('uint8')


def uint_to_img(uint_img, mean_img=None):
  """
  Do pre-processing to convert images from a normal format to caffe format.
  """
  img = uint_img.astype('float')
  img = img[:, :, [2, 1, 0]]
  img = img.transpose(2, 0, 1)
  img = img[np.newaxis, :, :, :]
  if mean_img is not None:
    img = img - mean_img.mean()
  return img


def resize_img(img, new_size, mean_img=None):
  img_uint = img_to_uint(img, mean_img)
  img_uint_r = imresize(img_uint, new_size, interp='bicubic')
  img_r = uint_to_img(img_uint_r, mean_img)
  return img_r
  high, low = img.max(), img.min()
  img_shifted = 255.0 * (img - low) / (high - low)
  img_uint = img_shifted.squeeze().transpose(1, 2, 0).astype('uint8')
  img_uint_r = imresize(img_uint, new_size)
  img_shifted_r = img_uint_r.astype(img.dtype).transpose(2, 0, 1)[None, :, :, :]
  img_r = (img_shifted_r / 255.0) * (high - low) + low
  return img_r


def write_temp_deploy(source_prototxt, batch_size):
  """
  Modifies an existing prototxt by adding force_backward=True and setting
  the batch size to a specific value. A modified prototxt file is written
  as a temporary file.
  
  Inputs:
  - source_prototxt: Path to a deploy.prototxt that will be modified
  - batch_size: Desired batch size for the network
  
  Returns:
  - path to the temporary file containing the modified prototxt
  """
  _, target = tempfile.mkstemp()
  with open(source_prototxt, 'r') as f:
    lines = f.readlines()
  force_back_str = 'force_backward: true\n'
  if force_back_str not in lines:
    lines.insert(1, force_back_str)

  found_batch_size_line = False
  with open(target, 'w') as f:
    for line in lines:
      if line.startswith('input_dim:') and not found_batch_size_line:
        found_batch_size_line = True
        line = 'input_dim: %d\n' % batch_size
      f.write(line)
  
  return target


def get_ranges(total_length, region_length, num):
  starts = np.linspace(0, total_length - region_length, num)
  starts = [int(round(s)) for s in starts]
  ranges = [(s, s + region_length) for s in starts]
  return ranges


def check_ranges(total_length, ranges):
  """
  Check to make sure the given ranges are valid.
  
  Inputs:
  - total_length: Integer giving total length
  - ranges: Sorted list of tuples giving (start, end) for each range.
  
  Returns: Boolean telling whether ranges are valid.
  """
  # The start of the first range must be 0
  if ranges[0][0] != 0:
    return False
  
  # The end of the last range must fill the length
  if ranges[-1][1] != total_length:
    return False
  
  for i, cur_range in enumerate(ranges):
    # The ranges must be distinct
    if i + 1 < len(ranges) and cur_range[0] == ranges[i + 1][0]:
      return False
    # The ranges must cover all the pixels
    if i + 1 < len(ranges) and cur_range[1] < ranges[i + 1][0]:
      return False
    # Each range should not overlap with its second neighbor
    if i + 2 < len(ranges) and cur_range[1] >= ranges[i + 2][0]:
      return False
  return True


def get_best_ranges(total_length, region_length, target_overlap):
  num = 1
  best_ranges = None
  best_diff = None
  while True:
    ranges = get_ranges(total_length, region_length, num)
    if check_ranges(total_length, ranges):
      overlap = 0 if len(ranges) == 1 else ranges[0][1] - ranges[1][0]
      overlap_diff = abs(overlap - target_overlap)
      if best_diff is None or overlap_diff < best_diff:
        best_diff = overlap_diff
        best_ranges = ranges
        return best_ranges
    else:
      # If an earlier num worked but this one does not, we have found
      # all feasible packings so break
      if best_ranges is not None:
        break
    num = num + 1
  return best_ranges


def get_regions(total_size, region_size, overlap):
  print 'total_size: ', total_size
  print 'region_size: ', region_size
  print 'overlap: ', overlap
  H, W = total_size
  h, w = region_size
  
  y_ranges = get_best_ranges(H, h, overlap)
  x_ranges = get_best_ranges(W, w, overlap)

  regions_even = []
  regions_odd = []
  all_regions = []
  for i, x_range in enumerate(x_ranges):
    for j, y_range in enumerate(y_ranges):
      region = (y_range[0], y_range[1], x_range[0], x_range[1])
      if i % 2 == j % 2:
        regions_even.append(region)
      else:
        regions_odd.append(region)
  return regions_even, regions_odd


def count_regions_per_pixel(total_size, regions):
  counts = np.zeros(total_size)
  for region in regions:
    y0, y1, x0, x1 = region
    counts[y0:y1, x0:x1] += 1
  return counts


def get_base_size(net_size, initial_image):
  if initial_image is None:
    return net_size[2:]
  else:
    img = imread(initial_image)
    return img.shape[:2]


def get_size_sequence(base_size, initial_size, final_size, num_sizes, resize_type):
  base_h, base_w = base_size
  
  def parse_size_str(size_str):
    if size_str is None:
      return base_size
    elif size_str.startswith('x'):
      scale = float(size_str[1:])
      h = int(scale * base_h)
      w = int(scale * base_w)
      return h, w
    elif 'x' in size_str:
      h, w = size_str.split('x')
      return int(h), int(w)
  
  initial_h, initial_w = parse_size_str(initial_size)
  final_h, final_w = parse_size_str(final_size)
  
  if num_sizes == 1:
    return [(initial_h, initial_w)]
  else:
    if resize_type == 'geometric':
      h0, h1 = np.log10(initial_h), np.log10(final_h)
      w0, w1 = np.log10(initial_w), np.log10(final_w)
      heights = np.logspace(h0, h1, num_sizes)
      widths = np.logspace(w0, w1, num_sizes)
    elif resize_type == 'linear':
      heights = np.linspace(initial_h, final_h, num_sizes)
      widths = np.linspace(initial_w, final_w, num_sizes)
    else:
      raise ValueError('Invalid resize_type "%s"' % resize_type)
    heights = np.round(heights).astype('int')
    widths = np.round(widths).astype('int')
    return zip(heights, widths)


def initialize_img(net_size, initial_image, initial_size, mean_img, scale, blur):
  _, C, H, W = net_size

  def init_size_fn(h, w):
    if initial_size is None:
      return h, w
    elif initial_size.startswith('x'):
      scale = float(initial_size[1:])
      return int(scale * h), int(scale * w)
    elif 'x' in initial_size:
      h, w = initial_size.split('x')
      return int(h), int(w)
  
  if initial_image is not None:
    init_img = imread(initial_image)
    init_h, init_w = init_img.shape[:2]
    init_h, init_w = init_size_fn(init_h, init_w)
    init_img = imresize(init_img, (init_h, init_w))
    init_img = uint_to_img(init_img, mean_img)
  else:
    init_h, init_w = init_size_fn(H, W)
    init_img = scale * np.random.randn(1, C, init_h, init_w)
    init_img_uint = img_to_uint(init_img, mean_img)
    init_img_uint_blur = gaussian_filter(init_img_uint, sigma=blur)
    init_img = uint_to_img(init_img_uint_blur, mean_img)

  return init_img


def build_parser():
  parser = argparse.ArgumentParser()
  
  # Model options
  parser.add_argument('--deploy_txt', default='$CAFFE_ROOT/models/bvlc_googlenet/deploy.prototxt')
  parser.add_argument('--caffe_model', default='$CAFFE_ROOT/models/bvlc_googlenet/bvlc_googlenet.caffemodel')
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--target_layer', default='inception_4d/3x3_reduce')
  parser.add_argument('--mean_image', default='$CAFFE_ROOT/python/caffe/imagenet/ilsvrc_2012_mean.npy')
  parser.add_argument('--image_type', default='amplify_layer')
  parser.add_argument('--target_neuron', default=0, type=int)
  parser.add_argument('--initial_image', default=None)
  parser.add_argument('--gpu', type=int, default=0)

  # Noise initialization options
  parser.add_argument('--initialization_scale', type=float, default=1.0)
  parser.add_argument('--initialization_blur', type=float, default=0.0)

  # Resize options
  parser.add_argument('--initial_size', default=None)
  parser.add_argument('--final_size', default=None)
  parser.add_argument('--num_sizes', default=1, type=int)
  parser.add_argument('--resize_type', default='geometric')
  parser.add_argument('--overlap', default=50, type=int)
  
  # Optimization options
  parser.add_argument('--learning_rate', type=float, default=1.0)
  parser.add_argument('--decay_rate', type=float, default=0.95)
  parser.add_argument('--num_steps', type=int, default=1000)
  parser.add_argument('--use_pixel_learning_rates', action='store_true')
  
  # Options for layer amplification
  parser.add_argument('--amplify_l1_weight', type=float, default=1.0)
  parser.add_argument('--amplify_l2_weight', type=float, default=1.0)
  parser.add_argument('--amplify_grad_clip', type=float, default=5.0)

  # P-norm regularization options
  parser.add_argument('--alpha', type=float, default=6.0)
  parser.add_argument('--p_scale', type=float, default=1.0)
  parser.add_argument('--p_reg', type=float, default=1e-4)
  
  # Auxillary P-norm regularization options
  parser.add_argument('--alpha_aux', type=float, default=6.0)
  parser.add_argument('--p_scale_aux', type=float, default=1.0)
  parser.add_argument('--p_reg_aux', type=float, default=0.0)

  # TV regularization options
  parser.add_argument('--beta', type=float, default=2.0)
  parser.add_argument('--tv_reg', type=float, default=0.5)
  parser.add_argument('--tv_reg_scale', type=float, default=1.0)
  parser.add_argument('--tv_reg_step', type=float, default=0.0)
  parser.add_argument('--tv_reg_step_iter', type=int, default=50)

  # Output options
  parser.add_argument('--output_file', default='out.png')
  parser.add_argument('--output_iter', default=50, type=int)
  parser.add_argument('--rescale_image', action='store_true')
  parser.add_argument('--iter_behavior', default='save+print')
  
  return parser


def main(args):
  if args.gpu < 0:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

  # Build the net; paths may have CAFFE_ROOT
  proto_file = os.path.expandvars(args.deploy_txt)
  proto_file = write_temp_deploy(proto_file, args.batch_size)
  caffe_model_file = os.path.expandvars(args.caffe_model)
  net = caffe.Net(proto_file, caffe_model_file, caffe.TEST)

  net_size = net.blobs['data'].data.shape
  C, H, W = net_size[1:]

  mean_img = np.load(os.path.expandvars(args.mean_image))
  init_img = initialize_img(net_size, args.initial_image, args.initial_size, mean_img,
                  args.initialization_scale,
                  args.initialization_blur)
  img = init_img.copy()
  if args.initial_image is None:
    init_img = None

  # Get size sequence
  base_size = get_base_size(net_size, args.initial_image)
  print 'base_size is %r' % (base_size,)
  size_sequence = get_size_sequence(base_size,
                                    args.initial_size,
                                    args.final_size,
                                    args.num_sizes,
                                    args.resize_type)
  
  # Run optimization
  for size_idx, size in enumerate(size_sequence):
    size_flag = False
    if size_idx > 0:
      img = resize_img(img, size, mean_img)
      if init_img is not None:
        raw_init = imread(args.initial_image)
        init_img_uint = imresize(raw_init, size)
        init_img = uint_to_img(init_img_uint, mean_img)

    tv_reg = args.tv_reg
    regions = get_regions((img.shape[2], img.shape[3]), (H, W), args.overlap)
    regions_even, regions_odd = regions
    regions_per_pixel = count_regions_per_pixel((img.shape[2], img.shape[3]), regions_even+regions_odd)
    pixel_learning_rates = 1.0 / regions_per_pixel
    caches = {}
    pix_history = defaultdict(list)
    pix = [(100, 100), (200, 200), (100, 200), (200, 100)]
    for t in xrange(args.num_steps):
      for c in [0, 1, 2]:
        for py, px in pix:
          pix_history[(c, py, px)].append(img[0, c, py, px])

      for cur_regions in [regions_even, regions_odd]:
        if len(cur_regions) == 0: continue
        cnn_grad = get_cnn_grads(img, cur_regions, net, args.target_layer,
                       step_type=args.image_type,
                       L1_weight=args.amplify_l1_weight,
                       L2_weight=args.amplify_l2_weight,
                       grad_clip=args.amplify_grad_clip,
                       target_neuron=args.target_neuron)
        for region_idx, region in enumerate(cur_regions):
          y0, y1, x0, x1 = region
          img_region = img[:, :, y0:y1, x0:x1]
          if init_img is not None:
            init_region = init_img[0, :, y0:y1, x0:x1]
            p_loss, p_grad = p_norm(img_region - init_region, p=args.alpha, scale=args.p_scale)
          else:
            p_loss, p_grad = p_norm(img_region, p=args.alpha, scale=args.p_scale)
          p_loss_aux, p_grad_aux = p_norm(img_region, p=args.alpha_aux, scale=args.p_scale_aux)
          tv_loss, tv_grad = tv_norm(img_region / args.tv_reg_scale, beta=args.beta, verbose=size_flag)
          tv_grad /= args.tv_reg_scale
          
          dimg = cnn_grad[region_idx] + args.p_reg * p_grad + args.p_reg_aux * p_grad_aux + tv_reg * tv_grad

          cache = caches.get(region, None)
          step, cache = rmsprop(dimg, cache=cache, decay_rate=args.decay_rate)
          caches[region] = cache
          step *= args.learning_rate
          if args.use_pixel_learning_rates:
            step *= pixel_learning_rates[y0:y1, x0:x1]
          img[:, :, y0:y1, x0:x1] += step

      if (t + 1) % args.tv_reg_step_iter == 0:
        tv_reg += args.tv_reg_step

      if (t + 1) % args.output_iter == 0:
        for p, h in pix_history.iteritems():
          plt.plot(h)
        plt.show()

        should_show = 'show' in args.iter_behavior
        should_save = 'save' in args.iter_behavior
        should_print = args.iter_behavior
        if should_print:
          print ('Finished iteration %d / %d for size %d / %d' % 
                (t + 1, args.num_steps, size_idx + 1, len(size_sequence)))
          print 'p_loss: ', p_loss
          print 'tv_loss: ', tv_loss
          print 'mean p_grad: ', np.abs(args.p_reg * p_grad).mean()
          print 'mean tv_grad: ', np.abs(tv_reg * tv_grad).mean()
          print 'mean cnn_grad: ', np.abs(cnn_grad).mean()
          print 'step mean, median: ', np.abs(step).mean(), np.median(np.abs(step))
          print 'image mean, std: ', img.mean(), img.std()
          print 'mean step / val: ', np.mean(np.abs(step) / np.abs(img_region))
        img_uint = img_to_uint(img, mean_img, rescale=args.rescale_image)
        if should_show:
          plt.imshow(img_uint, interpolation='none')
          plt.axis('off')
          plt.gcf().set_size_inches(15, 10)
          plt.show()
        if should_save:
          name, ext = os.path.splitext(args.output_file)
          filename = '%s_%d%s' % (name, t + 1, ext)
          imsave(filename, img_uint)
          
          
  img_uint = img_to_uint(img, mean_img, rescale=args.rescale_image)
  imsave(args.output_file, img_uint)


if __name__ == '__main__':
  parser = build_parser()
  args = parser.parse_args()
  main(args)

# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""AutoAugment policy for CIFAR10."""
import inspect
import math
import tensorflow.compat.v2 as tf
import tensorflow_addons.image as contrib_image


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.


def policy_cifar10():
  """CIFAR10 AutoAugment from https://arxiv.org/pdf/1805.09501.pdf Tab 7."""
  policy = [
      [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],  # 0
      [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],  # 1
      [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],  # 2
      [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],  # 3
      [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],  # 4
      [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],  # 5
      [('Color', 0.4, 3), ('Brightness', 0.6, 7)],  # 6
      [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],  # 7
      [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],  # 8
      [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)],  # 9
      [('Color', 0.7, 7), ('TranslateX', 0.5, 8)],  # 10
      [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],  # 11
      [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],  # 12
      [('Brightness', 0.9, 6), ('Color', 0.2, 8)],  # 13
      [('Solarize', 0.5, 2), ('Invert', 0., 3)],  # 14
      [('Equalize', 0.2, 0), ('AutoContrast', 0.6, 0)],  # 15
      [('Equalize', 0.2, 8), ('Equalize', 0.6, 4)],  # 16
      [('Color', 0.9, 9), ('Equalize', 0.6, 6)],  # 17
      [('AutoContrast', 0.8, 4), ('Solarize', 0.2, 8)],  # 18
      [('Brightness', 0.1, 3), ('Color', 0.7, 0)],  # 19
      [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],  # 20
      [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],  # 21
      [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],  # 22
      [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],  # 23
      [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)],  # 24
  ]
  return policy


def blend(image1, image2, factor):
  """Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  """
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1, tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def cutout(image, pad_size, replace=0):
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.

  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.

  Returns:
    An image Tensor that is of type uint8.
  """
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height,
      dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width,
      dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [image_height - (lower_pad + upper_pad),
                  image_width - (left_pad + right_pad)]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims, constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace,
      image)
  return image


def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)


def solarize_increasing(image, threshold=128):
  # Version of solarize where the magnitude of the transformation
  # increases with M.
  image = tf.cast(image, dtype=tf.float32)
  image = tf.where(image < (256 - threshold), image, 255 - image)
  return tf.cast(image, tf.uint8)


def solarize_add(image, addition=0, threshold=128):
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = tf.cast(image, tf.int64) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.where(image < threshold, added_image, image)


def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def rotate(image, degrees, replace):
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
    replace: A one or three value 1D tensor to fill empty pixels caused by
      the rotate operation.

  Returns:
    The rotated version of image.
  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = contrib_image.rotate(wrap(image), radians)
  return unwrap(image, replace)


def translate_x(image, pixels, replace):
  """Equivalent of PIL Translate in X dimension."""
  image = contrib_image.translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


def translate_x_rel(image, pct, replace):
  """Equivalent of PIL Translate in X dimension."""
  max_x = tf.shape(image)[1]
  pixels = tf.cast(max_x, tf.float32) * pct
  image = contrib_image.translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


def translate_y(image, pixels, replace):
  """Equivalent of PIL Translate in Y dimension."""
  image = contrib_image.translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


def translate_y_rel(image, pct, replace):
  """Equivalent of PIL Translate in Y dimension."""
  max_y = tf.shape(image)[0]
  pixels = tf.cast(max_y, tf.float32) * pct
  image = contrib_image.translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


def shear_x(image, level, replace):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = contrib_image.transform(
      wrap(image), [1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image, replace)


def shear_y(image, level, replace):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = contrib_image.transform(
      wrap(image), [1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image, replace)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def invert(image):
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


def wrap(image):
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], 2)
  return extended


def unwrap(image, replace):
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = flattened_image[:, 3]

  replace = tf.constant([replace] * 3 + [1], image.dtype)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.expand_dims(tf.equal(alpha_channel, 0), 1),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
  return image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize_increasing,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateXRel': translate_x_rel,
    'TranslateY': translate_y,
    'TranslateYRel': translate_y_rel,
    'Cutout': cutout,
}


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level):
  # Such that the magnitude of the transformation
  # increases with M and to have two-sided transformations.
  level = (level/_MAX_LEVEL) * 0.9
  level = 1. + _randomly_negate_tensor(level)
  return (level,)


def _shear_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level, translate_const):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_rel_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.45
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def level_to_arg(hparams):
  return {
      'AutoContrast': lambda level: (),
      'Equalize': lambda level: (),
      'Invert': lambda level: (),
      'Rotate': _rotate_level_to_arg,
      'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4),),
      'PosterizeIncreasing': lambda level: (4 - int((level/_MAX_LEVEL) * 4),),
      'Solarize': lambda level: (int((level/_MAX_LEVEL) * 256),),
      'SolarizeIncreasing': lambda level: (int((level/_MAX_LEVEL) * 256),),
      'SolarizeAdd': lambda level: (int((level/_MAX_LEVEL) * 110),),
      'Color': _enhance_level_to_arg,
      'ColorIncreasing': _enhance_increasing_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'ContrastIncreasing': _enhance_increasing_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'BrightnessIncreasing': _enhance_increasing_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'SharpnessIncreasing': _enhance_increasing_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      # pylint:disable=g-long-lambda
      'Cutout': lambda level: (
          int((level/_MAX_LEVEL) * hparams['cutout_const']),),
      'TranslateX': lambda level: _translate_level_to_arg(
          level, hparams['translate_const']),
      'TranslateXRel': _translate_rel_level_to_arg,
      'TranslateY': lambda level: _translate_level_to_arg(
          level, hparams['translate_const']),
      'TranslateYRel': _translate_rel_level_to_arg,
      # pylint:enable=g-long-lambda
  }


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = level_to_arg(augmentation_hparams)[name](level)

  # Check to see if prob is passed into function. This is used for operations
  # where we alter bboxes independently.
  # pytype:disable=wrong-arg-types
  if 'prob' in inspect.getfullargspec(func)[0]:
    args = tuple([prob] + list(args))
  # pytype:enable=wrong-arg-types

  # Add in replace arg if it is required for the function that is being called.
  # pytype:disable=wrong-arg-types
  if 'replace' in inspect.getfullargspec(func)[0]:
    # Make sure replace is the final argument
    assert 'replace' == inspect.getfullargspec(func)[0][-1]
    args = tuple(list(args) + [replace_value])
  # pytype:enable=wrong-arg-types

  return (func, prob, args)


def _apply_func_with_prob(func, image, args, prob):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)

  # If prob is a function argument, then this randomness is being handled
  # inside the function, so make sure it is always called.
  # pytype:disable=wrong-arg-types
  if 'prob' in inspect.getfullargspec(func)[0]:
    prob = 1.0
  # pytype:enable=wrong-arg-types

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image = tf.cond(
      should_apply_op,
      lambda: func(image, *args),
      lambda: image)
  return augmented_image


def select_and_apply_random_policy(policies, image):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image),
        lambda: image)
  return image


def build_and_apply_nas_policy(policies, image, augmentation_hparams):
  """Build a policy from the given policies passed in and apply to image.

  Args:
    policies: list of lists of tuples in the form `(func, prob, level)`, `func`
      is a string name of the augmentation function, `prob` is the probability
      of applying the `func` operation, `level` is the input argument for
      `func`.
    image: tf.Tensor that the resulting policy will be applied to.
    augmentation_hparams: Hparams associated with the NAS learned policy.

  Returns:
    A version of image that now has data augmentation applied to it based on
    the `policies` pass into the function.
  """
  replace_value = augmentation_hparams['replace_const']

  # func is the string name of the augmentation function, prob is the
  # probability of applying the operation and level is the parameter associated
  # with the tf op.

  # tf_policies are functions that take in an image and return an augmented
  # image.
  tf_policies = []
  for policy in policies:
    tf_policy = []
    # Link string name to the correct python function and make sure the correct
    # argument is passed into that function.
    for policy_info in policy:
      policy_info = list(policy_info) + [replace_value, augmentation_hparams]

      tf_policy.append(_parse_policy_info(*policy_info))
    # Now build the tf policy that will apply the augmentation procedue
    # on image.
    def make_final_policy(tf_policy_):
      def final_policy(image_):
        for func, prob, args in tf_policy_:
          image_ = _apply_func_with_prob(
              func, image_, args, prob)
        return image_
      return final_policy
    tf_policies.append(make_final_policy(tf_policy))

  augmented_image = select_and_apply_random_policy(
      tf_policies, image)
  return augmented_image


def distort_image_with_autoaugment(
    image, augmentation_name='v0',
    cutout_const=None, translate_const=None, replace_const=None):
  """Applies the AutoAugment policy to `image`.

  AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.

  Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    augmentation_name: The name of the AutoAugment policy to use. The available
      options are `v0` and `test`. `v0` is the policy used for
      all of the results in the paper and was found to achieve the best results
      on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
      found on the COCO dataset that have slight variation in what operations
      were used during the search procedure along with how many operations are
      applied in parallel to a single image (2 vs 3).
    cutout_const: The cutout patch size is 2 * cutout_const; defaults to
      8 on CIFAR-10 and 100 on ImageNet.
    translate_const: The maximum number of pixels the image is translated by;
        defaults to 32 on CIFAR-10 and 250 on ImageNet.
    replace_const: The constant value to fill empty pixels with;
        defaults to 121 on CIFAR-10 and 128 on ImageNet.
  Returns:
    A tuple containing the augmented versions of `image`.
  """
  if cutout_const is not None and cutout_const < 0:
    raise ValueError('Invalid cutout size')
  if translate_const is not None and translate_const < 0:
    raise ValueError('Invalid translation constant')
  if replace_const is not None and (replace_const < 0 or replace_const > 255):
    raise ValueError('Invalid constant to fill/replace pixels with')

  available_policies = {
      'cifar10': policy_cifar10,
  }
  if augmentation_name not in available_policies:
    raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))

  if augmentation_name == 'cifar10':
    cutout_const = cutout_const or 8
    translate_const = translate_const or 32
    replace_const = replace_const or 121
  else:
    # Defaults to ImageNet augmentation hyper-parameters.
    cutout_const = cutout_const or 100
    translate_const = translate_const or 250
    replace_const = replace_const or 128

  image_dtype = image.dtype
  image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

  policy = available_policies[augmentation_name]()
  augmentation_hparams = {
      'cutout_const': cutout_const,
      'translate_const': translate_const,
      'replace_const': replace_const,
  }
  image = build_and_apply_nas_policy(policy, image, augmentation_hparams)

  image = tf.image.convert_image_dtype(image, dtype=image_dtype)
  return image

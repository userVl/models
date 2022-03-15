# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MobileNet V3[1] + FPN[2] feature extractor for CenterNet[3] meta architecture.

[1]: https://arxiv.org/abs/1905.02244
[2]: https://arxiv.org/abs/1612.03144.
[3]: https://arxiv.org/abs/1904.07850
"""

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import center_net_meta_arch
from object_detection.models.keras_models import mobilenet_v3 as mobilenetv3


_MOBILENET_V3_LARGE_FPN_SKIP_LAYERS = [
    'expanded_conv_2/Add', 'expanded_conv_5/Add', 'expanded_conv_9/Add', 'multiply_19'
]
_NUM_FILTERS_LIST_LARGE = [80, 40, 24]

_MOBILENET_V3_SMALL_FPN_SKIP_LAYERS = [
   'expanded_conv/squeeze_excite/Mul', 'expanded_conv_2/Add', 'expanded_conv_7/Add', 'multiply_17'
]

_NUM_FILTERS_LIST_SMALL = [48, 24, 16]



class CenterNetMobileNetV3FPNFeatureExtractor(
    center_net_meta_arch.CenterNetFeatureExtractor):
  """The MobileNet V3 with FPN skip layers feature extractor for CenterNet."""

  def __init__(self,
               mobilenet_v3_net,
               is_small = False,
               channel_means=(0., 0., 0.),
               channel_stds=(1., 1., 1.),
               bgr_ordering=False,
               use_separable_conv=False,
               upsampling_interpolation='nearest'):
    """Intializes the feature extractor.

    Args:
      mobilenet_v3_net: The underlying mobilenet_v3 network to use.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
      bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.
      use_separable_conv: If set to True, all convolutional layers in the FPN
        network will be replaced by separable convolutions.
      upsampling_interpolation: A string (one of 'nearest' or 'bilinear')
        indicating which interpolation method to use for the upsampling ops in
        the FPN.
    """

    super(CenterNetMobileNetV3FPNFeatureExtractor, self).__init__(
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)
    self._base_model = mobilenet_v3_net

    output = self._base_model(self._base_model.input)

    skip_layer_names = _MOBILENET_V3_LARGE_FPN_SKIP_LAYERS
    num_filters_list = _NUM_FILTERS_LIST_LARGE

    if is_small:
      skip_layer_names = _MOBILENET_V3_SMALL_FPN_SKIP_LAYERS
      num_filters_list = _NUM_FILTERS_LIST_SMALL      


    # Add pyramid feature network on every layer that has stride 2.
    skip_outputs = [
        self._base_model.get_layer(skip_layer_name).output
        for skip_layer_name in skip_layer_names
    ]
    self._fpn_model = tf.keras.models.Model(
        inputs=self._base_model.input, outputs=skip_outputs)
    fpn_outputs = self._fpn_model(self._base_model.input)

    # Construct the top-down feature maps -- we start with an output of
    # 7x7x576 (small) or 7x7x960 (large), which we continually upsample, apply a residual on and merge.
    # This results in a 56x56x16 (small) or 56x56x24 (large) output volume.
    top_layer = fpn_outputs[-1]
    # Use normal convolutional layer since the kernel_size is 1.
    residual_op = tf.keras.layers.Conv2D(
        filters=num_filters_list[0], kernel_size=1, strides=1, padding='same')
    top_down = residual_op(top_layer)

    for i, num_filters in enumerate(num_filters_list):
      level_ind = len(num_filters_list) - 1 - i
      # Upsample.
      upsample_op = tf.keras.layers.UpSampling2D(
          2, interpolation=upsampling_interpolation)
      top_down = upsample_op(top_down)

      # Residual (skip-connection) from bottom-up pathway.
      # Use normal convolutional layer since the kernel_size is 1.
      residual_op = tf.keras.layers.Conv2D(
          filters=num_filters, kernel_size=1, strides=1, padding='same')
      residual = residual_op(fpn_outputs[level_ind])

      # Merge.
      top_down = top_down + residual
      next_num_filters = num_filters_list[i + 1] if i + 1 <= 2 else num_filters_list[-1]
      if use_separable_conv:
        conv = tf.keras.layers.SeparableConv2D(
            filters=next_num_filters, kernel_size=3, strides=1, padding='same')
      else:
        conv = tf.keras.layers.Conv2D(
            filters=next_num_filters, kernel_size=3, strides=1, padding='same')
      top_down = conv(top_down)
      top_down = tf.keras.layers.BatchNormalization()(top_down)
      top_down = tf.keras.layers.ReLU()(top_down)

    output = top_down

    self._feature_extractor_model = tf.keras.models.Model(
        inputs=self._base_model.input, outputs=output)

    

    

  def preprocess(self, resized_inputs):
    resized_inputs = super(CenterNetMobileNetV3FPNFeatureExtractor,
                           self).preprocess(resized_inputs)
    return tf.keras.applications.mobilenet_v3.preprocess_input(resized_inputs)

  def load_feature_extractor_weights(self, path):
    self._base_model.load_weights(path)

  @property
  def classification_backbone(self):
    return self._base_model

  def call(self, inputs):
    return [self._feature_extractor_model(inputs)]

  @property
  def out_stride(self):
    """The stride in the output image of the network."""
    return 4

  @property
  def num_feature_outputs(self):
    """The number of feature outputs returned by the feature extractor."""
    return 1


def mobilenet_v3_large_fpn(channel_means, channel_stds, bgr_ordering,
                     use_separable_conv=False, depth_multiplier=1.0,
                     upsampling_interpolation='nearest', **kwargs):
  """The MobileNetV3Large+FPN backbone for CenterNet."""
  del kwargs

  # Set to batchnorm_training to True for now.
  network = mobilenetv3.mobilenet_v3_large(
      batchnorm_training=True,
      alpha=depth_multiplier,
      include_top=False,
      weights = None)
      #weights='imagenet' if depth_multiplier == 1.0 else None)
  return CenterNetMobileNetV3FPNFeatureExtractor(
      network,
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering,
      use_separable_conv=use_separable_conv,
      upsampling_interpolation=upsampling_interpolation)

def mobilenet_v3_small_fpn(channel_means, channel_stds, bgr_ordering,
                     use_separable_conv=False, depth_multiplier=1.0,
                     upsampling_interpolation='nearest', **kwargs):
  """The MobileNetV3Small+FPN backbone for CenterNet."""
  del kwargs

  # Set to batchnorm_training to True for now.
  network = mobilenetv3.mobilenet_v3_small(
      batchnorm_training=True,
      alpha=depth_multiplier,
      include_top=False,
      weights = None)
      #weights='imagenet' if depth_multiplier == 1.0 else None)
  model = CenterNetMobileNetV3FPNFeatureExtractor(
      network,
      is_small=True,
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering,
      use_separable_conv=use_separable_conv,
      upsampling_interpolation=upsampling_interpolation)

  return model


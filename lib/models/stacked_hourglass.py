# Phi Vision, Inc.
# __________________

# [2020] Phi Vision, Inc.  All Rights Reserved.

# NOTICE:  All information contained herein is, and remains
# the property of Phi Vision Incorporated and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Phi Vision, Inc
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Phi Vision, Inc.
"""
Slim hour glass model for mobile applications
Fanghao Yang 11/25/2020
"""
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from lib.models.blocks import FrontModule, Hourglass, create_front_module, hourglass_module, bottleneck_mobile


class StackedHourglass(Model):
    # TODO: OOP code has issues input shape
    # error: incompatible with the layer: its rank is undefined, but the layer requires a defined rank.
    def get_config(self):
        return {'num_classes': self.num_classes,
                'num_stacks': self.num_stacks,
                'num_features': self.num_features}

    def call(self, inputs, training=None, mask=None):
        # front module, input to 1/4 resolution
        front_features = self._front_module(inputs)

        # form up hourglass stacks and get head of
        # each module for intermediate supervision
        head_next_stage = front_features
        outputs = []
        for i in range(self.num_stacks):
            head_next_stage, head_to_loss = self._hourglasses[i](head_next_stage)
            outputs.append(head_to_loss)
        return outputs

    def __init__(self, num_stacks: int, num_classes: int, num_features: int, mobile=True):
        super(StackedHourglass, self).__init__()
        self._front_module = FrontModule(num_features, mobile=mobile)
        self._hourglasses = [Hourglass(num_classes, num_features, i, mobile=mobile) for i in range(num_stacks)]
        self.num_stacks = num_stacks

    # @classmethod
    # def from_shape_type(cls, num_classes, num_stacks, num_features, input_shape=(256, 256), input_type='rgb'):
    #     if input_type == 'rgb':
    #         tensor_shape = (input_shape[0], input_shape[1], 3)
    #     elif input_type == 'depth':
    #         tensor_shape = (input_shape[0], input_shape[1])
    #     else:
    #         raise TypeError(f"Current model do not support {input_type} as inputs!")
    #     return cls(num_classes, num_stacks, num_features, tensor_shape)


def get_mobile_hg_model(num_classes, num_stacks, num_features, input_size=None, input_type='rgb'):
    """Functional API for mobile hourglass model

    Args:
        num_classes: number of classes of joints (24 by default)
        num_stacks: number of stacks hourglasses in the model (more stacks, more accurate, slower)
        num_features: number of features in input image (256 by default, 128 for tiny model)
        input_size: the shape of input image, shall match the channel number, this argument could be redundant
        input_type: define the type of input data, rgb image, depth map or other input type
                    ['rgb', 'depth', 'rgb-depth', 'grayscale-depth']

    Returns:
        Keras Model of hourglass
    """
    # prepare input tensor
    if input_type == 'rgb':
        num_channels = 3
    else:
        num_channels = 1
    if input_size:
        input_tensor = Input(shape=(input_size[0], input_size[1], num_channels), name='image_input')
    else:
        input_tensor = Input(shape=(None, None, num_channels), name='image_input')

    # front module, input to 1/4 resolution
    front_features = create_front_module(input_tensor, num_features, bottleneck_mobile)

    # form up hourglass stacks and get head of
    # each module for intermediate supervision
    head_next_stage = front_features
    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage,
                                                         num_classes, num_features, bottleneck_mobile, i)
        outputs.append(head_to_loss)

    model = Model(inputs=input_tensor, outputs=outputs)
    return model

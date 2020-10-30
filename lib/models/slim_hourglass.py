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
from lib.models.blocks import create_front_module, hourglass_module, bottleneck_mobile


class SlimHourglass(Model):
    # TODO: OOP code has issues input shape
    # error: incompatible with the layer: its rank is undefined, but the layer requires a defined rank.
    def get_config(self):
        return {'num_classes': self.num_classes,
                'num_stacks': self.num_stacks,
                'num_channels': self.num_channels,
                'input_shape': self.tensor_shape}

    def call(self, inputs, training=None, mask=None):
        # front module, input to 1/4 resolution
        front_features = create_front_module(inputs, self.num_channels, bottleneck_mobile)

        # form up hourglass stacks and get head of
        # each module for intermediate supervision
        head_next_stage = front_features
        self.outputs = []
        for i in range(self.num_stacks):
            head_next_stage, head_to_loss = hourglass_module(head_next_stage,
                                                             self.num_classes,
                                                             self.num_channels,
                                                             bottleneck_mobile,
                                                             i)
            self.outputs.append(head_to_loss)
        return self.outputs

    def __init__(self, num_classes, num_stacks, num_channels, input_shape):
        super(SlimHourglass, self).__init__()
        self.num_channels = num_channels
        self.num_stacks = num_stacks
        self.num_classes = num_classes
        self.tensor_shape = input_shape

    @classmethod
    def from_shape_type(cls, num_classes, num_stacks, num_channels, input_shape=(256, 256), input_type='rgb'):
        if input_type == 'rgb':
            tensor_shape = (input_shape[0], input_shape[1], 3)
        elif input_type == 'depth':
            tensor_shape = (input_shape[0], input_shape[1])
        else:
            raise TypeError(f"Current model do not support {input_type} as inputs!")
        return cls(num_classes, num_stacks, num_channels, tensor_shape)


def get_hourglass_model(num_classes, num_stacks, num_channels, input_size=None):
    """Functional API for mobile hourglass model

    Args:
        num_classes:
        num_stacks:
        num_channels:
        input_size:

    Returns:
        Keras Model of hourglass
    """
    # prepare input tensor
    if input_size:
        input_tensor = Input(shape=(input_size[0], input_size[1], 3), name='image_input')
    else:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    # front module, input to 1/4 resolution
    front_features = create_front_module(input_tensor, num_channels, bottleneck_mobile)

    # form up hourglass stacks and get head of
    # each module for intermediate supervision
    head_next_stage = front_features
    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage,
                                                         num_classes, num_channels, bottleneck_mobile, i)
        outputs.append(head_to_loss)

    model = Model(inputs=input_tensor, outputs=outputs)
    return model

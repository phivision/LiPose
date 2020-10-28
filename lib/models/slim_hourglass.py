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
import os, sys
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from lib.models.blocks import create_front_module, hourglass_module, bottleneck_mobile


class SlimHourglass(Model):
    def __init__(self, num_classes, num_stacks, num_channels, input_shape=(256, 256), input_type='rgb'):
        if input_type == 'rgb':
            input_tensor = Input(shape=(input_shape[0], input_shape[1], 3), name='image_input')
        elif input_type == 'depth':
            input_tensor = Input(shape=input_shape, name='depth_input')
        else:
            raise TypeError(f"Current model do not support {input_type} as inputs!")
        # front module, input to 1/4 resolution
        front_features = create_front_module(input_tensor, num_channels, bottleneck_mobile)

        # form up hourglass stacks and get head of
        # each module for intermediate supervision
        head_next_stage = front_features
        outputs = []
        for i in range(num_stacks):
            head_next_stage, head_to_loss = hourglass_module(head_next_stage,
                                                             num_classes,
                                                             num_channels,
                                                             bottleneck_mobile,
                                                             i)
            outputs.append(head_to_loss)
        super().__init__(inputs=input_tensor, outputs=outputs)

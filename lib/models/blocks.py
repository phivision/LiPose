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
Basic building blocks of hourglass model
Fanghao Yang, 10/25/2020
"""
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, MaxPool2D, Add, UpSampling2D
import tensorflow.keras.backend as k_backend


def hourglass_module(bottom, num_classes, num_channels, bottleneck, hg_id):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(bottom, bottleneck, hg_id, num_channels)

    # create right features, connect with left features
    rf1 = create_right_half_blocks(left_features, bottleneck, hg_id, num_channels)

    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hg_id, num_channels)

    return head_next_stage, head_parts


def bottleneck_block(bottom, num_out_channels, block_name):
    # skip layer
    if k_backend.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                       name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = Conv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name + '_residual')([_skip, _x])

    return _x


def bottleneck_mobile(bottom, num_out_channels, block_name):
    # skip layer
    if k_backend.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                                name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = SeparableConv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                         name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name + '_residual')([_skip, _x])

    return _x


def create_front_module(module_input, num_channels, bottleneck):
    # front module, input to 1/4 resolution
    # 1 7x7 conv + max_pooling
    # 3 residual block

    _x = Conv2D(64,
                kernel_size=(7, 7),
                strides=(2, 2),
                padding='same',
                activation='relu',
                name='front_conv_1x1_x1')(module_input)
    _x = BatchNormalization()(_x)

    _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

    _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
    _x = bottleneck(_x, num_channels, 'front_residual_x3')

    return _x


def create_left_half_blocks(bottom, bottleneck, hg_layer, num_channels):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hg_name = 'hg' + str(hg_layer)

    f1 = bottleneck(bottom, num_channels, hg_name + '_l1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

    f2 = bottleneck(_x, num_channels, hg_name + '_l2')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

    f4 = bottleneck(_x, num_channels, hg_name + '_l4')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

    f8 = bottleneck(_x, num_channels, hg_name + '_l8')

    return f1, f2, f4, f8


def connect_left_to_right(left, right, bottleneck, name, num_channels):
    """

    Args:
        left: connect left feature to right feature
        right:
        bottleneck:
        name: layer name
        num_channels:

    Returns:
    """
    # left -> 1 bottleneck
    # right -> up_sampling
    # Add   -> left + right

    _x_left = bottleneck(left, num_channels, name + '_connect')
    _x_right = UpSampling2D()(right)
    add = Add()([_x_left, _x_right])
    out = bottleneck(add, num_channels, name + '_connect_conv')
    return out


def bottom_layer(lf8, bottleneck, hg_id, num_channels):
    # blocks in lowest resolution
    # 3 bottleneck blocks + Add

    lf8_connect = bottleneck(lf8, num_channels, str(hg_id) + "_lf8")

    _x = bottleneck(lf8, num_channels, str(hg_id) + "_lf8_x1")
    _x = bottleneck(_x, num_channels, str(hg_id) + "_lf8_x2")
    _x = bottleneck(_x, num_channels, str(hg_id) + "_lf8_x3")

    rf8 = Add()([_x, lf8_connect])

    return rf8


def create_right_half_blocks(left_features, bottleneck, hg_layer, num_channels):
    lf1, lf2, lf4, lf8 = left_features

    rf8 = bottom_layer(lf8, bottleneck, hg_layer, num_channels)

    rf4 = connect_left_to_right(lf4, rf8, bottleneck, 'hg' + str(hg_layer) + '_rf4', num_channels)

    rf2 = connect_left_to_right(lf2, rf4, bottleneck, 'hg' + str(hg_layer) + '_rf2', num_channels)

    rf1 = connect_left_to_right(lf1, rf2, bottleneck, 'hg' + str(hg_layer) + '_rf1', num_channels)

    return rf1


def create_heads(pre_layer_features, rf1, num_classes, hg_id, num_channels):
    # two head, one head to next stage, one head to intermediate features
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same', name=str(hg_id) + '_conv_1x1_x1')(
        rf1)
    head = BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    head_parts = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                        name=str(hg_id) + '_conv_1x1_parts')(head)

    # use linear activation
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                  name=str(hg_id) + '_conv_1x1_x2')(head)
    head_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                    name=str(hg_id) + '_conv_1x1_x3')(head_parts)

    head_next_stage = Add()([head, head_m, pre_layer_features])
    return head_next_stage, head_parts


def euclidean_loss(x, y):
    return k_backend.sqrt(k_backend.sum(k_backend.square(x - y)))

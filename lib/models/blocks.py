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
from tensorflow.keras.layers import Layer, Conv2D, SeparableConv2D, BatchNormalization, MaxPool2D, Add, UpSampling2D
import tensorflow.keras.backend as k_backend


class FrontModule(Layer):
    """OOP style definition of the front module"""

    def __init__(self, num_features, block_name='front_module', mobile=True):
        """
        Front Module which convert input to 1/4 resolution, using:
            1 7x7 conv + max_pooling
            3 residual block
        Args:
            num_features: input image features (number of pixels in x or y, e.g. 256)
            block_name:
            mobile: if using mobile network model (separable conv)
        """
        super(FrontModule, self).__init__(name=block_name)
        self._conv2d = Conv2D(64,
                              kernel_size=(7, 7),
                              strides=(2, 2),
                              padding='same',
                              activation='relu',
                              name='front_conv_1x1_x1')
        self._batch_norm = BatchNormalization()
        self._max_pool_2d = MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self._bottleneck_blocks = [Bottleneck(num_features // 2, 'front_residual_x1', mobile=mobile),
                                   Bottleneck(num_features // 2, 'front_residual_x2', mobile=mobile),
                                   Bottleneck(num_features, 'front_residual_x3', mobile=mobile)]

    def call(self, inputs, **kwargs):
        _x = self._conv2d(inputs)
        _x = self._batch_norm(_x)
        _x = self._bottleneck_blocks[0](_x)
        _x = self._max_pool_2d(_x)
        _x = self._bottleneck_blocks[1](_x)
        return self._bottleneck_blocks[2](_x)


class Hourglass(Layer):
    """Single hourglass to be stacked in the full model."""

    def __init__(self, num_classes: int, num_features: int, hg_id: int, mobile=True):
        """

        Args:
            num_classes:
            num_features:
            hg_id: The ID of hour glass block layer
            mobile: if using mobile network with separable conv
        """
        hg_name = 'hg' + str(hg_id)
        super(Hourglass, self).__init__(name=hg_name)
        # create left half blocks for hourglass module
        # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution
        self._lf1 = Bottleneck(num_features, hg_name + '_l1', mobile=mobile)
        self._lf2 = Bottleneck(num_features, hg_name + '_l2', mobile=mobile)
        self._lf4 = Bottleneck(num_features, hg_name + '_l4', mobile=mobile)
        self._lf8 = Bottleneck(num_features, hg_name + '_l8', mobile=mobile)
        self._max_pools = [MaxPool2D(pool_size=(2, 2), strides=(2, 2)) for i in range(3)]
        # create right features, connect with left features
        self._rf8 = Bottom(num_features, hg_id, mobile=mobile)
        self._rf4 = Connector(num_features, 'hg' + str(hg_id) + '_rf4', mobile=mobile)
        self._rf2 = Connector(num_features, 'hg' + str(hg_id) + '_rf2', mobile=mobile)
        self._rf1 = Connector(num_features, 'hg' + str(hg_id) + '_rf1', mobile=mobile)
        # add 1x1 conv with two heads, head_next_stage is sent to next stage
        # head_parts is used for intermediate supervision
        self._heads = HourglassHeads(num_features, num_classes, hg_id)

    def call(self, inputs, **kwargs):
        # create left half blocks
        lf1 = self._lf1(inputs)
        _x = self._max_pools[0](lf1)
        lf2 = self._lf2(_x)
        _x = self._max_pools[1](lf2)
        lf4 = self._lf4(_x)
        _x = self._max_pools[2](lf4)
        lf8 = self._lf8(_x)
        # create right half blocks
        rf8 = self._rf8(lf8)
        rf4 = self._rf4([lf4, rf8])
        rf2 = self._rf2([lf2, rf4])
        rf1 = self._rf1([lf1, rf2])
        return self._heads([inputs, rf1])


class HourglassHeads(Layer):
    """Head for next stage of hourglass and another head for supervision"""

    def __init__(self, num_features: int, num_classes: int, hg_id: int):
        super(HourglassHeads, self).__init__(name='hg' + str(hg_id) + '_head')
        self._inter_feature = InterFeature(num_features, num_classes, hg_id)
        # 1st conv layer is in Intermediate Feature Layer (InterFeature)
        # use linear activation
        self._conv2d_2 = Conv2D(num_features, kernel_size=(1, 1), activation='linear', padding='same',
                                name=str(hg_id) + '_conv_1x1_x2')
        self._conv2d_3 = Conv2D(num_features, kernel_size=(1, 1), activation='linear', padding='same',
                                name=str(hg_id) + '_conv_1x1_x3')
        self._add = Add()

    def call(self, inputs, **kwargs):
        pre_layer_features = inputs[0]
        rf1 = inputs[1]
        head_parts = self._inter_feature(rf1)
        head = self._conv2d_2(rf1)
        head_m = self._conv2d_3(head_parts)
        head_next_stage = self._add([head, head_m, pre_layer_features])
        return head_next_stage, head_parts


class InterFeature(Layer):
    """for head as intermediate supervision, use 'linear' as activation."""

    def __init__(self, num_features: int, num_classes: int, hg_id: int):
        super(InterFeature, self).__init__(name='hg' + str(hg_id) + '_inter_features')
        self._conv2d_1 = Conv2D(num_features,
                                kernel_size=(1, 1),
                                activation='relu',
                                padding='same',
                                name=str(hg_id) + '_conv_1x1_x1')
        self._batch_norm = BatchNormalization()
        self._conv2d_parts = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                                    name=str(hg_id) + '_conv_1x1_parts')

    def call(self, inputs, **kwargs):
        _x = self._conv2d_1(inputs)
        _x = self._batch_norm(_x)
        return self._conv2d_parts(_x)


class Bottleneck(Layer):
    """OOP style bottleneck block"""

    def __init__(self, num_out_features, block_name, mobile=True):
        """
        Dynamically create bottleneck for both local and mobile applications
        Args:
            num_out_features: number of output features
            block_name:
            mobile: if using mobile network with separable conv
        """
        super(Bottleneck, self).__init__(name=block_name)
        self._conv2d_skip = Conv2D(num_out_features, kernel_size=(1, 1), activation='relu', padding='same',
                                   name=block_name + 'skip')
        # residual: 3 conv blocks,  [num_out_features/2  -> num_out_features/2 -> num_out_features]
        if mobile:
            conv_class = SeparableConv2D
        else:
            conv_class = Conv2D
        self._conv2d_blocks = [conv_class(num_out_features // 2,
                                          kernel_size=(1, 1), activation='relu', padding='same',
                                          name=block_name + '_conv_1x1_x1'),
                               conv_class(num_out_features // 2,
                                          kernel_size=(3, 3), activation='relu', padding='same',
                                          name=block_name + '_conv_3x3_x2'),
                               conv_class(num_out_features, kernel_size=(1, 1), activation='relu', padding='same',
                                          name=block_name + '_conv_1x1_x3')]
        self._batch_norm_blocks = [BatchNormalization(), BatchNormalization(), BatchNormalization()]
        self._add = Add(name=block_name + '_residual')
        self.num_out_features = num_out_features

    def call(self, inputs, **kwargs):
        if k_backend.int_shape(inputs)[-1] == self.num_out_features:
            _skip = inputs
        else:
            _skip = self._conv2d_skip(inputs)
        _x = inputs
        num_conv_layers = 3
        for i in range(num_conv_layers):
            _x = self._batch_norm_blocks[i](self._conv2d_blocks[i](_x))
        return self._add([_skip, _x])


class Connector(Layer):
    """Connect left and right pyramid blocks of a hourglass"""

    def __init__(self, num_features: int, block_name: str, mobile=True):
        super(Connector, self).__init__(name=block_name)
        self._x_left = Bottleneck(num_features, block_name + '_connect', mobile=mobile)
        self._x_right = UpSampling2D()
        self._add = Add()
        self._out = Bottleneck(num_features, block_name + '_connect_conv', mobile=mobile)

    def call(self, inputs, **kwargs):
        _left = inputs[0]
        _right = inputs[1]
        _x = self._add([self._x_left(_left), self._x_right(_right)])
        return self._out(_x)


class Bottom(Layer):
    """Blocks in lowest resolution"""

    def __init__(self, num_features: int, hg_id: int, mobile=True):
        super(Bottom, self).__init__(name=str(hg_id) + '_bottom')
        self._lf8_connect = Bottleneck(num_features, str(hg_id) + "_lf8", mobile=mobile)
        bottom_labels = ["_lf8_x1", "_lf8_x2", "_lf8_x3"]
        self._bottleneck_blocks = [Bottleneck(num_features, str(hg_id) + lb, mobile=mobile) for lb in bottom_labels]
        self._add = Add()

    def call(self, inputs, **kwargs):
        lf8_connect = self._lf8_connect(inputs)
        _x = inputs
        for bottleneck in self._bottleneck_blocks:
            _x = bottleneck(_x)
        return self._add([_x, lf8_connect])


# below is code using functional API
def hourglass_module(bottom, num_classes, num_features, bottleneck, hg_id):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(bottom, bottleneck, hg_id, num_features)

    # create right features, connect with left features
    rf1 = create_right_half_blocks(left_features, bottleneck, hg_id, num_features)

    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hg_id, num_features)

    return head_next_stage, head_parts


def bottleneck_block(bottom, num_out_features, block_name):
    """
    Default bottleneck module
    Args:
        bottom:
        num_out_features:
        block_name:

    Returns:

    """
    # skip layer
    if k_backend.int_shape(bottom)[-1] == num_out_features:
        _skip = bottom
    else:
        _skip = Conv2D(num_out_features, kernel_size=(1, 1), activation='relu', padding='same',
                       name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_features/2  -> num_out_features/2 -> num_out_features]
    _x = Conv2D(num_out_features // 2, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_features // 2, kernel_size=(3, 3), activation='relu', padding='same',
                name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_features, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name + '_residual')([_skip, _x])

    return _x


def bottleneck_mobile(bottom, num_out_features, block_name):
    """
    Using separable convolution layers for faster execution on mobile devices
    Args:
        bottom:
        num_out_features:
        block_name:

    Returns:

    """
    # skip layer
    if k_backend.int_shape(bottom)[-1] == num_out_features:
        _skip = bottom
    else:
        _skip = SeparableConv2D(num_out_features, kernel_size=(1, 1), activation='relu', padding='same',
                                name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_features/2  -> num_out_features/2 -> num_out_features]
    _x = SeparableConv2D(num_out_features // 2, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_features // 2, kernel_size=(3, 3), activation='relu', padding='same',
                         name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_features, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name + '_residual')([_skip, _x])

    return _x


def create_front_module(module_input, num_features, bottleneck):
    """
    Front Module which convert input to 1/4 resolution, using:
        1 7x7 conv + max_pooling
        3 residual block
    Args:
        module_input: 
        num_features: input image features (number of pixels in x or y, e.g. 256)
        bottleneck: 

    Returns:

    """
    _x = Conv2D(64,
                kernel_size=(7, 7),
                strides=(2, 2),
                padding='same',
                activation='relu',
                name='front_conv_1x1_x1')(module_input)
    _x = BatchNormalization()(_x)

    _x = bottleneck(_x, num_features // 2, 'front_residual_x1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

    _x = bottleneck(_x, num_features // 2, 'front_residual_x2')
    _x = bottleneck(_x, num_features, 'front_residual_x3')

    return _x


def create_left_half_blocks(bottom, bottleneck, hg_layer, num_features):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hg_name = 'hg' + str(hg_layer)

    f1 = bottleneck(bottom, num_features, hg_name + '_l1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

    f2 = bottleneck(_x, num_features, hg_name + '_l2')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

    f4 = bottleneck(_x, num_features, hg_name + '_l4')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

    f8 = bottleneck(_x, num_features, hg_name + '_l8')

    return f1, f2, f4, f8


def connect_left_to_right(left, right, bottleneck, name, num_features):
    """

    Args:
        left: connect left feature to right feature
        right:
        bottleneck:
        name: layer name
        num_features:

    Returns:
    """
    # left -> 1 bottleneck
    # right -> up_sampling
    # Add   -> left + right

    _x_left = bottleneck(left, num_features, name + '_connect')
    _x_right = UpSampling2D()(right)
    add = Add()([_x_left, _x_right])
    out = bottleneck(add, num_features, name + '_connect_conv')
    return out


def bottom_layer(lf8, bottleneck, hg_id, num_features):
    # blocks in lowest resolution
    # 3 bottleneck blocks + Add

    lf8_connect = bottleneck(lf8, num_features, str(hg_id) + "_lf8")

    _x = bottleneck(lf8, num_features, str(hg_id) + "_lf8_x1")
    _x = bottleneck(_x, num_features, str(hg_id) + "_lf8_x2")
    _x = bottleneck(_x, num_features, str(hg_id) + "_lf8_x3")

    rf8 = Add()([_x, lf8_connect])

    return rf8


def create_right_half_blocks(left_features, bottleneck, hg_layer, num_features):
    lf1, lf2, lf4, lf8 = left_features

    rf8 = bottom_layer(lf8, bottleneck, hg_layer, num_features)

    rf4 = connect_left_to_right(lf4, rf8, bottleneck, 'hg' + str(hg_layer) + '_rf4', num_features)

    rf2 = connect_left_to_right(lf2, rf4, bottleneck, 'hg' + str(hg_layer) + '_rf2', num_features)

    rf1 = connect_left_to_right(lf1, rf2, bottleneck, 'hg' + str(hg_layer) + '_rf1', num_features)

    return rf1


def create_heads(pre_layer_features, rf1, num_classes, hg_id, num_features):
    # two head, one head to next stage, one head to intermediate features
    head = Conv2D(num_features,
                  kernel_size=(1, 1),
                  activation='relu',
                  padding='same',
                  name=str(hg_id) + '_conv_1x1_x1')(rf1)
    head = BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    head_parts = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                        name=str(hg_id) + '_conv_1x1_parts')(head)

    # use linear activation
    head = Conv2D(num_features, kernel_size=(1, 1), activation='linear', padding='same',
                  name=str(hg_id) + '_conv_1x1_x2')(head)
    head_m = Conv2D(num_features, kernel_size=(1, 1), activation='linear', padding='same',
                    name=str(hg_id) + '_conv_1x1_x3')(head_parts)

    head_next_stage = Add()([head, head_m, pre_layer_features])
    return head_next_stage, head_parts


def euclidean_loss(x, y):
    return k_backend.sqrt(k_backend.sum(k_backend.square(x - y)))

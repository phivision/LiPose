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
from lib.models.blocks import FrontModule, Hourglass
from tensorflow.python.eager import backprop


class StackedHourglass(Model):
    """Subclassing model for fully customized network"""

    def __init__(self,
                 num_stacks: int,
                 num_classes: int,
                 num_features: int,
                 num_img_ch: int = 3,
                 mobile=True):
        super(StackedHourglass, self).__init__(name=f"Stacked_{num_stacks}_Hourglass")
        self._front_module = FrontModule(num_features, num_img_ch, mobile=mobile)
        self._hourglasses = [Hourglass(num_classes, num_features, i, mobile=mobile) for i in range(num_stacks)]
        self.num_stacks = num_stacks
        self.num_classes = num_classes
        self.num_features = num_features
        self.mobile = mobile
        # number of image channels, e.g. 3 for rgb, 1 for depth
        self.num_img_channels = num_img_ch

    def get_config(self):
        return {'num_stacks': self.num_stacks,
                'num_classes': self.num_classes,
                'num_features': self.num_features,
                'num_img_ch': self.num_img_channels,
                'mobile': self.mobile}

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

    def train_step(self, data):
        x, y = data
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, None, regularization_losses=self.losses)
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, None)
        return {m.name: m.result() for m in self.metrics}

    def summary(self, line_length=None, positions=None, print_fn=None):
        _x = Input(shape=(self.num_features, self.num_features, self.num_img_channels))
        model = Model(inputs=[_x], outputs=self.call(_x), name=self.name)
        return model.summary(line_length=line_length)

    @classmethod
    def from_shape_type(cls, num_stacks, num_classes, tiny=False, image_type='rgb'):
        """Generate a stacked hourglass network based on input shape and type

        Args:
            num_stacks:
            num_classes:
            tiny: if create tiny model for speed
            image_type: type of input, e.g. 'rgb' for RGB image, 'depth' for depth map

        Returns:

        """
        if image_type == 'rgb':
            num_image_ch = 3
        elif image_type == 'depth':
            num_image_ch = 1
        else:
            raise TypeError(f"Current model do not support {image_type} as inputs!")
        if tiny:
            num_features = 128
        else:
            num_features = 192
        return cls(num_stacks, num_classes, num_features, num_img_ch=num_image_ch, mobile=True)

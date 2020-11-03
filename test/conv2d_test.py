import tensorflow as tf
from lib.models.stacked_hourglass import StackedHourglass
from lib.models.blocks import FrontModule

# The inputs are 28x28 RGB images with `channels_last` and the batch
# size is 4.
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
    2, 3, activation='relu', input_shape=input_shape[1:])(x)
print(y.shape)

input_shape = (1, 256, 256, 3)
front_module = FrontModule(256)
front_module.build(input_shape)
x = tf.random.normal(input_shape)
y = front_module(x)
print(y.shape)

model = StackedHourglass(2, 24, 256)
model.build(input_shape)
y = model(x)
print(y[-1].shape)


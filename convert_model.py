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
Convert trained model to mobile deployment (CoreML, TFLite)
Fanghao Yang, 11/01/2020
"""

import click
from utilities.model_utils import convert_keras_to_coreml
from utilities.misc_utils import parse_image_channel


@click.command()
@click.option('--input_model', type=str, help="Trained model to be converted")
@click.option('--output_file', type=str, default='model/test.mlmodel', help="File path to the output model")
@click.option('--num_features', type=int, default=256, help="Number of features in one dimension of image (resolution)")
@click.option('--image_type', type=str, default='rgb', help="Image type for model input, e.g. rgb, depth")
def convert_model(input_model, output_file, num_features, image_type):
    # Remember the dim of input is 4, which includes the index of input elements
    num_input_ch = parse_image_channel(image_type)
    input_shape = (1, num_features, num_features, num_input_ch)
    if output_file.endswith('.mlmodel'):
        # check if the output format is supported
        convert_keras_to_coreml(input_model, input_shape, output_file)
    else:
        raise TypeError(f"Do not support to export model {input_model} as {output_file}")


if __name__ == '__main__':
    convert_model()

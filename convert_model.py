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


@click.command()
@click.option('--input_model', help="Trained model to be converted")
@click.option('--output_file', default='model/test.mlmodel', help="File path to the output model")
def convert_model(input_model, output_file):
    # Remember the dim of input is 4, which includes the index of input elements
    input_shape = (1, 256, 256, 3)
    convert_keras_to_coreml(input_model, input_shape, output_file)


if __name__ == '__main__':
    convert_model()

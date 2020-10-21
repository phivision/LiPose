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
load SURREAL serialized TFRecords format
By Fanghao Yang, 10/15/2020
"""

import click
from datasets.dataset_loader import load_surreal_data
from pathlib import Path
import tensorflow as tf
import pylab


@click.command()
@click.option('--input_file', help='input TFRecords file')
def load_surreal_test(input_file: str):
    dataset = load_surreal_data(Path(input_file))
    for example in dataset.take(1):
        rgb_map = tf.io.parse_tensor(example['rgb'], out_type=tf.uint8)
        pylab.figure()
        pylab.imshow(rgb_map)
        print(f"rgb image shape {rgb_map.shape}")
    pylab.show()


if __name__ == '__main__':
    load_surreal_test()

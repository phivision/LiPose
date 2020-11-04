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
load SURREAL serialized TFRecords format for training
By Fanghao Yang, 10/15/2020
"""

import click
from datasets.dataset_loader import load_surreal_data_training, parse_tfr_tensor
from pathlib import Path
import pylab
import numpy as np


@click.command()
@click.option('--input_file', help='input TFRecords file')
def load_training_data_test(input_file: str):
    dataset = load_surreal_data_training(Path(input_file), 1, image_type='rgb')
    # count the total number of examples
    count = int(dataset.reduce(np.int64(0), lambda x, _: x + 1))
    print(f"Total {count} examples in dataset: {input_file}")
    for rgb, heat_map in dataset.take(3):
        pylab.figure()
        pylab.imshow(rgb[0]/255.0)
        print(f"rgb image shape {rgb.shape}")
        # list heat maps
        heat_maps = [heat_map[0, :, :, i] for i in range(heat_map.shape[-1])]
        pylab.figure()
        blended_map = np.zeros(heat_maps[0].shape)
        for hm in heat_maps:
            blended_map += hm
        pylab.imshow(blended_map)
    pylab.show()


if __name__ == '__main__':
    load_training_data_test()

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
from datasets.dataset_loader import load_surreal_data, parse_tfr_tensor
from pathlib import Path
import pylab
import numpy as np


@click.command()
@click.option('--input_file', help='input TFRecords file')
def load_surreal_test(input_file: str):
    dataset = load_surreal_data(Path(input_file))
    for element in dataset.take(5):
        example = parse_tfr_tensor(element)
        pylab.figure()
        pylab.imshow(example['rgb'])
        print(f"rgb image shape {example['rgb'].shape}")
        pylab.figure()
        depth_map = example['depth'].numpy()
        depth_map[depth_map > 100] = 0.0
        pylab.imshow(depth_map)
        # list heat maps
        heat_maps = [example['heat_map'][:, :, i] for i in range(example['heat_map'].shape[-1])]
        # for heat_map in heat_maps:
        #     pylab.figure()
        #     pylab.imshow(heat_map)
        pylab.figure()
        blended_map = np.zeros(heat_maps[0].shape)
        for heat_map in heat_maps:
            blended_map += heat_map
        pylab.imshow(blended_map)
    pylab.show()


if __name__ == '__main__':
    load_surreal_test()

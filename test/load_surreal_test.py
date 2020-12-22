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
from datasets.dataset_loader import load_dataset, parse_tfr_tensor
from datasets.dataset_converter import TARGET_MAX_DEPTH
from utilities.image_utils import generate_blended_heatmap
from pathlib import Path
import pylab


@click.command()
@click.option('--input_file', help='input TFRecords file')
@click.option('--image_type', default='rgb', help='Type of image loading from the dataset')
def load_surreal_test(input_file: str, image_type: str):
    dataset = load_dataset(Path(input_file), image_type=image_type)
    for element in dataset.take(5):
        example = parse_tfr_tensor(element)
        print(f"Loading data for image: {example['name'].numpy().decode('ascii')}")
        pylab.figure()
        pylab.imshow(example['rgb'])
        print(f"rgb image shape {example['rgb'].shape}")
        pylab.figure()
        depth_map = example['depth'].numpy()
        pylab.imshow(depth_map / TARGET_MAX_DEPTH)
        # list heat maps
        blended_map = generate_blended_heatmap(example['heat_map'])
        # for heat_map in heat_maps:
        #     pylab.figure()
        #     pylab.imshow(heat_map)
        pylab.figure()
        pylab.imshow(blended_map)
    pylab.show()


if __name__ == '__main__':
    load_surreal_test()

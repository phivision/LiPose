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
Convert SURREAL raw data from internet to serialized TFRecords format
By Fanghao Yang, 10/15/2020
"""

import click
from datasets.dataset_converter import convert_surreal_data
from pathlib import Path


@click.command()
@click.option('--input_dir', help='input data to be converted')
@click.option('--output_file', help='output TFRecord data')
def convert_surreal(input_dir: str, output_file: str):
    convert_surreal_data(Path(input_dir), Path(output_file))


if __name__ == '__main__':
    convert_surreal()

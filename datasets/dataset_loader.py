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
Load custom data from TFRecords data

By Fanghao Yang, 10/15/2020
"""

import tensorflow as tf
from pathlib import Path


def _parse_tfr_element(element):
    parse_dic = {
        # Note that it is tf.string, not tf.float32
        'rgb': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.string),
        'joints_2d': tf.io.FixedLenFeature([], tf.string),
        'joints_3d': tf.io.FixedLenFeature([], tf.string),
        'cam_loc': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'frame_index': tf.io.FixedLenFeature([], tf.int64)}
    example_message = tf.io.parse_single_example(element, parse_dic)
    #
    # b_feature = example_message['b_feature'] # get byte string
    # feature = tf.io.parse_tensor(b_feature, out_type=tf.float64) # restore 2D array from byte string
    return example_message


def load_surreal_data(data_path: Path):
    tfr_data = tf.data.TFRecordDataset(str(data_path))
    dataset = tfr_data.map(_parse_tfr_element)
    return dataset

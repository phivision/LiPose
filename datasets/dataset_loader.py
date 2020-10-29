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
Load custom datasets from TFRecords data files
By Fanghao Yang, 10/15/2020
"""

import tensorflow as tf
from pathlib import Path

AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_tfr_tensor(element):
    """
    Parse all TFRecord data as a dict
    Args:
        element: an element of dataset

    Returns:
        a dict contains all data
    """
    new_element = {
        'rgb': tf.io.parse_tensor(element['rgb'], out_type=tf.uint8),
        'depth': tf.io.parse_tensor(element['depth'], out_type=tf.float32),
        'joints_2d': tf.io.parse_tensor(element['joints_2d'], out_type=tf.float32),
        'joints_3d': tf.io.parse_tensor(element['joints_3d'], out_type=tf.float32),
        'heat_map': tf.io.parse_tensor(element['heat_map'], out_type=tf.float32),
        'cam_loc': tf.io.parse_tensor(element['cam_loc'], out_type=tf.float32),
        'name': tf.strings.unicode_decode(element['name'], input_encoding='UTF-8'),
        'frame_index': element['frame_index'],
        'crop_box': element['crop_box']
    }
    return new_element


def _parse_tfr_element(element):
    """
    Deserialize TFRecord as a dict, numpy array is parsed as byte string. This is for debugging and evaluation.
    Args:
        element: an element in raw TFRecord dataset

    Returns:
        a dict needs further parse for full data
    """
    parse_dict = {
        # Note that it is tf.string, not tf.float32
        'rgb': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.string),
        'joints_2d': tf.io.FixedLenFeature([], tf.string),
        'joints_3d': tf.io.FixedLenFeature([], tf.string),
        'heat_map': tf.io.FixedLenFeature([], tf.string),
        'cam_loc': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'frame_index': tf.io.FixedLenFeature([], tf.int64),
        'crop_box': tf.io.FixedLenFeature([4], tf.int64)}
    example_message = tf.io.parse_single_example(element, parse_dict)
    return example_message


def _parse_tfr_rgb_training(element):
    """
    Parse TFRecord for rgb model training, which only generate partial data
    Args:
        element: an element in raw TFRecord dataset

    Returns:
        rgb tensor, heat map tensor
    """
    example_message = _parse_tfr_element(element)
    rgb = tf.io.parse_tensor(example_message['rgb'], out_type=tf.uint8)
    heatmap = tf.io.parse_tensor(example_message['heat_map'], out_type=tf.float32)
    return rgb, heatmap


def _load_tfr_data(data_path: Path):
    # ignore order of data to speed up loading
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    tfr_data = tf.data.TFRecordDataset(str(data_path))
    tfr_data = tfr_data.with_options(ignore_order)
    return tfr_data


def load_full_surreal_data(data_path: Path):
    """
    Load SURREAL TFRecord data and parsed as a dict a dict
    Args:
        data_path: file path to TFRecord

    Returns:
        a dataset mapped with TFR parser
    """
    tfr_data = _load_tfr_data(data_path)
    dataset = tfr_data.map(_parse_tfr_element, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def load_surreal_data_training(data_path: Path, batch_size: int, shuffle: bool = True, model_type: str = 'rgb'):
    tfr_data = _load_tfr_data(data_path)
    if model_type == 'rgb':
        dataset = tfr_data.map(_parse_tfr_rgb_training, num_parallel_calls=AUTOTUNE)
    elif model_type == 'depth':
        # TODO: finish depth training data parser
        dataset = tfr_data.map(_parse_tfr_rgb_training, num_parallel_calls=AUTOTUNE)
    else:
        raise TypeError(f"Do not support load training data for {model_type} model!")
    if shuffle:
        dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return dataset

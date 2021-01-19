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
from functools import partial
from utilities.misc_utils import parse_image_channel

AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_tfr_tensor(element, image_type='rgb', support_3d=False):
    """
    Parse all TFRecord data as a dict
    Args:
        element: an element of dataset
        image_type: type of tensor image data
        support_3d: support training / evaluation 3d model

    Returns:
        a dict contains all data
    """
    if image_type == 'rgb':
        new_element = {'rgb': tf.io.parse_tensor(element['rgb'], out_type=tf.uint8)}
    elif image_type == 'depth':
        # insert image channel index to the shape of depth map
        new_element = {'depth': tf.expand_dims(tf.io.parse_tensor(element['depth'], out_type=tf.float32), -1)}
    else:
        raise TypeError(f"Do not support image type {image_type}")
    if support_3d:
        new_element['joints_3d'] = tf.io.parse_tensor(element['joints_3d'], out_type=tf.float32)
    new_element['joints_2d'] = tf.io.parse_tensor(element['joints_2d'], out_type=tf.float32)
    new_element['heat_map'] = tf.io.parse_tensor(element['heat_map'], out_type=tf.float32)
    new_element['cam_loc'] = tf.io.parse_tensor(element['cam_loc'], out_type=tf.float32)
    new_element['name'] = element['name']
    new_element['frame_index'] = element['frame_index']
    new_element['crop_box'] = element['crop_box']
    return new_element


def _parse_tfr_element(element, image_type='rgb', support_3d=False):
    """
    Deserialize TFRecord as a dict, numpy array is parsed as byte string. This is for debugging and evaluation.
    Args:
        element: an element in raw TFRecord dataset
        image_type: type of tensor image data
        support_3d: support training / evaluation 3d model

    Returns:
        a dict needs further parse for full data
    """
    parse_dict = {image_type: tf.io.FixedLenFeature([], tf.string), 'joints_2d': tf.io.FixedLenFeature([], tf.string),
                  'heat_map': tf.io.FixedLenFeature([], tf.string), 'cam_loc': tf.io.FixedLenFeature([], tf.string),
                  'name': tf.io.FixedLenFeature([], tf.string), 'frame_index': tf.io.FixedLenFeature([], tf.int64),
                  'crop_box': tf.io.FixedLenFeature([4], tf.int64)}
    if support_3d:
        parse_dict['joints_3d'] = tf.io.FixedLenFeature([], tf.string)
    example_message = tf.io.parse_single_example(element, parse_dict)
    return example_message


def _parse_tfr_training(element, image_type='rgb', num_features=256):
    """
    Parse TFRecord for model training, which only generate partial data
    Args:
        element: an element in raw TFRecord dataset
        image_type: parse the serialized data based on the type for model input
        num_features: number of features (pixel resolution) of image data

    Returns:
        rgb tensor, heat map tensor
    """
    example_message = _parse_tfr_element(element, image_type=image_type)
    if image_type == 'rgb':
        image = tf.io.parse_tensor(example_message[image_type], out_type=tf.uint8)
    elif image_type == 'depth':
        image = tf.io.parse_tensor(example_message[image_type], out_type=tf.float32)
    else:
        raise TypeError(f"Do not support image type {image_type}")
    num_input_ch = parse_image_channel(image_type)
    heatmap = tf.io.parse_tensor(example_message['heat_map'], out_type=tf.float32)
    # the TFRecord dataset generator cannot explicitly generate datatype and cast data
    # we have to explicitly do it to match the input shape and datatype
    image = tf.cast(tf.reshape(image, (num_features, num_features, num_input_ch)), dtype=tf.float32)
    return image, heatmap


def _load_tfr_data(data_path: Path):
    # ignore order of data to speed up loading
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    if data_path.is_dir():
        record_list = [str(tfrecord) for tfrecord in data_path.glob('*.tfrecord')]
        tfr_data = tf.data.TFRecordDataset(record_list)
    else:
        tfr_data = tf.data.TFRecordDataset(str(data_path))
    tfr_data = tfr_data.with_options(ignore_order)
    return tfr_data


def load_dataset(data_path: Path, image_type='rgb', support_3d=False):
    """
    Load TFRecord dataset and parsed as a dict
    Args:
        data_path: file path to TFRecord
        image_type: type of tensor image data
        support_3d: support training / evaluation 3d model

    Returns:
        a dataset mapped with TFR parser
    """
    tfr_data = _load_tfr_data(data_path)
    parse_func = partial(_parse_tfr_element, image_type=image_type, support_3d=support_3d)
    dataset = tfr_data.map(parse_func, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def load_data_training(data_path: Path,
                       batch_size: int,
                       shuffle: bool = True,
                       num_features: int = 256,
                       image_type: str = 'rgb'):
    tfr_data = _load_tfr_data(data_path)
    parse_func = partial(_parse_tfr_training, image_type=image_type, num_features=num_features)
    dataset = tfr_data.map(parse_func, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return dataset

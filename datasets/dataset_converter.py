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
Convert custom data from existing datasets to serialized TFRecord data.
For faster data loading in training.
By Fanghao Yang, 10/15/2020
"""

import scipy.io as sio
import tensorflow as tf
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path
import glob

# since the uncompressed data is too large, we use a step to sample each 10 frames
FRAME_STEP = 101
# to distinguish human body from background, this threshold value is used
DEPTH_THRESHOLD = 1000
# new input image height and width
IMG_HEIGHT = 256
IMG_WIDTH = 256
# heat map size
HEAT_MAP_HEIGHT = 64
HEAT_MAP_WIDTH = 64


# Utility functions to serialize numpy array as byte string
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(value):
    """Returns an int32_list from a list of ints"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


# Converter functions
def get_surreal_video(input_dir: Path, name: str):
    mp4_file = input_dir.joinpath(name + '.mp4')
    vid = imageio.get_reader(mp4_file, 'ffmpeg')
    return vid


def get_surreal_depth(input_dir: Path, name: str):
    depth_file = input_dir.joinpath(name + '_depth.mat')
    with depth_file.open(mode='r'):
        depth_data = sio.loadmat(str(depth_file))
    return depth_data


def get_surreal_info(input_dir: Path, name: str):
    info_file = input_dir.joinpath(name + '_info.mat')
    with info_file.open(mode='r'):
        info_data = sio.loadmat(str(info_file))
    return info_data


def _make_gaussian(height, width, sigma=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square sigma is full-width-half-maximum,
    which can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def _generate_2d_crop_box(height, width, depth, box_pad_rate=0.05):
    """Automatically returns a padding vector and a bounding box given
    the size of the image and a list of joints.
    Args:
        height: Original Height
        width: Original Width
        depth: Depth map
        box_pad_rate: Box percentage (Use 20% to get a good bounding box)

    Returns:
        padding vector, crop box [center_w, center_h, width, height]
    """
    padding = [[0, 0], [0, 0]]
    # generate bbox from depth map
    human_depth = np.where(depth < DEPTH_THRESHOLD)
    box = (np.min(human_depth[1]), np.min(human_depth[0]), np.max(human_depth[1]), np.max(human_depth[0]))
    # extend bbox with padding
    crop_box = [box[0] - int(box_pad_rate * (box[2] - box[0])), box[1] - int(box_pad_rate * (box[3] - box[1])),
                box[2] + int(box_pad_rate * (box[2] - box[0])), box[3] + int(box_pad_rate * (box[3] - box[1]))]
    crop_box[0] = 0 if crop_box[0] < 0 else crop_box[0]
    crop_box[1] = 0 if crop_box[1] < 0 else crop_box[1]
    crop_box[2] = width - 1 if crop_box[2] > width - 1 else crop_box[2]
    crop_box[3] = height - 1 if crop_box[3] > height - 1 else crop_box[3]
    new_h = int(crop_box[3] - crop_box[1])
    new_w = int(crop_box[2] - crop_box[0])
    crop_box = [crop_box[0] + new_w // 2, crop_box[1] + new_h // 2, new_w, new_h]
    if new_h > new_w:
        bounds = (crop_box[0] - new_h // 2, crop_box[0] + new_h // 2)
        if bounds[0] < 0:
            padding[1][0] = abs(bounds[0])
        if bounds[1] > width - 1:
            padding[1][1] = abs(width - bounds[1])
    elif new_h < new_w:
        bounds = (crop_box[1] - new_w // 2, crop_box[1] + new_w // 2)
        if bounds[0] < 0:
            padding[0][0] = abs(bounds[0])
        if bounds[1] > width - 1:
            padding[0][1] = abs(height - bounds[1])
    crop_box[0] += padding[1][0]
    crop_box[1] += padding[0][0]
    return padding, crop_box


def relative_joints(box, padding, joints_2d, to_size=64):
    """ Convert Absolute joint coordinates to crop box relative joint coordinates
        Used to compute Heat Maps)
        Args:
            box: Bounding Box
            padding: Padding Added to the original Image
            joints_2d: 2d coordinate array of joints, shape [2, num of joints]
            to_size: Heat Map wanted Size
    """
    new_j = np.copy(joints_2d.T)
    max_l = max(box[2], box[3])
    new_j = new_j + [padding[1][0], padding[0][0]]
    new_j = new_j - [box[0] - max_l // 2, box[1] - max_l // 2]
    new_j = new_j * to_size / (max_l + 0.0000001)
    return new_j.astype(np.int32).T


def _generate_2d_heat_map(height, width, joints_2d, max_length):
    """ Generate a full Heap Map for every joints in an array

    Args:
        height: Height for the Heat Map
        width: Width for the Heat Map
        joints_2d: Array of Joints
        max_length: Length of the Bounding Box

    Returns:
        heat map
    """
    num_joints = joints_2d.shape[1]
    hm = np.zeros((height, width, num_joints), dtype=np.float32)
    for i in range(num_joints):
        s = int(np.sqrt(max_length) * max_length * 10 / 4096) + 2
        hm[:, :, i] = _make_gaussian(height, width, sigma=s, center=(joints_2d[0, i], joints_2d[1, i]))
    return hm


def _generate_new_depth(new_height, new_width, raw_depth):
    """Generate a resized new depth image from raw data
    This code ONLY works for converting 240x320 data to 256x256 data!!!
    Args:
        new_height: height of new depth map
        new_width: width of new depth map
        raw_depth: raw depth map

    Returns:
        new depth map, new and raw bbox of human in original depth map
    """
    # generate bbox from depth map
    # human_depth = np.where(raw_depth < DEPTH_THRESHOLD)
    # bounding box = (start width, start height, end width, end height)
    # bbox = [np.min(human_depth[1]), np.min(human_depth[0]), np.max(human_depth[1]), np.max(human_depth[0])]
    # human shape (human width, human height)
    # human_shape = (bbox[2]-bbox[0], bbox[3]-bbox[1])
    new_map = np.full((new_height, new_width), 1000000, dtype=np.float32)
    # new_bbox = [(new_width - human_shape[0]) // 2,
    #             (new_height - human_shape[1]) // 2,
    #             (new_height + human_shape[0]) // 2,
    #             (new_width + human_shape[0]) // 2]
    # new_bbox[0] = 0 if new_bbox[0] < 0 else new_bbox[0]
    # new_bbox[1] = 0 if new_bbox[1] < 0 else new_bbox[1]
    # new_map[new_bbox[1]:(new_bbox[1] + human_shape[1]), new_bbox[0]:(new_bbox[0] + human_shape[0])] = \
    #     raw_depth[bbox[1]:bbox[3], bbox[2]:bbox[0]]
    # shift = [(new_bbox[0] + new_bbox[2]) / 2 - (bbox[0] + bbox[2]) / 2,
    #          (new_bbox[1] + new_bbox[1]) / 2 - (bbox[0] + bbox[2]) / 2]
    raw_height = raw_depth.shape[0]
    raw_width = raw_depth.shape[1]
    shift = [(new_width - raw_width)//2, (new_height - raw_height)//2]
    new_map[shift[1]:shift[1]+raw_height, :] = raw_depth[:, -shift[0]:-shift[0]+new_width]
    return new_map, shift


def _generate_new_rgb(new_height, new_width, shift, raw_rgb):
    """Generate a resized new rgb image from raw data
    This code ONLY works for converting 240x320 data to 256x256 data!!!
    Args:
        new_height: height of new rgb map
        new_width: width of new rgb map
        raw_rgb: raw rgb image
        shift:

    Returns:
        new rgb
    """
    new_rgb = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    raw_height = raw_rgb.shape[0]
    new_rgb[shift[1]:shift[1]+raw_height, :] = raw_rgb[:, -shift[0]:-shift[0]+new_width]
    return new_rgb


def _generate_new_joints_2d(raw_joints_2d, shift):
    num_joints = raw_joints_2d.shape[1]
    new_joints_2d = np.zeros(raw_joints_2d.shape, dtype=np.float32)
    for i in range(num_joints):
        new_joints_2d[0, i] = raw_joints_2d[0, i] + shift[0]
        new_joints_2d[1, i] = raw_joints_2d[1, i] + shift[1]
    return new_joints_2d


def convert_surreal_data(input_path: Path, output_path: Path, max_count=1000000):
    """Convert SURREAL dataset to TFRecord serialized data

    Args:
        input_path: input path of dataset
        output_path: output path of TFRecord file
        max_count: maximum count of converted examples

    Returns:
        None
    """
    if output_path.suffix != '.tfrecord':
        print("The output format is not supported, please use TFRecord format!")
        return
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        # find all video files in the input dir
        mp4_files = tqdm(glob.glob(str(input_path.joinpath('**/*.mp4')), recursive=True))
        conversion_count = 0
        for vid_file in mp4_files:
            if conversion_count >= max_count:
                # stop conversion since it reach max count
                return
            vid_file = Path(vid_file)
            vid_dir = vid_file.parent
            vid_name = vid_file.stem
            mp4_files.set_postfix_str(f"Video Name: {vid_name}")
            video = get_surreal_video(vid_dir, vid_name)
            depth_data = get_surreal_depth(vid_dir, vid_name)
            info_data = get_surreal_info(vid_dir, vid_name)
            num_frames = info_data['shape'].shape[1]
            if num_frames != (len(depth_data) - 3):
                raise ValueError("length of depth map and joint data does not match!")
            for i in range(0, num_frames, FRAME_STEP):
                raw_rgb_image = video.get_data(i)
                raw_depth_map = depth_data['depth_' + str(i + 1)]
                depth_map, shift = _generate_new_depth(IMG_HEIGHT, IMG_WIDTH, raw_depth_map)
                rgb_image = _generate_new_rgb(IMG_HEIGHT, IMG_WIDTH, shift, raw_rgb_image)
                if num_frames == 1:
                    raw_joints_2d = info_data['joints2D']
                    joints_3d = info_data['joints3D']
                else:
                    raw_joints_2d = info_data['joints2D'][:, :, i]
                    joints_3d = info_data['joints3D'][:, :, i]
                cam_loc = info_data['camLoc']
                joints_2d = _generate_new_joints_2d(raw_joints_2d, shift)
                # generate heat map array
                pad_vec, c_box = _generate_2d_crop_box(IMG_HEIGHT, IMG_WIDTH, depth_map)
                resized_joints_2d = relative_joints(c_box, pad_vec, joints_2d, to_size=HEAT_MAP_HEIGHT)
                heat_map = _generate_2d_heat_map(HEAT_MAP_HEIGHT, HEAT_MAP_WIDTH, resized_joints_2d, HEAT_MAP_WIDTH)
                feature = {'rgb': _bytes_feature(serialize_array(rgb_image)),
                           'depth': _bytes_feature(serialize_array(depth_map)),
                           'heat_map': _bytes_feature(serialize_array(heat_map)),
                           'joints_2d': _bytes_feature(serialize_array(joints_2d)),
                           'joints_3d': _bytes_feature(serialize_array(joints_3d)),
                           'cam_loc': _bytes_feature(serialize_array(cam_loc)),
                           'name': _bytes_feature(vid_name.encode('utf-8')),
                           'frame_index': _int64_feature(i),
                           'crop_box': _int64_feature_list(c_box)}
                example_message = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_message.SerializeToString())
                conversion_count += 1

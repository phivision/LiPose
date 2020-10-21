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
Convert custom data from existing datasets to serialized TFRecord data
For faster data loading in training
By Fanghao Yang, 10/15/2020
"""

import scipy.io as sio
import tensorflow as tf
import imageio
from tqdm import tqdm
from pathlib import Path
import glob

# since the uncompressed data is too large, we use a step to sample each 10 frames
FRAME_STEP = 10


# Utility functions to serialize numpy array as byte string
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


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
        depth_data = sio.loadmat(depth_file)
    return depth_data


def get_surreal_info(input_dir: Path, name: str):
    info_file = input_dir.joinpath(name + '_info.mat')
    with info_file.open(mode='r'):
        info_data = sio.loadmat(info_file)
    return info_data


def convert_surreal_data(input_path: Path, output_path: Path, ):

    if output_path.suffix != '.tfrecord':
        print("The output format is not supported, please use TFRecord format!")
        return
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        # find all video files in the input dir
        mp4_files = tqdm(glob.glob(str(input_path.joinpath('**/*.mp4')), recursive=True))
        for vid_file in mp4_files:
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
                rgb_image = video.get_data(i)
                depth_map = depth_data['depth_'+str(i+1)]
                if num_frames == 1:
                    joints_2d = info_data['joints2D']
                    joints_3d = info_data['joints3D']
                else:
                    joints_2d = info_data['joints2D'][:, :, i]
                    joints_3d = info_data['joints3D'][:, :, i]
                cam_loc = info_data['camLoc']
                feature = {'rgb': _bytes_feature(serialize_array(rgb_image)),
                           'depth': _bytes_feature(serialize_array(depth_map)),
                           'joints_2d': _bytes_feature(serialize_array(joints_2d)),
                           'joints_3d': _bytes_feature(serialize_array(joints_3d)),
                           'cam_loc': _bytes_feature(serialize_array(cam_loc)),
                           'name': _bytes_feature(vid_name.encode('utf-8')),
                           'frame_index': _int64_feature(i)}
                example_message = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_message.SerializeToString())





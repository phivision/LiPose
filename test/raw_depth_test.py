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
load raw unlabeled lidar data to test the CoreML model and visualize it
By Fanghao Yang, 11/15/2020
"""
import os
import click
import pylab
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
from utilities.model_utils import load_eval_model
from eval import hourglass_predict_coreml
from lib.postprocessing import post_process_heatmap
from lib.visualization import draw_joints_on_depth
from utilities.misc_utils import get_classes
from datasets.dataset_converter import generate_new_depth
DEPTH_HEIGHT = 256
DEPTH_WIDTH = 256
PRED_THRESHOLD = 0.3


@click.command()
@click.option('--input_file', help='input depth data .npy or .mat file')
@click.option('--model_path', help="model used to test raw data")
@click.option('--joint_list', default='datasets/surreal/joint_list.txt', help="a list of joint names in txt file")
def raw_depth_test(input_file: str, model_path: str, joint_list: str):
    with open(input_file, 'rb') as depth_file:
        if input_file.endswith('.npy'):
            depth_data = np.load(depth_file.name)
            resized_depth = zoom(depth_data, 0.5, order=1)
            print(f"Shape of resized depth map {resized_depth.shape}")
            assert (resized_depth.shape[1] <= DEPTH_WIDTH)
            assert (resized_depth.shape[0] <= DEPTH_HEIGHT)
            # padding the input to get standard size
            shift = (DEPTH_HEIGHT - resized_depth.shape[0]) // 2
            std_input = np.pad(resized_depth[:-1, :], ((shift, shift+1), (0, 0)), constant_values=np.max(resized_depth))
            std_input = np.expand_dims(std_input, axis=2)
            print(f"Shape of padded depth map {std_input.shape}")
        elif input_file.endswith('.mat'):
            depth_data = sio.loadmat(depth_file.name)['depth_1']
            std_input, _ = generate_new_depth(DEPTH_HEIGHT, DEPTH_WIDTH, depth_data)
            std_input = np.expand_dims(std_input, axis=2)
            print(f"Shape of depth map {std_input.shape}")
        else:
            raise TypeError("Do not support input file format!")

        # test raw data with model
        model, _ = load_eval_model(model_path)
        heatmaps = hourglass_predict_coreml(model, std_input)
        # get predict joints from heatmap
        pred_joints = post_process_heatmap(heatmaps, 0.3)
        print(pred_joints)
        # get a list of joint names
        work_dir = os.getcwd()
        project_dir = os.path.dirname(work_dir)
        joint_names = get_classes(os.path.join(project_dir, joint_list))
        heatmap_width = DEPTH_WIDTH/heatmaps.shape[1]
        draw_joints_on_depth(pred_joints, joint_names, std_input, heatmap_width)

    pylab.show()


if __name__ == '__main__':
    raw_depth_test()
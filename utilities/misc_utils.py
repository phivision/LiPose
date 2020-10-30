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
Miscellaneous utility functions.
Fanghao Yang 10/28/2020
"""
import cv2
import os


def optimize_tf_gpu(tensor_flow, backend):
    if tensor_flow.__version__.startswith('2'):
        gpus = tensor_flow.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    # pay careful attention to the usage of GPU memory, over claim of memory will lead to problems
                    config = [tensor_flow.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)]
                    tensor_flow.config.experimental.set_virtual_device_configuration(gpu, config)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        config = tensor_flow.ConfigProto()
        config.gpu_options.allow_growth = True   # dynamic alloc GPU resource
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # GPU memory threshold 0.3
        session = tensor_flow.Session(config=config)
        # set session
        backend.set_session(session)


def touch_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_classes(classes_path):
    """loads the classes

    Args:
        classes_path:

    Returns:

    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_model_type(num_stacks, mobile, tiny, input_size):
    mobile_str = 'mobile_' if mobile else ''
    tiny_str = 'tiny_' if tiny else ''

    model_type = 'hg_s' + str(num_stacks) + '_' \
                 + mobile_str \
                 + tiny_str \
                 + str(input_size[0]) + '_' + str(input_size[1])
    return model_type


def get_skeleton(skeleton_path):
    """loads the skeleton

    Args:
        skeleton_path:

    Returns:

    """
    with open(skeleton_path) as f:
        skeleton_lines = f.readlines()
    skeleton_lines = [s.strip() for s in skeleton_lines]
    return skeleton_lines


def get_match_points(match_point_path):
    """loads the matched keypoints

    Args:
        match_point_path:

    Returns:

    """
    with open(match_point_path) as f:
        match_point_lines = f.readlines()
    match_point_lines = [s.strip() for s in match_point_lines]
    return match_point_lines


def render_skeleton(image, keypoints_dict, skeleton_lines=None, conf_threshold=0.001, colors=None):
    """
    Render keypoints skeleton on provided image with
    keypoints dict and skeleton lines definition.
    If no skeleton_lines provided, we'll only render
    keypoints.
    """
    def get_color(color_pattern):
        if color_pattern == 'r':
            base_color = (255, 0, 0)
        elif color_pattern == 'g':
            base_color = (0, 255, 0)
        elif color_pattern == 'b':
            base_color = (0, 0, 255)
        else:
            raise ValueError('invalid color pattern')
        return base_color

    def draw_line(img, start_point, end_point, line_color=(255, 0, 0)):
        x_start, y_start, conf_start = start_point
        x_end, y_end, conf_end = end_point

        if (x_start > 1 and y_start > 1 and conf_start > conf_threshold) and \
                (x_end > 1 and y_end > 1 and conf_end > conf_threshold):
            cv2.circle(img, center=(int(x_start), int(y_start)), color=line_color, radius=3, thickness=-1)
            cv2.circle(img, center=(int(x_end), int(y_end)), color=line_color, radius=3, thickness=-1)
            cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), color=line_color, thickness=1)
        return img

    def draw_keypoints(img, key_points, line_color):
        for key_point in key_points:
            x, y, conf = key_point
            if x > 1 and y > 1 and conf > conf_threshold:
                cv2.circle(img, center=(int(x), int(y)), color=line_color, radius=3, thickness=-1)
        return img

    if skeleton_lines:
        for skeleton_line in skeleton_lines:
            # skeleton line format: [start_point_name,end_point_name,color]
            skeleton_list = skeleton_line.split(',')
            color = colors
            if color is None:
                color = get_color(skeleton_list[2])
            image = draw_line(image,
                              keypoints_dict[skeleton_list[0]],
                              keypoints_dict[skeleton_list[1]],
                              line_color=color)
    else:
        if colors is None:
            colors = (0, 0, 0)
        image = draw_keypoints(image, list(keypoints_dict.values()), colors)
    return image

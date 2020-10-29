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
Calculate PCK for Hourglass model on validation dataset
Fanghao Yang 10/28/2020
"""
import argparse
import json
import operator
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as keras_backend
from PIL import Image
from tensorflow.keras.models import load_model
from tqdm import tqdm

from lib.postprocessing import post_process_heatmap
from utilities.image_utils import invert_transform_kp
from utilities.misc_utils import touch_dir, get_classes, get_skeleton, render_skeleton, optimize_tf_gpu
from utilities.model_utils import get_normalize
from datasets.dataset_loader import load_surreal_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, keras_backend)


def check_pred_keypoints(pred_keypoint, gt_keypoint, threshold, normalize):
    # check if ground truth keypoint is valid
    if gt_keypoint[0] > 1 and gt_keypoint[1] > 1:
        # calculate normalized euclidean distance between pred and gt keypoints
        distance = np.linalg.norm(gt_keypoint[0:2] - pred_keypoint[0:2]) / normalize
        if distance < threshold:
            # succeed prediction
            return 1
        else:
            # fail prediction
            return 0
    else:
        # invalid gt keypoint
        return -1


def keypoint_accuracy(pred_keypoints, gt_keypoints, threshold, normalize):
    assert pred_keypoints.shape[0] == gt_keypoints.shape[0], 'keypoint number mismatch'

    result_list = []
    for i in range(gt_keypoints.shape[0]):
        # compare pred keypoint with gt keypoint to get result
        result = check_pred_keypoints(pred_keypoints[i, :], gt_keypoints[i, :], threshold, normalize)
        result_list.append(result)

    return result_list


def adjust_axes(r, t, fig, axes):
    """
     Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    """
     Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def revert_keypoints(keypoints, element, heatmap_size):
    """
    invert transform keypoints based on center & scale
    Args:
        keypoints: 
        element: element of the dataset
        heatmap_size: 

    Returns:

    """
    crop_box = element['crop_box']
    center = crop_box[:2]
    max_l = max(crop_box[2], crop_box[3])
    scale = heatmap_size / (max_l + 0.0000001)
    reverted_keypoints = invert_transform_kp(keypoints, center, scale, heatmap_size, rot=0)

    return reverted_keypoints


def save_keypoints_detection(pred_keypoints, metainfo, class_names, skeleton_lines):
    result_dir = os.path.join('result', 'detection')
    touch_dir(result_dir)

    image_name = metainfo['name']
    image = Image.open(image_name)
    image_array = np.array(image, dtype='uint8')

    gt_keypoints = metainfo['pts']

    # form up gt keypoints & predict keypoints dict
    gt_keypoints_dict = {}
    pred_keypoints_dict = {}

    for i, keypoint in enumerate(gt_keypoints):
        gt_keypoints_dict[class_names[i]] = (keypoint[0], keypoint[1], 1.0)

    for i, keypoint in enumerate(pred_keypoints):
        pred_keypoints_dict[class_names[i]] = (keypoint[0], keypoint[1], keypoint[2])

    # render gt and predict keypoints skeleton on image
    image_array = render_skeleton(image_array, gt_keypoints_dict, skeleton_lines, colors=(255, 255, 255))
    image_array = render_skeleton(image_array, pred_keypoints_dict, skeleton_lines)

    image = Image.fromarray(image_array)
    # here we handle the RGBA image
    if len(image.split()) == 4:
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    image.save(os.path.join(result_dir, image_name.split(os.path.sep)[-1]))
    return


def hourglass_predict_keras(model, image_data):
    prediction = model.predict(image_data)

    # check to handle multi-output model
    if isinstance(prediction, list):
        prediction = prediction[-1]
    heatmap = prediction[0]
    return heatmap


def hourglass_predict_tflite(interpreter, image_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    # height = input_details[0]['shape'][1]
    # width = input_details[0]['shape'][2]

    image_data = image_data.astype('float32')
    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    prediction = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        prediction.append(output_data)

    heatmap = prediction[-1][0]
    return heatmap


def get_result_dict(pred_keypoints, element):
    """
    form up coco result dict with following format:
    Args:
        pred_keypoints:
        element: an element of dataset

    Returns:

    """
    # dict of results
    # {
    #  "image_id": int,
    #  "category_id": int,
    #  "keypoints": [x1,y1,v1,...,xk,yk,vk],
    #  "score": float
    # }

    image_name = element['name']
    image_id = int(os.path.basename(image_name).split('.')[0])

    result_dict = {}
    keypoints_list = []
    result_score = 0.0
    for i, keypoint in enumerate(pred_keypoints):
        keypoints_list.append(keypoint[0])
        keypoints_list.append(keypoint[1])
        keypoints_list.append(1)  # visibility value. simply set vk=1
        result_score += keypoint[2]

    result_dict['image_id'] = image_id
    result_dict['category_id'] = 1  # person id
    result_dict['keypoints'] = keypoints_list
    result_dict['score'] = result_score / len(pred_keypoints)
    return result_dict


def eval_pck(model, model_format, eval_dataset, class_names, score_threshold, normalize, conf_threshold,
             save_result=False, skeleton_lines=None):

    succeed_dict = {class_name: 0 for class_name in class_names}
    fail_dict = {class_name: 0 for class_name in class_names}
    accuracy_dict = {class_name: 0. for class_name in class_names}

    # init output list for coco result json generation
    # coco keypoints result is a list of following format dict:
    # {
    #  "image_id": int,
    #  "category_id": int,
    #  "keypoints": [x1,y1,v1,...,xk,yk,vk],
    #  "score": float
    # }
    #
    output_list = []

    count = 0
    batch_size = 1
    pbar = tqdm(total=eval_dataset.get_dataset_size(), desc='Eval model')
    for element in eval_dataset.take(batch_size):
        image_data = element['rgb']
        # gt_heatmap = element['heat_map']
        # fetch validation data from daset, which will crop out single person area,
        # resize to input_size and normalize image
        count += batch_size
        if count > eval_dataset.get_dataset_size():
            break

        # support of tflite model
        if model_format == 'TFLITE':
            heatmap = hourglass_predict_tflite(model, image_data)
        # normal keras h5 model
        elif model_format == 'H5':
            heatmap = hourglass_predict_keras(model, image_data)
        else:
            raise ValueError('invalid model format')

        heatmap_size = heatmap.shape[0:2]

        # get predict keypoints from heatmap
        pred_keypoints = post_process_heatmap(heatmap, conf_threshold)
        pred_keypoints = np.array(pred_keypoints)

        # get ground truth keypoints (transformed)
        gt_keypoints = element['joints_2d']

        # calculate succeed & failed keypoints for prediction
        result_list = keypoint_accuracy(pred_keypoints, gt_keypoints, score_threshold, normalize)

        for i, class_name in enumerate(class_names):
            if result_list[i] == 0:
                fail_dict[class_name] = fail_dict[class_name] + 1
            elif result_list[i] == 1:
                succeed_dict[class_name] = succeed_dict[class_name] + 1

        # revert predict keypoints back to origin image size
        reverted_pred_keypoints = revert_keypoints(pred_keypoints, element, heatmap_size)

        # get coco result dict with predict keypoints and image info
        result_dict = get_result_dict(reverted_pred_keypoints, element)
        # add result dict to output list
        output_list.append(result_dict)

        if save_result:
            # render keypoints skeleton on image and save result
            save_keypoints_detection(reverted_pred_keypoints, element, class_names, skeleton_lines)
        pbar.update(batch_size)
    pbar.close()

    # save to coco result json
    touch_dir('result')
    json_fp = open(os.path.join('result', 'keypoints_result.json'), 'w')
    json_str = json.dumps(output_list)
    json_fp.write(json_str)
    json_fp.close()

    # calculate accuracy for each class
    for i, class_name in enumerate(class_names):
        accuracy_dict[class_name] = succeed_dict[class_name] * 1.0 / (succeed_dict[class_name] + fail_dict[class_name])

    # get PCK accuracy from succeed & failed keypoints
    total_succeed = np.sum(list(succeed_dict.values()))
    total_fail = np.sum(list(fail_dict.values()))
    total_accuracy = total_succeed * 1.0 / (total_fail + total_succeed)

    if save_result:
        '''
         Draw PCK plot
        '''
        window_title = "PCK evaluation"
        plot_title = "PCK@{0} score = {1:.2f}%".format(score_threshold, total_accuracy)
        x_label = "Accuracy"
        output_path = os.path.join('result', 'PCK.jpg')
        draw_plot_func(accuracy_dict, len(accuracy_dict), window_title, plot_title, x_label, output_path, to_show=False,
                       plot_color='royalblue', true_p_bar='')

    return total_accuracy, accuracy_dict


def load_eval_model(model_path):
    # support of tflite model
    if model_path.endswith('.tflite'):
        from tensorflow.lite.python import interpreter as interpreter_wrapper
        model = interpreter_wrapper.Interpreter(model_path=model_path)
        model.allocate_tensors()
        model_format = 'TFLITE'

    # normal keras h5 model
    elif model_path.endswith('.h5'):
        model = load_model(model_path, compile=False)
        model_format = 'H5'
        keras_backend.set_learning_phase(0)
    else:
        raise ValueError('invalid model file')

    return model, model_format


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                     description='evaluate Hourglass model (h5/tflite) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--classes_path', type=str, required=False,
        help='path to class definitions, default=%(default)s', default='datasets/surreal/joint_list.txt')

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='dataset path containing images and annotation file')

    parser.add_argument(
        '--score_threshold', type=float,
        help='score threshold for PCK evaluation, default=%(default)s', default=0.5)

    # parser.add_argument(
    # '--normalize', type=float,
    # help='normalized coefficient of keypoint distance for PCK evaluation , default=6.4', default=6.4)

    parser.add_argument(
        '--conf_threshold', type=float,
        help='confidence threshold for filtering keypoint in postprocess, default=%(default)s', default=1e-6)

    parser.add_argument(
        '--model_image_size', type=str,
        help='model image input size as <height>x<width>, default=%(default)s', default='256x256')

    parser.add_argument(
        '--save_result', default=False, action="store_true",
        help='Save the detection result image in result/detection dir')

    parser.add_argument(
        '--skeleton_path', type=str, required=False,
        help='path to keypoint skeleton definitions, default None', default=None)

    args = parser.parse_args()

    # param parse
    if args.skeleton_path:
        skeleton_lines = get_skeleton(args.skeleton_path)
    else:
        skeleton_lines = None

    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))
    normalize = get_normalize(model_image_size)

    model, model_format = load_eval_model(args.model_path)

    eval_dataset = load_surreal_data(args.dataset_path)

    total_accuracy, accuracy_dict = eval_pck(model, model_format, eval_dataset, class_names, args.score_threshold,
                                             normalize, args.conf_threshold, args.save_result, skeleton_lines)

    print('\nPCK evaluation')
    for (class_name, accuracy) in accuracy_dict.items():
        print('%s: %f' % (class_name, accuracy))
    print('total acc: %f' % total_accuracy)


if __name__ == '__main__':
    main()

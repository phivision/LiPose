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
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as keras_backend
from PIL import Image
from tqdm import tqdm
from lib.postprocessing import post_process_heatmap
from lib.visualization import draw_plot_func
from utilities.image_utils import invert_transform_kp
from utilities.misc_utils import touch_dir, get_classes, get_skeleton, render_skeleton, optimize_tf_gpu, \
    count_tfrecord_examples
from utilities.model_utils import load_eval_model, get_normalize
from datasets.dataset_loader import load_full_surreal_data, parse_tfr_tensor
from datasets.dataset_converter import relative_joints

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, keras_backend)


def check_pred_joints(pred_keypoint, gt_keypoint, threshold, normalize):
    # check if ground truth keypoint is valid
    if gt_keypoint[0] > 1 and gt_keypoint[1] > 1:
        # calculate normalized euclidean distance between pred and gt joints
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


def keypoint_accuracy(pred_joints, gt_joints, threshold, normalize):
    assert pred_joints.shape[0] == gt_joints.shape[0], 'keypoint number mismatch'

    result_list = []
    for i in range(gt_joints.shape[0]):
        # compare pred keypoint with gt keypoint to get result
        result = check_pred_joints(pred_joints[i, :], gt_joints[i, :], threshold, normalize)
        result_list.append(result)

    return result_list


def revert_joints(joints, element, heatmap_size):
    """
    invert transform joints based on center & scale
    Args:
        joints:
        element: element of the dataset
        heatmap_size: 

    Returns:

    """
    crop_box = element['crop_box']
    center = crop_box[:2]
    max_l = max(crop_box[2], crop_box[3])
    scale = max(heatmap_size) / (float(max_l) + 0.0000001)
    reverted_joints = invert_transform_kp(joints, center, scale, heatmap_size, rot=0)

    return reverted_joints


def save_joints_detection(pred_joints, metainfo, class_names, skeleton_lines):
    result_dir = os.path.join('result', 'detection')
    touch_dir(result_dir)

    image_name = metainfo['name']
    image_array = metainfo['rgb'].numpy()

    gt_joints = metainfo['pts']

    # form up gt joints & predict joints dict
    gt_joints_dict = {}
    pred_joints_dict = {}

    for i, keypoint in enumerate(gt_joints):
        gt_joints_dict[class_names[i]] = (keypoint[0], keypoint[1], 1.0)

    for i, keypoint in enumerate(pred_joints):
        pred_joints_dict[class_names[i]] = (keypoint[0], keypoint[1], keypoint[2])

    # render gt and predict joints skeleton on image
    image_array = render_skeleton(image_array, gt_joints_dict, skeleton_lines, colors=(255, 255, 255))
    image_array = render_skeleton(image_array, pred_joints_dict, skeleton_lines)

    image = Image.fromarray(image_array)
    # here we handle the RGBA image
    if len(image.split()) == 4:
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    image.save(os.path.join(result_dir, image_name.split(os.path.sep)[-1]))
    return


def hourglass_predict_keras(model, image_data):
    # cast the raw image datatype (uint8 or float) to float32
    input_data = tf.expand_dims(tf.cast(image_data, dtype=tf.float32), axis=0)
    prediction = model.predict(input_data)
    # check to handle multi-output model
    if isinstance(prediction, list):
        prediction = prediction[-1]
    heatmap = prediction[0]
    return heatmap


def hourglass_predict_coreml(mlmodel, image_data):
    if tf.is_tensor(image_data):
        # convert to numpy data if the input were tensor
        image_data = image_data.numpy()
    input_data = np.expand_dims(image_data, axis=0)
    prediction_dict = mlmodel.predict({"input_1": input_data})
    heatmap = prediction_dict["Identity_1"][0]
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


def get_result_dict(pred_joints, element):
    """
    form up coco result dict with following format:
    Args:
        pred_joints:
        element: an element of dataset

    Returns:

    """
    # dict of results
    # {
    #  "image_id": int,
    #  "category_id": int,
    #  "joints": [x1,y1,v1,...,xk,yk,vk],
    #  "score": float
    # }

    image_id = element['name'].numpy().decode('ascii')

    result_dict = {}
    joints_list = []
    result_score = 0.0
    for i, keypoint in enumerate(pred_joints):
        joints_list.append(keypoint[0])
        joints_list.append(keypoint[1])
        joints_list.append(1)  # visibility value. simply set vk=1
        result_score += keypoint[2]

    result_dict['image_id'] = image_id
    result_dict['category_id'] = 1  # person id
    result_dict['joints'] = joints_list
    result_dict['score'] = result_score / len(pred_joints)
    return result_dict


def eval_pck(model,
             model_format,
             eval_dataset,
             class_names,
             score_threshold,
             normalize,
             conf_threshold,
             image_type='rgb',
             save_result=False,
             skeleton_lines=None):
    """Evaluate trained model

    Args:
        model:
        model_format: serialization format of model, e.g. TFLITE, PB, H5
        eval_dataset: path to evaluation dataset
        class_names: name of joint labels (classes)
        score_threshold:
        normalize:
        conf_threshold:
        image_type: type of image for model input, e.g. rgb, depth, etc
        save_result: if save evaluation result
        skeleton_lines:

    Returns:

    """
    succeed_dict = {class_name: 0 for class_name in class_names}
    fail_dict = {class_name: 0 for class_name in class_names}
    accuracy_dict = {class_name: 0. for class_name in class_names}

    # init output list for coco result json generation
    # coco joints result is a list of following format dict:
    # {
    #  "image_id": int,
    #  "category_id": int,
    #  "joints": [x1,y1,v1,...,xk,yk,vk],
    #  "score": float
    # }
    #
    output_list = []

    total_example = count_tfrecord_examples(eval_dataset)
    p_bar = tqdm(total=total_example, desc='Eval model')
    # fetch validation data from dataset, which will crop out single person area,
    for element in eval_dataset.take(total_example):
        example = parse_tfr_tensor(element)
        if image_type == 'rgb':
            image_data = example[image_type]
        elif image_type == 'depth':
            # insert image channel index to the shape of depth map
            image_data = tf.expand_dims(example[image_type], -1)
        else:
            raise TypeError(f"Do not support image type {image_type}")
        # gt_heatmap = example['heat_map']

        # support of tflite model
        if model_format == 'TFLITE':
            heatmap = hourglass_predict_tflite(model, image_data)
        # normal keras h5 model
        elif model_format == 'H5' or model_format == 'PB':
            # the tf keras h5 format or keras subclassing model in protobuf format
            heatmap = hourglass_predict_keras(model, image_data)
        elif model_format == 'COREML':
            heatmap = hourglass_predict_coreml(model, image_data)
        else:
            raise ValueError('invalid model format')

        heatmap_size = heatmap.shape[0:2]

        # get predict joints from heatmap
        pred_joints = post_process_heatmap(heatmap, conf_threshold)
        pred_joints = np.array(pred_joints)

        # get ground truth joints (transformed)
        crop_box = example['crop_box'].numpy()
        joints_2d = example['joints_2d'].numpy()
        padding = [[0, 0], [0, 0]]
        gt_joints = relative_joints(crop_box, padding, joints_2d, to_size=heatmap_size[0]).T

        # calculate succeed & failed joints for prediction
        result_list = keypoint_accuracy(pred_joints, gt_joints, score_threshold, normalize)

        for i, class_name in enumerate(class_names):
            if result_list[i] == 0:
                fail_dict[class_name] = fail_dict[class_name] + 1
            elif result_list[i] == 1:
                succeed_dict[class_name] = succeed_dict[class_name] + 1

        # revert predict joints back to origin image size
        reverted_pred_joints = revert_joints(pred_joints, example, heatmap_size)

        # get coco result dict with predict joints and image info
        result_dict = get_result_dict(reverted_pred_joints, example)
        # add result dict to output list
        output_list.append(result_dict)

        if save_result:
            # render joints skeleton on image and save result
            save_joints_detection(reverted_pred_joints, example, class_names, skeleton_lines)
        p_bar.update(1)
    p_bar.close()

    # save to coco result json
    touch_dir('result')
    json_fp = open(os.path.join('result', 'joints_result.json'), 'w')
    json_str = json.dumps(output_list)
    json_fp.write(json_str)
    json_fp.close()

    # calculate accuracy for each class
    for i, class_name in enumerate(class_names):
        accuracy_dict[class_name] = succeed_dict[class_name] * 1.0 / (succeed_dict[class_name] + fail_dict[class_name])

    # get PCK accuracy from succeed & failed joints
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

    parser.add_argument('--image_type', type=str, required=False, default='rgb',
                        help="Type of model input, e.g. rgb, depth map, etc")

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

    eval_dataset = load_full_surreal_data(args.dataset_path)

    total_accuracy, accuracy_dict = eval_pck(model, model_format, eval_dataset, class_names, args.score_threshold,
                                             normalize, args.conf_threshold, args.image_type, args.save_result,
                                             skeleton_lines)

    print('\nPCK evaluation')
    for (class_name, accuracy) in accuracy_dict.items():
        print('%s: %f' % (class_name, accuracy))
    print('total acc: %f' % total_accuracy)


if __name__ == '__main__':
    main()

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
Training slim neural network for mobile applications.
This model uses separable convolution by default, for faster speed
Fanghao Yang 10/29/2020
"""

import argparse
import os

import tensorflow.keras.backend as keras_backend
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN
from tensorflow.keras.losses import mean_squared_error

from datasets.dataset_loader import load_surreal_data_training
from lib.models.callbacks import EvalCallBack
from lib.models.stacked_hourglass import StackedHourglass
from utilities.misc_utils import get_classes, get_model_type, optimize_tf_gpu, count_tfrecord_examples
from utilities.model_utils import get_optimizer, save_model

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

optimize_tf_gpu(tf, keras_backend)


def main(arguments):
    log_dir = arguments.log_path
    os.makedirs(log_dir, exist_ok=True)

    class_names = get_classes(arguments.classes_path)
    num_classes = len(class_names)
    # TODO: add random image flip, rotate, zoom to training data
    # if arguments.matchpoint_path:
    #     matchpoints = get_match_points(arguments.matchpoint_path)
    # else:
    #     matchpoints = None
    image_type = arguments.image_type

    # get train dataset
    train_dataset = load_surreal_data_training(arguments.dataset_path,
                                               arguments.batch_size,
                                               num_features=arguments.num_features,
                                               shuffle=True,
                                               image_type=image_type)
    # check if the dataset matches the arguments
    input_size = None
    for image, _ in train_dataset.take(1):
        input_size = image.shape[1:]
    model_type = get_model_type(arguments.num_stacks, True, arguments.tiny, input_size)
    # callbacks for training process
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False,
                              write_images=False, update_freq='batch')
    # load validation data for evaluation callback
    eval_callback = EvalCallBack(log_dir, arguments.val_data_path, class_names, input_size, model_type, image_type)
    terminate_on_nan = TerminateOnNaN()
    callbacks = [tensorboard, eval_callback, terminate_on_nan]

    # prepare optimizer
    optimizer = get_optimizer(arguments.optimizer, arguments.learning_rate, decay_type=None)

    # support multi-gpu training
    if arguments.gpu_num >= 2:
        if arguments.gpu_num == 2:
            gpus = [0, 2]
        else:
            gpus = range(arguments.gpu_num)
        devices_list = ["/gpu:{}".format(n) for n in gpus]
        strategy = tf.distribute.MirroredStrategy(devices=devices_list)
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            # get multi-gpu train model
            model = StackedHourglass.from_shape_type(arguments.num_stacks, num_classes,
                                                     tiny=arguments.tiny,
                                                     image_type=image_type)

            # compile model
            model.compile(optimizer=optimizer, loss=mean_squared_error)
            # this customized summary code needs to be the same scope of the model instantiation
            model.summary(line_length=120)
    else:
        # get normal train model, doesn't specify input size
        model = StackedHourglass.from_shape_type(arguments.num_stacks, num_classes,
                                                 tiny=arguments.tiny,
                                                 image_type=image_type)
        # compile model
        model.compile(optimizer=optimizer, loss=mean_squared_error)
        model.summary(line_length=120)

    print(f"Create Mobile Stacked Hourglass model with stack number {arguments.num_stacks}, "
          f"channel number {model.num_features}. "
          f"train input size {input_size}")

    if arguments.weights_path:
        model.load_weights(arguments.weights_path, by_name=True)  # , skip_mismatch=True)
        print('Load weights {}.'.format(arguments.weights_path))

    # start training
    # WARNING: if set the step for each epoch, the training loop will stop at epoch 2, unknown bug
    model.fit(train_dataset,
              epochs=arguments.total_epoch,
              initial_epoch=arguments.init_epoch,
              callbacks=callbacks)

    save_model(model, log_dir, 'trained_final')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument("--num_stacks", type=int, required=False, default=2,
                        help='number of hourglass stacks, default=%(default)s')
    parser.add_argument('--num_features', type=int, required=False, default=256,
                        help="number of input image size")
    parser.add_argument("--tiny", default=False, action="store_true",
                        help="tiny network for speed, feature channel=128")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
                        help="Pretrained model/weights file for fine tune")
    parser.add_argument('--image_type', type=str, required=False, default='rgb',
                        help="Type of model input, e.g. rgb, depth map, etc")

    # Data options
    parser.add_argument('--dataset_path', type=str, required=False,
                        default='/home/fanghao/Documents/surreal_tfrecords/train/run0.tfrecord',
                        help='dataset path containing images and annotation file, default=%(default)s')
    parser.add_argument('--val_data_path', type=str, required=False,
                        default='/home/fanghao/Documents/surreal_tfrecords/val/run0.tfrecord',
                        help='dataset path for validation, default=%(default)s')
    parser.add_argument('--classes_path', type=str, required=False, default='datasets/surreal/joint_list.txt',
                        help='path to keypoint class definitions, default=%(default)s')
    parser.add_argument('--log_path', type=str, required=False, default='logs/000',
                        help="Path to set log files and trained models")

    # Training options
    parser.add_argument("--batch_size", type=int, required=False, default=16,
                        help='batch size for training, default=%(default)s')
    parser.add_argument('--optimizer', type=str, required=False, default='rmsprop',
                        help="optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=5e-4,
                        help="Initial learning rate, default=%(default)s")
    parser.add_argument("--init_epoch", type=int, required=False, default=0,
                        help="initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument("--total_epoch", type=int, required=False, default=100,
                        help="total training epochs, default=%(default)s")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
                        help='Number of GPU to use, default=%(default)s')

    args = parser.parse_args()

    main(args)

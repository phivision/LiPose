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
from lib.models.slim_hourglass import get_hourglass_model
from utilities.misc_utils import get_classes, get_model_type, optimize_tf_gpu
from utilities.model_utils import get_optimizer

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

optimize_tf_gpu(tf, keras_backend)


def main(arguments):
    log_dir = 'logs/000'
    os.makedirs(log_dir, exist_ok=True)

    class_names = get_classes(arguments.classes_path)
    num_classes = len(class_names)
    # TODO: add random image flip, rotate, zoom to training data
    # if arguments.matchpoint_path:
    #     matchpoints = get_match_points(arguments.matchpoint_path)
    # else:
    #     matchpoints = None

    # choose model type
    if arguments.tiny:
        num_channels = 128
        # input_size = (192, 192)
    else:
        num_channels = 256
        # input_size = (256, 256)

    input_size = arguments.model_image_size

    # get train/val dataset
    train_dataset = load_surreal_data_training(arguments.dataset_path, arguments.batch_size, shuffle=True)

    model_type = get_model_type(arguments.num_stacks, True, arguments.tiny, input_size)

    # callbacks for training process
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False,
                              write_images=False, update_freq='batch')
    # load validation data for evaluation callback
    eval_callback = EvalCallBack(log_dir, arguments.val_data_path, class_names, input_size, model_type)
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
            model = get_hourglass_model(num_classes, arguments.num_stacks, num_channels,
                                        input_size=(num_channels, num_channels))
            # compile model
            model.compile(optimizer=optimizer, loss=mean_squared_error)
    else:
        # get normal train model, doesn't specify input size
        model = get_hourglass_model(num_classes, arguments.num_stacks, num_channels)
        # compile model
        model.compile(optimizer=optimizer, loss=mean_squared_error)

    print(f"Create Mobile Stacked Hourglass model with stack number {arguments.num_stacks}, "
          f"channel number {num_channels}. "
          f"train input size {input_size}")
    model.summary()

    if arguments.weights_path:
        model.load_weights(arguments.weights_path, by_name=True)  # , skip_mismatch=True)
        print('Load weights {}.'.format(arguments.weights_path))

    # start training
    if arguments.gpu_num >= 2:
        model.fit(train_dataset,
                  epochs=arguments.total_epoch,
                  initial_epoch=arguments.init_epoch,
                  workers=arguments.gpu_num,
                  use_multiprocessing=True,
                  callbacks=callbacks)
    else:
        model.fit(train_dataset,
                  epochs=arguments.total_epoch,
                  initial_epoch=arguments.init_epoch,
                  callbacks=callbacks)

    model.save(os.path.join(log_dir, 'trained_final.h5'))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument("--num_stacks", type=int, required=False, default=2,
                        help='number of hourglass stacks, default=%(default)s')
    parser.add_argument("--tiny", default=False, action="store_true",
                        help="tiny network for speed, feature channel=128")
    parser.add_argument('--model_image_size', type=str, required=False, default='256x256',
                        help="model image input size as <height>x<width>, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
                        help="Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--dataset_path', type=str, required=False,
                        default='/home/fanghao/Documents/surreal_tfrecords/train/run0.tfrecord',
                        help='dataset path containing images and annotation file, default=%(default)s')
    parser.add_argument('--val_data_path', type=str, required=False,
                        default='/home/fanghao/Documents/surreal_tfrecords/val/run0.tfrecord',
                        help='dataset path for validation, default=%(default)s')
    parser.add_argument('--classes_path', type=str, required=False, default='datasets/surreal/joint_list.txt',
                        help='path to keypoint class definitions, default=%(default)s')

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
    height, width = args.model_image_size.split('x')
    args.model_image_size = (int(height), int(width))

    main(args)

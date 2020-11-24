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
Model utility functions.
Fanghao Yang 10/28/2020
"""
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.models import load_model, Model
from lib.models.stacked_hourglass import StackedHourglass
from pathlib import Path
import tensorflow.keras.backend as keras_backend
import tensorflow_model_optimization as tfmot
import coremltools as ct
import os


def load_eval_model(model_path):
    """Load trained models

    Args:
        model_path:

    Returns:
        model object, model extension string
    """
    # support of tflite model
    model_path = Path(model_path)
    if model_path.suffix == '.tflite':
        from tensorflow.lite.python import interpreter as interpreter_wrapper
        model = interpreter_wrapper.Interpreter(model_path=model_path)
        model.allocate_tensors()
        model_format = 'TFLITE'
    # normal keras h5 model
    elif model_path.suffix == '.h5':
        model = load_model(model_path, compile=False)
        model_format = 'H5'
        keras_backend.set_learning_phase(0)
    elif os.path.isdir(model_path):
        # load from tensorflow weights
        model = load_model(model_path)
        # model is saved as tensorflow protobuf serialized model
        model_format = 'PB'
    elif model_path.suffix == '.mlmodel':
        model = ct.models.MLModel(str(model_path))
        model_format = 'COREML'
    else:
        raise ValueError('invalid model file')

    return model, model_format


def save_model(model: Model, log_path, model_name):
    """
    
    Args:
        model: model obj to be saved
        log_path: log dir path
        model_name: name of trained model

    Returns:

    """
    print(f"Saving model to {log_path} as {model_name}")
    if isinstance(model, StackedHourglass):
        # this a customized subclassing model, need to saved as tf format
        save_path = os.path.join(log_path, model_name)
        model.save(save_path)
    else:
        save_path = os.path.join(log_path, model_name + '.h5')
        model.save(save_path)


def convert_keras_to_coreml(input_path, input_shape, output_path):
    """Convert TF Keras model to iOS CoreML model

    Args:
        input_path: input model path
        input_shape: shape of image image
        output_path: output path to save model

    Returns:

    """
    tfk_model, _ = load_eval_model(input_path)
    # not using image as input
    # TODO: the converted model may not has consistent input placeholder, need a fix
    input_type = ct.TensorType(shape=input_shape, name='input')
    ct_model = ct.convert(tfk_model, source='tensorflow', inputs=[input_type])
    ct_model.save(output_path)


def add_metrics(model, metric_dict):
    '''
    add metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, metric) in metric_dict.items():
        # seems add_metric() is newly added in tf.keras. So if you
        # want to customize metrics on raw keras model, just use
        # "metrics_names" and "metrics_tensors" as follow:
        #
        # model.metrics_names.append(name)
        # model.metrics_tensors.append(loss)
        model.add_metric(metric, name=name, aggregation='mean')


def get_pruning_model(model, begin_step, end_step):
    import tensorflow as tf
    if tf.__version__.startswith('2'):
        # model pruning API is not supported in TF 2.0 yet
        raise Exception('model pruning is not fully supported in TF 2.x, Please switch env to TF 1.x for this feature')

    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                               final_sparsity=0.7,
                                                               begin_step=begin_step,
                                                               end_step=end_step,
                                                               frequency=100)
    }

    pruning_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    return pruning_model


def get_lr_scheduler(learning_rate, decay_type, decay_steps):
    if decay_type:
        decay_type = decay_type.lower()

    if decay_type is None:
        lr_scheduler = learning_rate
    elif decay_type == 'cosine':
        lr_scheduler = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps)
    elif decay_type == 'exponential':
        lr_scheduler = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=0.9)
    elif decay_type == 'polynomial':
        lr_scheduler = PolynomialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, end_learning_rate=learning_rate/100)
    else:
        raise ValueError('Unsupported lr decay type')

    return lr_scheduler


def get_optimizer(optim_type, learning_rate, decay_type='cosine', decay_steps=100000):
    optim_type = optim_type.lower()

    lr_scheduler = get_lr_scheduler(learning_rate, decay_type, decay_steps)

    if optim_type == 'adam':
        optimizer = Adam(learning_rate=lr_scheduler)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_scheduler)
    elif optim_type == 'sgd':
        optimizer = SGD(learning_rate=lr_scheduler)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer


def get_normalize(input_size):
    """
    rescale keypoint distance normalize coefficient
    based on input size, used for PCK evaluation
    NOTE: 6.4 is standard normalize coefficient under
          input size (256,256)
    """
    assert input_size[0] == input_size[1], 'only support square input size.'

    scale = float(input_size[0]) / 256.0

    return 6.4*scale

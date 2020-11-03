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
Callback to evaluate models after every epoch
Fanghao Yang, 10/29/2020
"""
import os
from tensorflow.keras.callbacks import Callback
from datasets.dataset_loader import load_full_surreal_data
from utilities.model_utils import get_normalize, save_model
from eval import eval_pck


class EvalCallBack(Callback):
    def __init__(self, log_dir, dataset_path, class_names, input_size, model_type, image_type):
        super().__init__()
        self.log_dir = log_dir
        self.dataset_path = dataset_path
        self.class_names = class_names
        self.normalize = get_normalize(input_size)
        self.input_size = input_size
        self.best_accuracy = 0.0
        self.image_type = image_type

        # record model & dataset name to draw training curve
        with open(os.path.join(self.log_dir, 'val.txt'), 'a+') as xfile:
            xfile.write('model:' + model_type + ';dataset:' + dataset_path + '\n')
        xfile.close()

    def on_epoch_end(self, epoch, logs=None):
        val_dataset = load_full_surreal_data(self.dataset_path)

        val_acc, _ = eval_pck(self.model, 'H5', val_dataset, self.class_names,
                              image_type=self.image_type,
                              score_threshold=0.5,
                              normalize=self.normalize,
                              conf_threshold=1e-6,
                              save_result=False)
        print('validate accuracy', val_acc, '@epoch', epoch)

        # record accuracy for every epoch to draw training curve
        with open(os.path.join(self.log_dir, 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(val_acc) + '\n')
        xfile.close()

        if val_acc > self.best_accuracy:
            # Save best accuracy value and model checkpoint
            self.best_accuracy = val_acc
            model_name = 'ep{epoch:03d}-loss{loss:.3f}-val_acc{val_acc:.3f}'.format(epoch=(epoch + 1),
                                                                                    loss=logs.get('loss'),
                                                                                    val_acc=val_acc)
            save_model(self.model, self.log_dir, model_name)

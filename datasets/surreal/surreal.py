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
SURREAL dataset for synthetic human pose estimation
By Fanghao Yang, 10/15/2020
"""

import tensorflow_datasets as tfds

_DESCRIPTION = """
Estimating human pose, shape, and motion from images and video are fundamental challenges with many applications. 
Recent advances in 2D human pose estimation use large amounts of manually-labeled training data for learning 
convolutional neural networks (CNNs). Such data is time consuming to acquire and difficult to extend. 
Moreover, manual labeling of 3D pose, depth and motion is impractical. In this work we present SURREAL: 
a new large-scale dataset with synthetically-generated but realistic images of people rendered from 3D sequences of 
human motion capture data. We generate more than 6 million frames together with ground truth pose, depth maps, 
and segmentation masks. We show that CNNs trained on our synthetic dataset allow for accurate human depth estimation 
and human part segmentation in real RGB images. Our results and the new dataset open up new possibilities for advancing 
person analysis using cheap and large-scale synthetic data.
"""

_CITATION = """
@INPROCEEDINGS{varol17_surreal,
  title     = {Learning from Synthetic Humans},
  author    = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and 
  Laptev, Ivan and Schmid, Cordelia},
  booktitle = {CVPR},
  year      = {2017}
}
"""


class Surreal(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for surreal dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(surreal): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'rgb': tfds.features.Image(shape=(320, 240, 3))
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # e.g. ('image', 'label')
            homepage='https://www.di.ens.fr/willow/research/surreal/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(surreal): Downloads the data and defines the splits
        # dl_manager is a tfds.download.DownloadManager that can be used to
        # download and extract URLs
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={},
            ),
        ]

    def _generate_examples(self):
        """Yields examples."""
        # TODO(surreal): Yields (key, example) tuples from the dataset
        yield 'key', {}

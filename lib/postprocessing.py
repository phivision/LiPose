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
Postprocessing of heatmap
Fanghao yang 10/28/2020
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def non_max_suppression(plain, window_size=3, conf_threshold=1e-6):
    # clear value less than conf_threshold
    under_th_indices = plain < conf_threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))


def _get_keypoint(heatmap):
    y, x = np.where(heatmap == heatmap.max())
    if len(x) > 0 and len(y) > 0:
        return int(x[0]), int(y[0]), heatmap[y[0], x[0]]
    else:
        return 0, 0, 0.0


def post_process_heatmap(heatmap, conf_threshold=1e-6):
    keypoint_list = list()
    for i in range(heatmap.shape[-1]):
        _map = heatmap[:, :, i]
        # do a heatmap blur with gaussian_filter
        _map = gaussian_filter(_map, sigma=0.5)
        # get peak point in heatmap with 3x3 max filter
        _nmsPeaks = non_max_suppression(_map, window_size=3, conf_threshold=conf_threshold)

        # choose the max point in heatmap (we only pick 1 keypoint in each heatmap)
        # and get its coordinate & confidence
        keypoint_list.append(_get_keypoint(_nmsPeaks))
    return keypoint_list


def post_process_heatmap_single(heatmap, conf_threshold=1e-6):
    keypoint_list = list()
    for i in range(heatmap.shape[-1]):
        # ignore last channel, background channel
        _map = heatmap[:, :, i]
        # clear value less than conf_threshold
        under_th_indices = _map < conf_threshold
        _map[under_th_indices] = 0

        # choose the max point in heatmap (we only pick 1 keypoint in each heatmap)
        # and get its coordinate & confidence
        keypoint_list.append((_get_keypoint(_map)))
    return keypoint_list

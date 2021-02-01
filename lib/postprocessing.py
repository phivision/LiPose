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


def _get_multi_keypoint(heatmap, num_points=4):
    """get multiple keypoint from existing heatmaps

    Args:
        heatmap:
        num_points: find up to number of key points from the heatmap, chosen 4 by default since human has 4 limbs

    Returns:

    """
    point_indexes = np.argpartition(heatmap.flatten(), -num_points)[-num_points:]
    width, _ = heatmap.shape
    point_list = []
    for ind in point_indexes:
        y = ind // width
        x = ind % width
        if heatmap[y, x] > 0.0:
            point_list.append((x, y, heatmap[y, x]))
    return point_list


def post_process_heatmap(heatmap, conf_threshold=1e-6):
    """return a list of multiple possible joint locations

    Args:
        heatmap:
        conf_threshold:

    Returns:

    """
    multi_point_list = list()
    for i in range(heatmap.shape[-1]):
        _map = heatmap[:, :, i]
        # do a heatmap blur with gaussian_filter
        _map = gaussian_filter(_map, sigma=0.5)
        # get peak point in heatmap with 3x3 max filter
        _nmsPeaks = non_max_suppression(_map, window_size=3, conf_threshold=conf_threshold)

        # choose the max point in heatmap (we pick multiple points from the heatmap)
        # and get its coordinate & confidence
        multi_point_list.append(_get_multi_keypoint(_nmsPeaks))
    return multi_point_list


def match_keypoint(keypoint_list, match_pairs, dist_threshold=6):
    """find matched pair of keypoints and filter out unmatched points

    Args:
        keypoint_list: a list of multiple possible joints from heat map
        match_pairs: a list of matched pairs of points
        dist_threshold: the distance threshold to filter out sparse point pairs

    Returns:

    """
    matched_list = [points[0] if len(points) == 1 else (0, 0, 0.0) for points in keypoint_list]
    for first, second in match_pairs:
        if len(keypoint_list[first]) > 0 and len(keypoint_list[second]) > 0:
            for x1, y1, conf1 in keypoint_list[first]:
                for x2, y2, conf2 in keypoint_list[second]:
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < dist_threshold:
                        # compare the least confidence of each pairs
                        if min(conf1, conf2) > min(matched_list[first][2], matched_list[second][2]):
                            # if the least confidence is larger, replace the pair
                            matched_list[first] = (x1, y1, conf1)
                            matched_list[second] = (x2, y2, conf2)
    return matched_list


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

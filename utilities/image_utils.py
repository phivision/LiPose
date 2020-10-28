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
image processing utilities
Fanghao Yang 11/28/2020
"""
import numpy as np
from PIL import Image
import cv2


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    new_ht = int(np.math.floor(ht / sf))
    new_wd = int(np.math.floor(wd / sf))
    img = np.array(Image.fromarray(img).resize((new_wd, new_ht), Image.BICUBIC))
    center = center * 1.0 / sf
    scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.uint8)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = np.array(Image.fromarray(new_img).rotate(rot))
        new_img = new_img[pad:-pad, pad:-pad]

    if 0 in new_img.shape:
        # in case we got a empty image
        return None

    new_img = np.array(Image.fromarray(new_img).resize(res, Image.BICUBIC))
    return new_img


def horizontal_flip(image, joints, center, matchpoints=None):
    joints = np.copy(joints)

    # some keypoint pairs also need to be fliped
    # on new image
    # matchpoints = (
    # [0, 5],  # ankle
    # [1, 4],  # knee
    # [2, 3],  # hip
    # [10, 15],  # wrist
    # [11, 14],  # elbow
    # [12, 13]  # shoulder
    # )

    org_height, org_width, channels = image.shape

    # horizontal flip image: flipCode=1
    flipimage = cv2.flip(image, flipCode=1)

    # horizontal flip each joints
    joints[:, 0] = org_width - joints[:, 0]

    # horizontal flip matched keypoints
    if matchpoints and len(matchpoints) != 0:
        for i, j in matchpoints:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

    # horizontal flip center
    flip_center = center
    flip_center[0] = org_width - center[0]

    return flipimage, joints, flip_center


def vertical_flip(image, joints, center, matchpoints=None):
    joints = np.copy(joints)

    # some keypoint pairs also need to be fliped
    # on new image

    org_height, org_width, channels = image.shape

    # vertical flip image: flipCode=0
    flipimage = cv2.flip(image, flipCode=0)

    # vertical flip each joints
    joints[:, 1] = org_height - joints[:, 1]

    # vertical flip matched keypoints
    if matchpoints and len(matchpoints) != 0:
        for i, j in matchpoints:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

    # vertical flip center
    flip_center = center
    flip_center[1] = org_height - center[1]

    return flipimage, joints, flip_center


def normalize(imgdata, color_mean):
    """

    Args:
        imgdata: image in 0 ~ 255
        color_mean:

    Returns:
        image from 0.0 to 1.0
    """
    imgdata = imgdata / 255.0

    for i in range(imgdata.shape[-1]):
        imgdata[:, :, i] -= color_mean[i]

    return imgdata


def draw_labelmap(img, pt, sigma, map_type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if map_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif map_type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)
    else:
        raise TypeError(f"{map_type} is not supported!")

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def transform_kp(joints, center, scale, res, rot):
    newjoints = np.copy(joints)
    for i in range(joints.shape[0]):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            _x = transform(newjoints[i, 0:2] + 1, center=center, scale=scale, res=res, invert=0, rot=rot)
            newjoints[i, 0:2] = _x
    return newjoints


def invert_transform_kp(joints, center, scale, res, rot):
    newjoints = np.copy(joints)
    for i in range(joints.shape[0]):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            _x = transform(newjoints[i, 0:2] + 1, center=center, scale=scale, res=res, invert=1, rot=rot)
            newjoints[i, 0:2] = _x
    return newjoints


def generate_gtmap(joints, sigma, outres):
    npart = joints.shape[0]
    gtmap = np.zeros(shape=(outres[0], outres[1], npart), dtype=float)
    for i in range(npart):
        visibility = joints[i, 2]
        if visibility > 0:
            gtmap[:, :, i] = draw_labelmap(gtmap[:, :, i], joints[i, :], sigma)
    return gtmap


def preprocess_image(image, model_input_size, mean=(0.4404, 0.4440, 0.4327)):
    """
    Prepare model input image data with
    resize, normalize and dim expansion
    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).
    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = image.resize(model_input_size, Image.BICUBIC)
    image_data = np.asarray(resized_image).astype('float32')

    mean = np.array(mean, dtype=np.float)
    image_data = normalize(image_data, mean)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension
    return image_data

import numpy as np
import cv2
import random
import tensorflow as tf

from anchors import generate_anchors
from config import *
from box_utils import match_target


def _matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def _crop(image, boxes, labels, landm, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = _matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        landms_t = landms_t.reshape([-1, 10])


	# make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return image, boxes, labels, landm, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image


class DataGenerator:
    def __init__(self):
        self.img_paths = []
        self.labels = []
        self.output_shape = image_shape
        self.rgb_means = rgb_means

        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        img_label = np.zeros((0, 15))
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    img_label_copy = img_label.copy()
                    self.labels.append(img_label_copy)
                    img_label = np.zeros((0, 15))
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.img_paths.append(path)
            else:
                line = line.split(' ')
                line = [float(x) for x in line]

                label = np.zeros((1, 15))
                label[0, 0] = line[0]  # x1
                label[0, 1] = line[1]  # y1
                label[0, 2] = line[0] + line[2]  # x2
                label[0, 3] = line[1] + line[3]  # y2

                # landmarks
                label[0, 4] = line[4]    # l0_x
                label[0, 5] = line[5]    # l0_y
                label[0, 6] = line[7]    # l1_x
                label[0, 7] = line[8]    # l1_y
                label[0, 8] = line[10]   # l2_x
                label[0, 9] = line[11]   # l2_y
                label[0, 10] = line[13]  # l3_x
                label[0, 11] = line[14]  # l3_y
                label[0, 12] = line[16]  # l4_x
                label[0, 13] = line[17]  # l4_y
                if (label[0, 4]<0):
                    label[0, 14] = -1
                else:
                    label[0, 14] = 1
                img_label = np.append(img_label, label, axis=0)
        self.labels.append(img_label)
        self.anchors = generate_anchors(min_sizes, steps, image_shape)
        self.total = len(self.img_paths)


    def __len__(self):
        return self.total


    def _augment(self, image, targets):
        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.output_shape[0])
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t, self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.output_shape[0], self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))

        return image_t, targets_t


    def _preprocess(self):
        for img_path, img_label in zip(self.img_paths, self.labels):
            image = cv2.imread(img_path)
            image, target = self._augment(image, img_label)
            target = tf.constant(target, dtype=tf.float32)

            gt_boxes = target[:, :4]
            gt_labels = target[:, -1]
            gt_landms = target[:, 4:14]
            gt_locs, gt_confs, gt_landms = match_target(gt_boxes, gt_labels, gt_landms, self.anchors, variance, iou_threshold=0.35)
            
            gt_locs_inf = tf.math.is_inf(gt_locs)
            if tf.math.reduce_any(gt_locs_inf):
                print("====================Inf==================")
                continue
            
            yield image, gt_locs, gt_confs, gt_landms


    def generate(self):
        data_gen = tf.data.Dataset.from_generator(self._preprocess, (tf.float32, tf.float32, tf.int64, tf.float32))
        data_gen = data_gen.shuffle(50)
        data_gen = data_gen.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        data_gen = data_gen.batch(batch_size)
        data_gen = data_gen.take(-1)

        return data_gen

import tensorflow as tf
import math
import itertools


def generate_anchors(min_sizes, steps, image_shape):
    anchors = []

    feature_maps = [[math.ceil(image_shape[0] / step), math.ceil(image_shape[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        k_minsizes = min_sizes[k]
        for i, j in itertools.product(range(f[0]), range(f[1])):
            for min_size in k_minsizes:
                s_kx = min_size / image_shape[1]
                s_ky = min_size / image_shape[0]
                cx = (j + 0.5) * steps[k] / image_shape[1]
                cy = (i + 0.5) * steps[k] / image_shape[0]
                anchors.append([cx, cy, s_kx, s_ky])

    anchors = tf.constant(anchors)
    anchors = tf.clip_by_value(anchors, 0.0, 1.0)
    
    return anchors
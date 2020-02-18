import tensorflow as tf


def compute_area(top_left, bot_right):
    """ Compute area given top_left and bottom_right coordinates
    Args:
        top_left: tensor (num_boxes, 2)
        bot_right: tensor (num_boxes, 2)
    Returns:
        area: tensor (num_boxes,)
    """
    # top_left: N x 2
    # bot_right: N x 2
    hw = tf.clip_by_value(bot_right - top_left, 0.0, 512.0)
    area = hw[..., 0] * hw[..., 1]

    return area


def compute_iou(boxes_a, boxes_b):
    """ Compute overlap between boxes_a and boxes_b
    Args:
        boxes_a: tensor (num_boxes_a, 4)
        boxes_b: tensor (num_boxes_b, 4)
    Returns:
        overlap: tensor (num_boxes_a, num_boxes_b)
    """
    # boxes_a => num_boxes_a, 1, 4
    boxes_a = tf.expand_dims(boxes_a, 1)

    # boxes_b => 1, num_boxes_b, 4
    boxes_b = tf.expand_dims(boxes_b, 0)
    top_left = tf.math.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = tf.math.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = compute_area(top_left, bot_right)
    area_a = compute_area(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = compute_area(boxes_b[..., :2], boxes_b[..., 2:])

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap


def center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box


def corner_to_center(boxes):
    """ Transform boxes of format (xmin, ymin, xmax, ymax)
        to format (cx, cy, w, h)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """
    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box


def match_target(gt_boxes, gt_labels, gt_landms, anchors, variance, iou_threshold=0.35):
    iou = compute_iou(center_to_corner(anchors), gt_boxes)

    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(best_gt_idx, tf.expand_dims(best_default_idx, 1), tf.range(best_default_idx.shape[0], dtype=tf.int64))

    best_gt_iou = tf.tensor_scatter_nd_update(best_gt_iou, tf.expand_dims(best_default_idx, 1), tf.ones_like(best_default_idx, dtype=tf.float32))

    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(tf.math.less(best_gt_iou, iou_threshold), tf.zeros_like(gt_confs),gt_confs)
    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    gt_landms = tf.gather(gt_landms, best_gt_idx)
    gt_locs = encode(anchors, gt_boxes, variance)
    gt_landms = encode_lmk(anchors, gt_landms, variance)

    return gt_locs, gt_confs, gt_landms



def encode(default_boxes, boxes, variance=[0.1, 0.2]):
    """ Compute regression values
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
        variance: variance for center point and size
    Returns:
        locs: regression values, tensor (num_default, 4)
    """
    # Convert boxes to (cx, cy, w, h) format
    transformed_boxes = corner_to_center(boxes)

    locs = tf.concat([(transformed_boxes[..., :2] - default_boxes[:, :2]) / (default_boxes[:, 2:] * variance[0]), tf.math.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]],axis=-1)

    return locs


def encode_lmk(default_boxes, landms, variance=[0.1, 0.2]):
    landms = tf.reshape(landms, (landms.shape[0], 5, 2))
    default_boxes = tf.expand_dims(default_boxes, 1)
    default_boxes = tf.broadcast_to(default_boxes, [landms.shape[0], 5, 4])

    landms_encoded = (landms[..., :2] - default_boxes[..., :2]) / (default_boxes[..., 2:] * variance[0])
    landms_encoded = tf.reshape(landms_encoded, [landms.shape[0], 10])

    return landms_encoded



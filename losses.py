import tensorflow as tf


class MultiBoxLoss:
    def __init__(self, neg_ratio=7, num_classes=2):
        self.neg_ratio = neg_ratio
        self.num_classes = num_classes


    def cal_loss(self, locs, confs, landms, gt_locs, gt_confs, gt_landms):
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')
        batch_size = locs.shape[0]

        landms_pos = gt_confs > 0
        num_pos_landm = tf.math.maximum(tf.math.count_nonzero(landms_pos, dtype=tf.float32), 1)
        # landms_pos = tf.reshape(landms_pos, (-1,))
        # landms = tf.reshape(landms, (-1, 10))
        # gt_landms = tf.reshape(gt_landms, (-1, 10))
        loss_landm = smooth_l1_loss(gt_landms[landms_pos], landms[landms_pos])

        pos = gt_confs != 0
        num_pos = tf.reduce_sum(tf.dtypes.cast(pos, tf.int32), axis=1)
        gt_confs = tf.where(pos, tf.constant(1, dtype=tf.int64), gt_confs)
        # locs = tf.reshape(locs, (-1, 4))
        # gt_locs = tf.reshape(gt_locs, (-1, 4))
        loss_loc = smooth_l1_loss(gt_locs[pos], locs[pos])

        batch_conf = tf.reshape(confs, (-1, self.num_classes))
        gt_confs = tf.reshape(gt_confs, (-1, 1))
        b_index = tf.stack([tf.range(gt_confs.shape[0], dtype=tf.int64)[:, None], gt_confs], axis=2)
        loss_c = tf.reduce_logsumexp(batch_conf, axis=1, keepdims=True) - tf.gather_nd(batch_conf, b_index)
        gt_confs = tf.reshape(gt_confs, [batch_size, -1])
        loss_c = tf.reshape(loss_c, [batch_size, -1])
        loss_c = tf.where(pos, tf.constant(0, dtype=tf.float32), loss_c)

        # Hard negative mining
        num_neg = num_pos * self.neg_ratio
        rank = tf.argsort(loss_c, axis=1, direction='DESCENDING')
        rank = tf.argsort(rank, axis=1)
        neg = rank < tf.expand_dims(num_neg, 1)
        loss_conf = cross_entropy(gt_confs[tf.math.logical_or(pos, neg)], confs[tf.math.logical_or(pos, neg)])
        loss_conf = tf.cast(loss_conf, tf.float32)

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos, tf.float32))
        num_pos = tf.math.maximum(num_pos, 1)
        loss_landm /= num_pos_landm
        loss_loc /= num_pos
        loss_conf /= num_pos

        return loss_landm, loss_loc, loss_conf

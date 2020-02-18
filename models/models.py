import tensorflow as tf
from backbones.resnet_v1_fpn import ContextModule, FPN


class RetinaFace(tf.keras.Model):
    def __init__(self, num_class=2, anchor_per_scale=2):
        super(RetinaFace, self).__init__()
        self.num_class = num_class
        self.fpn = FPN()
        self.cm = [ContextModule() for _ in range(3)]

        self.cls_conv = [tf.keras.layers.Conv2D(num_class * anchor_per_scale, (1, 1), padding='valid', activation="softmax") for _ in range(3)]
        self.box_conv = [tf.keras.layers.Conv2D(4 * anchor_per_scale, (1, 1), padding='valid') for _ in range(3)]
        self.lmk_conv = [tf.keras.layers.Conv2D(10 * anchor_per_scale, (1, 1), padding='valid') for _ in range(3)]

    def call(self, inputs):
        # FPN
        features = self.fpn(inputs)
        # Context Module
        x = [self.cm[i](features[i]) for i in range(len(features))]

        cls = [self.cls_conv[i](x[i]) for i in range(len(features))]
        box = [self.box_conv[i](x[i]) for i in range(len(features))]
        lmk = [self.lmk_conv[i](x[i]) for i in range(len(features))]

        # # no param part, for calc convenience
        # cls = [tf.reshape(cls[i], (cls[i].shape[0], -1, self.num_class)) for i in range(len(features))]
        # cls = tf.concat(cls, axis=1)
        # cls = self.softmax(cls)

        # box = [tf.reshape(box[i], (box[i].shape[0], -1, 4)) for i in range(len(features))]
        # box = tf.concat(box, axis=1)

        # lmk = [tf.reshape(lmk[i], (lmk[i].shape[0], -1, 10)) for i in range(len(features))]
        # lmk = tf.concat(lmk, axis=1)

        return cls, box, lmk
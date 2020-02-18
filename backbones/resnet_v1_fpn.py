import tensorflow as tf
from backbones.resnet_v1 import ResNet_v1_50


def conv_bn1X1(out_channels=256, stride=1, leaky=0):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out_channels, 1, strides=stride, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=leaky)
    ])


def conv_bn(out_channels=256, stride=1, leaky=0):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out_channels, 3, strides=stride, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=leaky)
    ])
    

def conv_bn_no_relu(out_channels=256, stride=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out_channels, 3, strides=stride, padding='same'),
        tf.keras.layers.BatchNormalization()
    ])


class ContextModule(tf.keras.Model):
    def __init__(self):
        super(ContextModule, self).__init__()
        self.conv3X3 = conv_bn_no_relu(256 // 2)

        self.conv5X5_1 = conv_bn(256 // 4)
        self.conv5X5_2 = conv_bn_no_relu(256 // 4)
        
        self.conv7X7_2 = conv_bn(256 // 4)
        self.conv7x7_3 = conv_bn_no_relu(256 // 4)

        self.relu = tf.keras.layers.ReLU()

        # self.conv1 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        # self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        # self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        # self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        

    def call(self, inputs):
        conv3X3 = self.conv3X3(inputs)

        conv5X5_1 = self.conv5X5_1(inputs)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = tf.concat([conv3X3, conv5X5, conv7X7], axis=-1)
        out = self.relu(out)
        return out


class FPN(tf.keras.Model):
    """Feature Pyramid Network - https://arxiv.org/abs/1612.03144"""

    def __init__(self, backbone=ResNet_v1_50):
        super(FPN, self).__init__()
        self.backbone = backbone()

        self.output1 = conv_bn1X1()
        self.output2 = conv_bn1X1()
        self.output3 = conv_bn1X1()

        self.merge1 = conv_bn()
        self.merge2 = conv_bn()

        self.top_down = tf.keras.layers.UpSampling2D(size=(2, 2))


    def call(self, inputs):
        c3, c4, c5 = self.backbone(inputs)  # c2: (None, 160, 160, 256), c3: (None, 80, 80, 512), c4: (None, 40, 40, 1024), c5: (None, 20, 20, 2048)

        output1 = self.output1(c3)
        output2 = self.output2(c4)
        output3 = self.output3(c5)

        up3 = self.top_down(output3)
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = self.top_down(output2)
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3

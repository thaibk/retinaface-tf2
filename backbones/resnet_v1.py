import tensorflow as tf


class Bottleneck(tf.keras.layers.Layer):

    def __init__(self, filters=64, strides=(1, 1)):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding='same', strides=strides)
        self.bn4 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if x.shape == inputs.shape:
            res = inputs
        else:
            res = self.conv4(inputs)
            res = self.bn4(res)
        x += res
        x = self.relu(x)
        return x


class ResNet_v1_50(tf.keras.Model):
    def __init__(self, Block=Bottleneck, layers=(3, 4, 6, 3)):
        super(ResNet_v1_50, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')
        self.blocks1 = tf.keras.Sequential([Block(filters=64, strides=(1, 1)) for _ in range(layers[0])])
        self.blocks2 = tf.keras.Sequential(
            [Block(filters=128, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[1])])
        self.blocks3 = tf.keras.Sequential(
            [Block(filters=256, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[2])])
        self.blocks4 = tf.keras.Sequential(
            [Block(filters=512, strides=(2, 2) if i < 1 else (1, 1)) for i in range(layers[3])])


    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.blocks1(x)
        c3 = self.blocks2(c2)
        c4 = self.blocks3(c3)
        c5 = self.blocks4(c4)

        return c3, c4, c5
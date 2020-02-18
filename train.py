import tensorflow as tf
import os

from losses import MultiBoxLoss
from dataset import DataGenerator
from models import RetinaFace
from config import *


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


@tf.function
def train_step(images, gt_locs, gt_confs, gt_landms, model, criterion, optimizer):
    with tf.GradientTape() as tape:
        confs, locs, landms = model(images)
        confs = [tf.reshape(confs[i], (confs[i].shape[0], -1, 2)) for i in range(len(confs))]
        confs = tf.concat(confs, axis=1)
        locs = [tf.reshape(locs[i], (locs[i].shape[0], -1, 4)) for i in range(len(locs))]
        locs = tf.concat(locs, axis=1)
        landms = [tf.reshape(landms[i], (landms[i].shape[0], -1, 10)) for i in range(len(landms))]
        landms = tf.concat(landms, axis=1)

        loss_landm, loss_loc, loss_conf = criterion(locs, confs, landms, gt_locs, gt_confs, gt_landms)
        loss = loc_weight * loss_loc + loss_landm + loss_conf

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, loss_landm, loss_loc, loss_conf


if __name__ == "__main__":
    train_gen = DataGenerator().generate()
    model = RetinaFace()
    ckpt_path = tf.train.latest_checkpoint(os.path.join(PROJECT_DIR, 'checkpoint'))
    print(ckpt_path)
    if ckpt_path is not None:
        print("INFO: load ckpt from {}...".format(ckpt_path))
        model.load_weights(ckpt_path)
    criterion = MultiBoxLoss().cal_loss
    # optimizer = tf.keras.optimizers.SGD(lr=0.0005, decay=5e-4, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epoch_start, num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0

        for i, (images, gt_locs, gt_confs, gt_landms) in enumerate(train_gen):
            loss, loss_landm, loss_loc, loss_conf = train_step(images, gt_locs, gt_confs, gt_landms, model, criterion, optimizer)
            
            if (i + 1) % 50 == 0:
                if (i + 1) % 500 == 0:
                    checkpoint = os.path.join(PROJECT_DIR, 'checkpoint/weights-{}-{}-{:.3f}'.format(epoch+1, i+1, loss_conf))
                    model.save_weights(checkpoint)
                print('Epoch: {} Batch {} | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f} Landm: {:.4f}'.format(epoch + 1, i + 1, loss, loss_conf, loss_loc, loss_landm))
from explorations.utils.all_imports import *
from explorations.u_net import unet
from explorations.utils.utils import data_gen_small
import tensorflow as tf

data_dir = "../small_dataset/train/"
mask_dir = "../small_dataset/train_masks/"
all_images = os.listdir(data_dir)
batch_size = 32
epochs = 10


def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    isct = tf.reduce_sum(y_true * y_pred)
    return (2 * isct + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

train_images, validation_images = train_test_split(all_images, train_size=0.8)

model = unet()

train_gen = data_gen_small(data_dir, mask_dir, train_images, batch_size)
val_gen = data_gen_small(data_dir, mask_dir, validation_images, batch_size)

c1 = ModelCheckpoint("../models/unet_128_best.h5", monitor='val_dice_coef')

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
model.fit_generator(train_gen, steps_per_epoch = int(len(train_images)/batch_size), validation_data = val_gen, \
                    validation_steps = int(len(validation_images)/batch_size), callbacks=[c1])



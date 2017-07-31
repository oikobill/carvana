""" This is a script that contains some utility functions"""
from explorations.utils.all_imports import *

def dice_coef(y_pred, y_true):
    """
    Given a candidate mask and the ground truth, calculate the dice coefficient.
    The assumption is that y_pred and y_true are 3D numpy tensors.
    """
    y_pred = np.round(np.array(y_pred).flatten()) # convert to 0s and 1s the candidate mask
    y_true = np.array(y_true).flatten()

    isct = np.sum(y_pred * y_true)
    denom = np.sum(y_pred) + np.sum(y_true)

    dice = (2. * isct) / denom

    return dice

def to_rgb(im):
    """Converting grey-scale images to RGB"""
    return np.dstack([im] * 3).copy(order='C')

# taken from this very helpful kernel https://www.kaggle.com/zfturbo/baseline-optimal-mask
def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    '''
    bytes = np.where(img.flatten() == 1)[0]
    runs = []
    prev = -2
    for b in bytes:
        if (b > prev + 1): runs.extend((b + 1, 0))
        runs[-1] += 1
        prev = b

    return ' '.join([str(i) for i in runs])

# A generator that takes in the images and a batch_size and returns a generator for the data
def data_gen_small(data_dir, mask_dir, images, batch_size):
        while True:
            ix = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                img = img_to_array(load_img(data_dir + images[i])) / 255
                imgs.append(img)
                label = (img_to_array(load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')) / 255)[:, :, 0].reshape(128, 128, 1)
                labels.append(label)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels

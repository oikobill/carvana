# This is meant to be a baseline submission. Predict the average mask in the dataset.
# Purpose is to establish a baseline and get all the util functions working.

from explorations.utils.all_imports import *

mask_dir = "../small_dataset/train_masks/"
mask_files = os.listdir(mask_dir)
n_masks = len(mask_files)

avg_mask = np.zeros([128, 128])

for f in tqdm(mask_files):
    filename = mask_dir+f
    img = img_to_array(load_img(filename))[:, :, 0]/255
    avg_mask += img


avg_mask /= n_masks

avg_mask = np.rint(avg_mask)

assert np.sum(avg_mask.flatten()==1) + np.sum(avg_mask.flatten()==0) == len(avg_mask.flatten())

np.save("../models/avg_mask", avg_mask)



from explorations.utils.all_imports import *
from explorations.utils.utils import dice_coef

# Evaluate how well the average mask does
avg_mask = np.load("../models/avg_mask.npy")

dice_results = []
mask_dir = "../small_dataset/train_masks/"
mask_files = os.listdir(mask_dir)

for f in tqdm(mask_files):
    filename = mask_dir+f
    img = img_to_array(load_img(filename))[:, :, 0]/255
    dice_results.append(dice_coef(img, avg_mask))

print("Mean dice coefficient in the smaller dataset is: {}".format(np.mean(dice_results)))
print("SD dice coefficient in the smaller dataset is: {}".format(np.std(dice_results)))

# Evaluate how well the average mask does
avg_mask_big = np.load("../models/avg_mask_big.npy")

dice_results = []
mask_dir = "../data/train_masks/"
mask_files = os.listdir(mask_dir)

for f in tqdm(mask_files):
    filename = mask_dir+f
    img = img_to_array(load_img(filename))[:, :, 0]/255
    dice_results.append(dice_coef(img, avg_mask_big))

print("Mean dice coefficient in the big dataset is: {}".format(np.mean(dice_results)))
print("SD dice coefficient in the big dataset is: {}".format(np.std(dice_results)))
from explorations.utils.all_imports import *

data_dir = "data/train/"
mask_dir = "data/train_masks/"

target_data_dir = "small_dataset/train/"
target_mask_dir = "small_dataset/train_masks/"

data_files = os.listdir(data_dir)
mask_files = os.listdir(mask_dir)

target_size = (128, 128)

# move all the image files
for f in tqdm(data_files):
    filename = data_dir+f
    # read the image
    img = load_img(filename)
    # convert to an array
    arr_img = img_to_array(img)
    # resize image to the smaller target
    resized_img = imresize(arr_img, target_size)
    # save the image to the new directory
    imsave(target_data_dir+f, resized_img)

# move all the mask files
for f in tqdm(mask_files):
    filename = mask_dir+f
    # read the image
    img = load_img(filename)
    # convert to an array
    arr_img = img_to_array(img)
    # resize image to the smaller target
    resized_img = imresize(arr_img, target_size)
    # save the image to the new directory
    imsave(target_mask_dir+f, resized_img)


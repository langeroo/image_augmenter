import os
import sys
import shutil

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image

###################################################################################################
########################################## SETTINGS ###############################################
###################################################################################################

num_augmentations_per_image = 10;
verbose = False;

###################################################################################################
##################################### GO GET SOME IMAGES ##########################################
###################################################################################################

# where are your source images?
if len(sys.argv) < 2:
    print("Usage python3 augment_all_photos.py /path/to/your/folder");
    quit();
orig_dir = sys.argv[1];
if not os.path.isdir(orig_dir):
    print(orig_dir + " isn't a directory. Try again.");
    quit();

# copy the folder and augment them in the <whatever>-augmented folder    
walk_dir = orig_dir.strip().strip('/') + "-augmented";
if os.path.isdir(walk_dir):
    shutil.rmtree(walk_dir);
shutil.copytree(orig_dir, walk_dir)

# define what image files look like
extensions = ['.bmp','.pbm','.pgm','.ppm','.sr','.ras','.jpeg',
              '.jpg','.jpe','.jp2','.tiff','.tif','.png'];

# let's find us some images
input_images = [];
for root, dirs, files in os.walk(walk_dir, topdown=False):
    for name in files:
        extension = os.path.splitext(name)[-1];
        if extension in extensions:
            input_images.append(os.path.join(root, name))

# did you find any images? Quit if not
if len(input_images) == 0:
    print("Didn't find any images in " + orig_dir + " or any of the folders contained therein.");
    print("Looking for images with the following extensions:");
    for ext in extensions:
        print("\t" + ext)
    quit();

# report the discovered images to work with
print("Found " + str(len(input_images)) + " image(s)!")
for img in input_images:
        print("\t" + img)

###################################################################################################
####################################### AUGMENT IMAGES ############################################
###################################################################################################

if verbose:
    print("Starting image augmentation of " + str(len(input_images)) + " image(s)...");
    print("Will create " + str(num_augmentations_per_image) + " augmented image(s) per input image.");

input_counter = 0;

for input_image in input_images:
    if verbose:
        print("Starting processing on image " + str(input_counter+1) + " of " + str(len(input_images)))

    image = Image.open(input_image)
    # convert image to numpy array
    data = np.asarray(image)
    # summarize shape
    images = np.array(([data] * num_augmentations_per_image), dtype=np.uint8)

    ia.seed(1)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    images_aug = seq(images=images)

    aug_counter = 0;
    elements = os.path.splitext(input_image);

    for new_img in images_aug:
        if verbose:
            print("\tSaving augmented image " + str(aug_counter+1) + " of " + str(len(images_aug)) )
        # build filename
        out_filename = elements[0] + "-aug-" + str(aug_counter).zfill(4) + elements[1];

        # save img
        img_out = Image.fromarray(new_img)
        img_out.save(out_filename)

        aug_counter += 1;
    input_counter += 1;

#!/usr/bin/env python3

import os
import sys

import cv2
import numpy as np
import progressbar
from imutils import paths
from skimage.transform import resize

from subroutines.HDF5 import HDF5DatasetWriterMask

folder = {}
# Use this to easily run the code in different directories/devices
folder["initial"] = "/home/robot/extars/"
# The path where the repository is stored
folder["main"] = folder["initial"] + "crack_detection_CNN_masonry/"
folder["images"] = folder["main"] + "dataset/custom_images/"
folder["masks"] = folder["main"] + "dataset/custom_masks/"
outputPath = folder["main"] + "output/hdf5/custom/test.hdf5"
IMAGE_DIMS = (224, 224, 3)

if folder["main"] == "":
    folder["main"] = os.getcwd()
sys.path.append(folder["main"])

# grab the paths to the images and masks
testPaths = list(paths.list_images(folder["images"]))
testLabels = list(paths.list_images(folder["masks"]))

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = (testPaths, testLabels)

# create HDF5 writer
print("[INFO] building {}...".format(outputPath))
writer = HDF5DatasetWriterMask(
    (len(testPaths), IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]), outputPath
)

# initialize the progress bar
widgets = [
    "Building Dataset: ",
    progressbar.Percentage(),
    " ",
    progressbar.Bar(),
    " ",
    progressbar.ETA(),
]
pbar = progressbar.ProgressBar(maxval=len(testPaths), widgets=widgets).start()

# loop over the image paths
for (ii, (im_path, mask_path)) in enumerate(zip(testPaths, testLabels)):

    # load the image and process it
    image = cv2.imread(im_path)

    # resize image if dimensions are different
    if IMAGE_DIMS != image.shape:
        image = resize(image, (IMAGE_DIMS), mode="constant", preserve_range=True)

    # normalize intensity values: [0,1]
    image = image / 255

    # label
    mask = cv2.imread(mask_path, 0)

    # resize image if dimensions are different
    if IMAGE_DIMS[0:2] != mask.shape:
        mask = resize(
            mask,
            (IMAGE_DIMS[0], IMAGE_DIMS[1]),
            mode="constant",
            preserve_range=True,
        )

    # normalize intensity values: [0,1]
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / 255

    # add the image and label to the HDF5 dataset
    writer.add([image], [mask])

    # update progress bar
    pbar.update(ii)

# close the progress bar and the HDF5 writer
pbar.finish()
writer.close()

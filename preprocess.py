"""
Diogo Amorim, 2018-07-13
Pré-processamento do Dataset LiTS para tumores no fígado - Vnet

- Converts .nii file to a resized ndarray
- HU augmentation
- Resize layers
- Normalization [0, 1]
- ...
"""

import os
import re

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
import scipy.misc
import scipy.ndimage

INPUT_SIZE = 128  # Input feature width/height
INPUT_DEPTH = 64  # Input depth

""" K. Sahi, S. Jackson, E. Wiebe, G. Armstrong, S. Winters, R. Moore, et al., ”The value of liver windows
settings in the detection of small renal cell carcinomas on unenhanced computed tomography,”
Canadian Association of Radiologists Journal, vol. 65, pp. 71-76, 2014."""
MIN_HU = -160  # Min HU Value
MAX_HU = 240  # Max HU Value


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def hu_window(image):
    image[image < MIN_HU] = MIN_HU
    image[image > MAX_HU] = MAX_HU
    image = (image - MIN_HU) / (MAX_HU - MIN_HU)
    image = image.astype("float32")
    return image


def scale_volume(volume, img_depth=INPUT_DEPTH, img_px_size=INPUT_SIZE, hu_value=True):
    if hu_value:
        volume = hu_window(volume)

    size_scale_factor = img_px_size / volume.shape[0]
    depth_scale_factor = img_depth / volume.shape[-1]

    # Use ndimage directly instead of deprecated interpolation
    volume = ndimage.rotate(volume, 90, reshape=False)
    vol_zoom = ndimage.zoom(volume, [size_scale_factor, size_scale_factor, depth_scale_factor], order=1)

    vol_zoom[vol_zoom < 0] = 0
    vol_zoom[vol_zoom > 1] = 1
    return vol_zoom


def scale_segmentation(segmentation, img_depth=INPUT_DEPTH, img_px_size=INPUT_SIZE):
    size_scale_factor = img_px_size / segmentation.shape[0]
    depth_scale_factor = img_depth / segmentation.shape[-1]

    segmentation = ndimage.rotate(segmentation, 90, reshape=False)
    # Nearest neighbor interpolation for segmentation masks
    seg_zoom = ndimage.zoom(segmentation, [size_scale_factor, size_scale_factor, depth_scale_factor], order=0)
    return seg_zoom


def get_data(vol_dir, seg_dir, crop=False):
    img_vol = nib.load(vol_dir)
    img_seg = nib.load(seg_dir)

    # updated nibabel call
    volume = img_vol.get_fdata().astype(np.float32)
    segmentation = img_seg.get_fdata().astype(np.uint8)

    if crop:
        aux = []
        for i in range(segmentation.shape[2]):
            if np.sum(segmentation[:, :, i]) > 0:
                aux.append(i)

        volume = volume[:, :, (np.min(aux) - 1):(np.max(aux) + 1)]
        segmentation = segmentation[:, :, (np.min(aux) - 1):(np.max(aux) + 1)]

    return volume, segmentation


def create_dataset(path, px_size=INPUT_SIZE, slice_count=INPUT_DEPTH, crop=False):
    """Returns dataset with shape (m, z, x, y, n)"""

    files = sorted_alphanumeric(os.listdir(path))
    segmentations = []
    volumes = []
    for name in files:
        if name[0] == 's':
            segmentations.append(name)
        elif name[0] == 'v':
            volumes.append(name)

    m = len(volumes)
    # m = 8
    if crop:
        print("Creating Cropped Data Set:")
    else:
        print("Creating Data Set:")

    slices = []
    print("0/%i (0%%)" % m)
    for i in range(m):
        volume, segmentation = get_data(path + volumes[i], path + segmentations[i], crop)

        volume = scale_volume(volume, slice_count, px_size)
        segmentation = scale_segmentation(segmentation, slice_count, px_size)
        slices.append([volume, segmentation])
        print("%i/%i (%i%%)" % (i+1, m, ((i+1)/m*100)))

    dataset = np.array(slices)

    print("Dataset finished with shape:")
    print(dataset.shape)

    return dataset

def write_dataset(data_set, path):
    """
    Save volume and multi-class segmentation mask.
    data_set shape = (N, 2, Z, X, Y)
    index 0 = image
    index 1 = segmentation mask
    """
    h5f = h5py.File(path, 'w')

    # Save CT volume
    h5f.create_dataset('data', data=np.expand_dims(data_set[:, 0], -1))

    # Save multi-class segmentation (0=background, 1=liver, 2=tumor)
    seg = np.expand_dims(data_set[:, 1], -1)
    seg = divide_segmentation(seg)
    h5f.create_dataset('truth', data=seg)

    h5f.close()
    print("Dataset saved @ %s" % path)


def divide_segmentation(segmentation):
    """
    FIXED:
    Keep original LiTS labels:
    0 = background
    1 = liver
    2 = tumor

    segmentation input shape: (N, Z, X, Y, 1)
    output: same shape with correct 0/1/2 classes
    """

    seg = segmentation[..., 0]       # remove last channel
    seg = seg.astype(np.uint8)

    # Ensure only valid classes remain
    seg[seg > 2] = 0

    # Add channel dimension back
    seg = np.expand_dims(seg, axis=-1)

    return seg


# data_dir = r"D:\Lits challange dataset"

# train_dir = os.path.join(data_dir, "train/")
# val_dir   = os.path.join(data_dir, "validation/")
# save_dir  = r'D:\SAM SEGMENTATION MASK\repo 6 bamf liver tumor segmentation\VNet\dataset'



# print("Obtaining Training Data:")
# train_set = create_dataset(train_dir, crop=True)
# write_dataset(train_set, os.path.join(save_dir, "train_data_at_128.h5"))

# print("Obtaining Validation Data:")
# val_set = create_dataset(val_dir, crop=True)
# write_dataset(val_set, os.path.join(save_dir, "val_data_at_128.h5"))

# print("Obtaining Test Data:")
# test_set = create_dataset(val_dir)
# write_dataset(test_set, save_dir + "test_data.h5")


"""
Single NII File Prediction & Visualization Pipeline
---------------------------------------------------
Steps:
 1) Preprocess CT volume + segmentation
 2) Predict using VNet model
 3) Visualize CT + GT + Prediction
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import (
    hu_window, scale_volume, scale_segmentation,
    INPUT_SIZE, INPUT_DEPTH
)
from vnet import vnet


# ----------------------------------------------------------
# LOAD RAW NII FILES
# ----------------------------------------------------------
def load_single_nii(volume_path, segmentation_path):
    """Load CT volume + segmentation mask."""
    vol = nib.load(volume_path).get_fdata().astype(np.float32)
    seg = nib.load(segmentation_path).get_fdata().astype(np.uint8)
    return vol, seg


# ----------------------------------------------------------
# PREPROCESS
# ----------------------------------------------------------
def preprocess_single_case(volume, segmentation, crop=True):
    """Apply SAME preprocessing used in training."""
    # Crop using segmentation
    if crop:
        slices = np.where(np.sum(segmentation, axis=(0, 1)) > 0)[0]
        if len(slices) > 0:
            mn = max(0, slices.min() - 1)
            mx = min(segmentation.shape[2] - 1, slices.max() + 1)
            volume = volume[:, :, mn:mx]
            segmentation = segmentation[:, :, mn:mx]

    # Resize + windowing
    volume_pp = scale_volume(volume, INPUT_DEPTH, INPUT_SIZE)
    seg_pp = scale_segmentation(segmentation, INPUT_DEPTH, INPUT_SIZE)

    # Match model input shape
    return volume_pp[..., np.newaxis], seg_pp[..., np.newaxis]


# ----------------------------------------------------------
# MODEL PREDICTION
# ----------------------------------------------------------

def predict_single(volume_processed, model_path, num_classes=None):
    """Run VNet model prediction. Returns class map (integers)."""
    model = vnet(input_size=(128, 128, 64, 1))
    # model = vnet(input_size=(128, 128, 64, 3))
    model.load_weights(model_path)

    # predict -> could be shape (1, X, Y, Z, C) or (1, X, Y, Z)
    pred = model.predict(volume_processed[np.newaxis, ...])
    pred = np.squeeze(pred)  # remove batch -> shape maybe (X, Y, Z, C) or (X, Y, Z)

    print(f"[debug] raw prediction shape: {pred.shape}, dtype: {pred.dtype}")

    # If model produced multi-channel probabilities (softmax) -> take argmax across last axis
    if pred.ndim == 4:
        # last dim = channels
        class_map = np.argmax(pred, axis=-1).astype(np.uint8)  # 0,1,2,...
    else:
        # single channel probability map (binary)
        class_map = (pred > 0.1).astype(np.uint8)

    print(f"[debug] class_map shape: {class_map.shape}, unique labels: {np.unique(class_map)}")
    return class_map  # shape (X,Y,Z)

# ----------------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------------

#     }
def visualize(volume_pp, seg_pp, class_map, slice_index=40):
    """Generate overlay visualizations. class_map contains integer labels."""
    ct =  volume_pp[:, :, slice_index, 0] # shape (128,128,64)
    gt = seg_pp[:, :, slice_index, 0]  # ground truth mask slice (if multi-channel, adjust)
    # class_map shape (128,128,64)
    slice_index = min(slice_index, class_map.shape[2]-1)
    pred_slice = class_map[:, :, slice_index]

    liver_pred = (pred_slice == 1).astype(np.uint8)
    tumor_pred = (pred_slice == 2).astype(np.uint8)

    def render_overlay(base, overlay=None):
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(base, cmap="gray")
        if overlay is not None:
            plt.imshow(overlay, cmap="jet", alpha=0.4)
        plt.axis("off")
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return img

    return {
        "ct": render_overlay(ct),
        "pred_liver": render_overlay(ct, liver_pred),
        "pred_liver_and_tumor": render_overlay(ct, gt),
        # "pred_tumor": render_overlay(ct, tumor_pred),
    }

def run_pipeline(volume_path, segmentation_path, weights_path,
                 slice_index=50, progress_callback=None):

    if progress_callback: progress_callback("Loading NII file...")
    vol, seg = load_single_nii(volume_path, segmentation_path)
    print("[debug] raw volume shape:", vol.shape, "raw seg shape:", seg.shape)

    if progress_callback: progress_callback("Preprocessing...")
    vol_pp, seg_pp = preprocess_single_case(vol, seg)
    print("[debug] preprocessed shapes:", vol_pp.shape, seg_pp.shape)

    if progress_callback: progress_callback("Model Predicting...")
    class_map = predict_single(vol_pp, weights_path)

    if progress_callback: progress_callback("Visualizing results...")
    imgs = visualize(vol_pp, seg_pp, class_map, slice_index)

    if progress_callback: progress_callback("Completed.")
    return imgs

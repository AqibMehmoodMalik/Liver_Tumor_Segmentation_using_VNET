"""
V-Net 3D implementation (TF2-compatible)
Updated for TensorFlow 2.x / tf.keras APIs.
"""

from typing import Tuple, Iterable

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, Conv3DTranspose, concatenate, add, PReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def downward_layer(input_layer, n_convolutions: int, n_output_channels: int):
    """
    Downward block:
      - n_convolutions of (Conv3D -> PReLU)
      - residual add with input (if channels match)
      - downsample via Conv3D(strides=2)
    Returns: downsampled tensor, skip-connection tensor
    """
    inl = input_layer
    for _ in range(n_convolutions):
        inl = Conv3D(filters=n_output_channels,
                     kernel_size=5,
                     padding='same',
                     kernel_initializer='he_normal')(inl)
        inl = PReLU()(inl)

    # Residual add (only if channels match)
    if inl.shape[-1] == input_layer.shape[-1]:
        add_l = add([inl, input_layer])
    else:
        add_l = inl

    downsample = Conv3D(filters=n_output_channels,
                        kernel_size=2,
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal')(add_l)
    downsample = PReLU()(downsample)
    return downsample, add_l


def upward_layer(input0, input1, n_convolutions: int, n_output_channels: int):
    """
    Upward block:
      - concatenate (input0 from previous upsample, input1 from skip)
      - n_convolutions of (Conv3D -> PReLU)
      - residual add with merged
      - learnable upsample via Conv3DTranspose (strides=2)
    """
    merged = concatenate([input0, input1], axis=-1)
    inl = merged
    for _ in range(n_convolutions):
        inl = Conv3D(
            filters=n_output_channels,
            kernel_size=5,
            padding='same',
            kernel_initializer='he_normal'
        )(inl)
        inl = PReLU()(inl)

    # Residual connection if channels match
    if inl.shape[-1] == merged.shape[-1]:
        add_l = add([inl, merged])
    else:
        add_l = inl

    upsample = Conv3DTranspose(
        filters=n_output_channels,
        kernel_size=2,
        strides=2,
        padding='same',
        kernel_initializer='he_normal'
    )(add_l)

    return PReLU()(upsample)


def vnet(input_size: Tuple[int, int, int, int] = (128, 128, 128, 1),
         optimizer=Adam(learning_rate=1e-4),
         loss='categorical_crossentropy',
         metrics=('accuracy',)):
    """
    Builds V-Net in tf.keras Functional API.
    """
    inputs = Input(shape=input_size)

    # Initial conv
    conv1 = Conv3D(16, kernel_size=5, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = PReLU()(conv1)

    # Downward path
    down1, add1 = downward_layer(conv1, n_convolutions=1, n_output_channels=32)
    down2, add2 = downward_layer(down1, n_convolutions=2, n_output_channels=64)
    down3, add3 = downward_layer(down2, n_convolutions=3, n_output_channels=128)
    down4, add4 = downward_layer(down3, n_convolutions=3, n_output_channels=256)

    # Bottleneck
    conv5 = down4
    for _ in range(3):
        conv5 = Conv3D(256, kernel_size=5, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = PReLU()(conv5)
    add5 = add([conv5, down4])

    # Upward path
    up5 = Conv3DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(add5)
    up5 = PReLU()(up5)

    up6 = upward_layer(up5, add4, n_convolutions=3, n_output_channels=128)
    up7 = upward_layer(up6, add3, n_convolutions=3, n_output_channels=64)
    up8 = upward_layer(up7, add2, n_convolutions=2, n_output_channels=32)

    # Final reconstruction
    merged9 = concatenate([up8, add1], axis=-1)
    conv9 = Conv3D(64, kernel_size=5, padding='same', kernel_initializer='he_normal')(merged9)
    conv9 = PReLU()(conv9)
    add9 = add([conv9, merged9])

    outputs = Conv3D(3, kernel_size=1, padding='same', activation='softmax', kernel_initializer='he_normal')(add9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics))

    return model
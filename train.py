
# # train.py
# import os
# from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# from utils import HDF5BatchGenerator
# from vnet import vnet

# data_dir = r"D:\SAM SEGMENTATION MASK\repo 6 bamf liver tumor segmentation\VNet\dataset"
# train_dir = os.path.join(data_dir, "train_data.h5")
# val_dir   = os.path.join(data_dir, "val_data.h5")
# save_dir  = data_dir
# weights_dir = os.path.join(save_dir, 'weights_vnet.weights.h5')

# # NEW — use generators instead of full loading
# train_gen = HDF5BatchGenerator(train_dir, batch_size=1, shuffle=True)
# val_gen   = HDF5BatchGenerator(val_dir, batch_size=1, shuffle=False)

# callbacks = [
#     ModelCheckpoint(weights_dir, monitor='val_loss', save_weights_only=True, save_best_only=True),
#     CSVLogger(os.path.join(save_dir, "training.log"), append=True),
#     ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
#     EarlyStopping(patience=2)
# ]

# model = vnet(input_size=(256, 256, 64, 1))

# model.fit(
#     train_gen,
#     epochs=8,
#     validation_data=val_gen,
#     callbacks=callbacks
# )

# model.save_weights(weights_dir)
# print("Training complete!")
import os
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from utils import HDF5BatchGenerator
from vnet import vnet

data_dir = r"D:\SAM SEGMENTATION MASK\repo 6 bamf liver tumor segmentation\VNet\dataset"
train_dir = os.path.join(data_dir, "train_data.h5")
val_dir   = os.path.join(data_dir, "val_data.h5")
save_dir  = data_dir
weights_dir = os.path.join(save_dir, 'weights_vnet.weights.h5')

# Use generators
train_gen = HDF5BatchGenerator(train_dir, batch_size=1, shuffle=True)
val_gen   = HDF5BatchGenerator(val_dir, batch_size=1, shuffle=False)

callbacks = [
    ModelCheckpoint(weights_dir, monitor='val_loss', save_weights_only=True, save_best_only=True),
    CSVLogger(os.path.join(save_dir, "training.log"), append=True),
    ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
    EarlyStopping(patience=2)
]

model = vnet(input_size=(256, 256, 64, 1))

model.fit(
    train_gen,
    epochs=8,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=callbacks
)

# Save model weights and full model optionally
model.save_weights(weights_dir)
model.save(os.path.join(save_dir, "vnet_full_model.h5"))
print("Training complete!")
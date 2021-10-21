# %%
# importing the libraries
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# version of tensorflow
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

# %%
data_path = 'C:/workspace/traindata/traindata'
train_datagen = ImageDataGenerator(

    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    validation_split=1 / 4,
    horizontal_flip=True,
    fill_mode='nearest',
    shear_range=0.2,
    zoom_range=0.2,

)

# normal genrator
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=1 / 4,

)

# Set batch size
batch_size = 30

train_data = train_datagen.flow_from_directory(
    data_path,
    target_size=(300, 300),
    subset='training',
    batch_size=batch_size
)

validation_data = train_datagen.flow_from_directory(
    data_path,
    target_size=(300, 300),
    subset='validation',
    batch_size=batch_size
)

train_size = train_data[0][0].shape[0]*len(train_data)
validation_size = validation_data[0][0].shape[0]*len(validation_data)
print(validation_size)


# %%
# reshaping the images
# shape of the training and test set
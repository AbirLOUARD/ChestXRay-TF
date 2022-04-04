import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator




train = 'chest_xray/train'
test = 'chest_xray/test'
val= 'chest_xray/val'


train_normal = len(os.listdir(os.path.join(train, 'NORMAL')))
train_pneumonia = len(os.listdir(os.path.join(train, 'PNEUMONIA')))
print("Train set:")
print(f"Normal:    {train_normal}")
print(f"Pneumonia: {train_pneumonia}")

test_normal = len(os.listdir(os.path.join(test, 'NORMAL')))
test_pneumonia = len(os.listdir(os.path.join(test, 'PNEUMONIA')))
print("\nTest set:")
print(f"Normal:    {test_normal}")
print(f"Pneumonia: {test_pneumonia}")

img1 = 'chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg'
imggg = plt.imread(img1)

imggg_3d = tf.expand_dims(imggg,2)

brighten = tf.image.adjust_brightness(imggg_3d, delta=0.2)
contrasten = tf.image.adjust_contrast(imggg_3d, 2.)
print(imggg.shape, imggg_3d.shape)

plt.imshow(imggg)
plt.title("Original img")
plt.show()

plt.imshow(brighten)
plt.title("Brightened img")
plt.show()

plt.imshow(contrasten)
plt.title("Contranst enhanced img")
plt.show()

BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = 240, 240
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
#Epoch
N = 10
STEPS_PER_EPOCH = 4173 / BATCH_SIZE
VALIDATION_STEPS = 1043 / BATCH_SIZE
METRICS = ['accuracy']


def custom_augmentation(np_tensor):
    def random_contrast(np_tensor):
        return tf.image.random_contrast(np_tensor, 0.5, 2)

    augmnted_tensor = random_contrast(np_tensor)
    return np.array(augmnted_tensor)


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   brightness_range=(1, 1.2),
                                   preprocessing_function=custom_augmentation,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_ds = train_datagen.flow_from_directory(train,
                                             subset = "training",
                                             class_mode = 'binary',
                                             seed = 123,
                                             target_size = IMG_SIZE,
                                             batch_size = BATCH_SIZE)

val_ds = train_datagen.flow_from_directory(train,
                                           subset = "validation",
                                           class_mode = 'binary',
                                           shuffle = False,
                                           seed = 123,
                                           target_size = IMG_SIZE,
                                           batch_size = BATCH_SIZE)

test_ds = test_datagen.flow_from_directory(test,
                                           class_mode = 'binary',
                                           shuffle = False,
                                           target_size = IMG_SIZE,
                                           batch_size = BATCH_SIZE)

model = Sequential([

    layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = optimizers.RMSprop(learning_rate = 1e-4),
                loss = 'binary_crossentropy',
                metrics = METRICS)
model.summary()
model.fit_generator(train_ds, epochs=5, validation_data=val_ds)

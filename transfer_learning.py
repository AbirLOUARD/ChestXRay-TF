import os
from glob import glob
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Flatten
from keras.models import Model
import tensorflow as tf
from keras.models import Sequential

#importer la base de données
data = "/Users/l.abir/Documents/ChestXray/chest_xray"

#le chemin des bases de données train et test

x_train = os.path.join(data, "train")
x_test= os.path.join(data, "test")


IMG_SHAPE = (224, 224, 3)
folders = glob(x_train+ '/*')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=IMG_SHAPE),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3),activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(folders), activation="softmax")
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#Créer les photos augmentées
y_train = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
y_test = ImageDataGenerator(rescale= 1./255)
trainning_set= y_train.flow_from_directory(x_train, target_size=(224,224), batch_size=32, class_mode='categorical')
test_set= y_test.flow_from_directory(x_test, target_size=(224,224), batch_size=32, class_mode='categorical')
model.fit_generator(trainning_set, epochs=5, validation_data=test_set)

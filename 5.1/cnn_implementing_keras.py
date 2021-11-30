from keras import optimizers
from keras.optimizer_v1 import Adam
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import numpy as np
from keras.datasets import mnist
import warnings
import matplotlib.pyplot as plt
### preproccesings

warnings.filterwarnings("ignore")

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#prepare the data
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalization
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

#dimensions
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#label encoding - one hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(5, 5), padding="same", activation="relu", input_shape= (28, 28, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

#defining optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#compile the model
model.compile(optimizer = optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 10
batch_size = 250

#data augmentation
datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=0.5,
                            zoom_range=0.5,
                            width_shift_range=0.5,
                            height_shift_range=0.5,
                            horizontal_flip=False,
                            vertical_flip=False,
                            )

#fit the model
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                epochs= epochs, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0] // batch_size)



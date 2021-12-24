from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

#load and preprocess
def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)
    data = data.to_numpy()
    np.random.shuffle(data)
    x = data[:,1:].reshape(-1, 28, 28, 1) / 255.0
    y = data[:,0].astype(np.int32)
    y = to_categorical(y, num_classes=len(set(y)))

    return x, y


train_data_path = r"C:\Users\yagiz\Desktop\data\mnist\mnist_train.csv"
test_data_path = r"C:\Users\yagiz\Desktop\data\mnist\mnist_test.csv"

x_train, y_train = load_and_preprocess(train_data_path)
x_test, y_test = load_and_preprocess(test_data_path)

#visualizing
index = 3
vis = x_train.reshape(60000, 28, 28)
plt.imshow(vis[index, :, :])
plt.legend()
plt.axis("off")
plt.show()

print(np.argmax(y_train[index]))

#CNN
numberOfClasses = y_train.shape[1]

model = Sequential()

model.add(Conv2D(input_shape = (28, 28, 1), filters= 16, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size= (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters=128, kernel_size= (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units = 256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(units = numberOfClasses))
model.add(Activation("softmax"))

model.compile(loss= "categorical_crossentropy",
            optimizer="adam",
            metrics = ["accuracy"])

#train
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=4000)

model.save_weights("cnn_mnist_model.h5")

print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label= "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"], label = "Validation Accuracy")
plt.legend()
plt.show()






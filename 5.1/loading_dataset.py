import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
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

"""
print(classes[np.argmax(y_train[0])])
plt.imshow(x_train[0])
plt.show()
"""

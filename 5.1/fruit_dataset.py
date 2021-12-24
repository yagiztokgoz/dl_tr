#libraries
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob

train_path = r"C:\Users\yagiz\Desktop\data\fruits-360_dataset\fruits-360\Training"
test_path = r"C:\Users\yagiz\Desktop\data\fruits-360_dataset\fruits-360\Test"

img = load_img(train_path + r"\Apple Braeburn\0_100.jpg")

""" plt.imshow(img)
plt.axis("off")
plt.show() """

x = img_to_array(img)
print(x.shape)

""" className = glob(train_path + "\*")
cnt = 0
for cname in className:
    print(cname)
    cnt += 1
print(cnt) """
#131


#CNN Model

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(x.shape)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(131))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
            optimizer= "rmsprop",
            metrics=["accuracy"])

batch_size = 32 


#Data Generation - Train - Test
train_datagen = ImageDataGenerator(rescale=1./255,
                shear_range=0.3,
                horizontal_flip=True,
                zoom_range=0.3,
                ) 

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path, 
                                                target_size=x.shape[:2],
                                                batch_size=batch_size, 
                                                color_mode="rgb",
                                                class_mode="categorical")

test_generator = test_datagen.flow_from_directory(test_path,
                                                target_size=x.shape[:2],
                                                batch_size=batch_size,
                                                color_mode="rgb",
                                                class_mode="categorical")

hist = model.fit_generator(generator=train_generator,
                steps_per_epoch= 1600 // batch_size,
                epochs=2,
                validation_data=test_generator,
                validation_steps=800 // batch_size)


#model save
""" model.save_weights("deneme.h5") """

print(hist.history.keys())

plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Val Loss")
plt.legend()
plt.show()

plt.plot(hist.history["accuracy"], label = "Acc Loss")
plt.plot(hist.history["val_accuracy"], label = "Acc Loss")
plt.legend()
plt.show()



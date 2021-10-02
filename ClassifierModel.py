import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import matplotlib.pyplot as plt
import cv2 as op

plt.style.use('ggplot')

class Classifier:

    def __init__(self, img_size, train_path):
        self.IMG_SIZE = img_size
        self.TRAIN_PATH = train_path
        self.base_model = MobileNetV2(input_shape=self.IMG_SIZE + (3,), include_top=False, weights='imagenet')

    def data_augmentation(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_set = train_datagen.flow_from_directory(
            r'C:\Users\Asus\Desktop\ML (self work)\Deep Learning\Projects\Road\Datasets\train',
            # Provide path to dataset,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )

    def train(self):
        self.base_model.trainable = False

        self.data_augmentation()

        x = Flatten()(self.base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        prediction = Dense(2, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input, outputs=prediction)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            self.train_set,
            epochs=5,
            steps_per_epoch=len(self.train_set)
        )

    def predict_image(self, img):
        img = img.reshape(1, self.IMG_SIZE[0],self.IMG_SIZE[1], 3)
        img = img/255
        prediction_info = self.model.predict(img)
        return "Clean Road" if np.argmax(prediction_info) == 0 else "Dirty Road"

    def plot_loss_accuracy(self):
        # Plot the loss
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.legend()
        plt.show()

        # Plot the accuracy
        plt.plot(self.history.history['accuracy'], label='train_acc')
        plt.legend()
        plt.show()

    def save_model(self):
        self.model.save("Road_Classification.h5")


if __name__ == '__main__':
    main()

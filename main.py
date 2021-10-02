from ClassifierModel import Classifier
import cv2 as op
import numpy as np
import matplotlib.pyplot as plt

def input_image(path, size):
    img = op.imread(path)
    img = op.resize(img, size)
    return img


def main():
    img_size = (224, 224)
    train_path = r'C:\Users\Asus\Desktop\ML (self work)\Deep Learning\Projects\Road\Datasets'

    model = Classifier(img_size, train_path)

    model.train()

    img_path = r'C:\Users\Asus\Desktop\ML (self work)\Deep Learning\Projects\Road\Evaluate\6.jpg'
    image = input_image(img_path, img_size)

    prediction = model.predict_image(image)

    print(prediction)


if __name__ == '__main__':
    main()

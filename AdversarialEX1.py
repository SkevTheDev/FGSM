import tensorflow as tf
import sys
from Utils import Utils
from DeepModel import DeepModel
from Adversarial import Adversarial
from tensorflow.keras.callbacks import LambdaCallback

import numpy as np
import random

import matplotlib.pyplot as plt

def main():
    print(tf.__version__)
    utils = Utils()
    dmodel = DeepModel()
    adver = Adversarial()
    
    x_train, y_train, x_test, y_test = utils.read_mnist_data()
    labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    img_rows = 28
    img_cols = 28
    channels = 1
    num_classes = 10
    model = dmodel.create_model(img_rows, img_cols, channels, num_classes)  # img_rows, img_cols
    print(model.summary())
    model = dmodel.train_model(model, x_train, y_train, x_test, y_test, 5, 32)

    # Assess base model accuracy on regular images
    print("Base accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))

    # Create a single adversarial example
    image = x_train[0]
    image_label = y_train[0]
    channels = 1
    adver_image = adver.create_adversarial_example(model, img_rows, img_cols, image, channels, image_label, 0.1)

    print(labels[model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()])
    print(labels[model.predict(adver_image).argmax()])

    if channels == 1:
        plt.imshow(adver_image.reshape((img_rows, img_cols)))
    else:
        plt.imshow(adver_image.reshape((img_rows, img_cols, channels)))
    plt.show()

    # Generate and visualize 12 adversarial images
    adversarials, correct_labels = next(adver.generate_adversarials(model, x_train,y_train,img_rows,img_cols,channels,12))
    for adversarial, correct_label in zip(adversarials, correct_labels):
        print('Prediction:', labels[model.predict(adversarial.reshape((1, img_rows, img_cols, channels))).argmax()], 'Truth:', labels[correct_label.argmax()])
        if channels == 1:
            plt.imshow(adversarial.reshape(img_rows, img_cols))
        else:
            plt.imshow(adversarial)
        plt.show()

    # Generate adversarial data
    # x_adversarial, y_adversarial = np.load("x_adv_10k.npy"), np.load("y_adv_10k.npy")
    x_adversarial_train, y_adversarial_train = next(adver.generate_adversarials(model,x_train, y_train,img_rows,img_cols,channels,20000))
    x_adversarial_test, y_adversarial_test = next(adver.generate_adversarials(model, x_train, y_train,img_rows,img_cols,channels,10000))

    # Assess base model on adversarial data
    print("Base accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

    # Learn from adversarial data
    model.fit(x_adversarial_train, y_adversarial_train,
              batch_size=32,
              epochs=10,
              validation_data=(x_test, y_test))

    # Assess defended model on adversarial data
    print("Defended accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

    # Assess defended model on regular data
    print("Defended accuracy on regular images:", model.evaluate(x=x_test, y=y_test, verbose=0))

    x_adversarial_test, y_adversarial_test = next(adver.generate_adversarials(model,x_train, y_train,img_rows,img_cols,channels,10000))
    print("Defended accuracy on adversarial images:", model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0))

if __name__ == "__main__":
    sys.exit(int(main() or 0))



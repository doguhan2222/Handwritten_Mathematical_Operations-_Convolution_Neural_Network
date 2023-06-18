import keras
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
import imutils
from imutils.contours import sort_contours
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

class AlexNet:

    def __init__(self):
        self.x = []
        self.y = []
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []

    def load_dataset(self):
        datadir = 'data/dataset'
        for folder in os.listdir(datadir):
            path = os.path.join(datadir, folder)
            for images in os.listdir(path):
                img = cv2.imread(os.path.join(path, images))
                self.x.append(img)
                self.y.append(folder)

    def print_dataset(self):
        print(len(self.x))
        print(len(self.y))
        print(f'labels : {list(set(self.y))}')

    def visualizing_dataset(self):
        figure = plt.figure(figsize=(10, 10))
        j = 0
        for i in list(set(self.y)):
            idx = self.y.index(i)
            img = self.x[idx]
            img = cv2.resize(img, (256, 256))
            figure.add_subplot(5, 5, j + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(i)
            j += 1
        plt.show()

    def preprocessing_data(self):
        X = []
        for i in range(len(self.x)):
            #     print(i)
            img = self.x[i]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            threshold_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            #threshold_image = cv2.resize(threshold_image, (32, 32))
            threshold_image = cv2.resize(threshold_image, (227, 227))
            X.append(threshold_image)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(self.y)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, y, test_size=0.2)

    # def data_distrubution(self):
    #
    #     unique_train, count_train = np.unique(self.Y_train, return_counts=True)
    #     figure = plt.figure(figsize=(20, 10))
    #     sn.barplot(unique_train, count_train).set_title('Number of Images per category in Train Set')
    #     plt.show()

    def defining_model(self):
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.Y_train = np.array(self.Y_train)
        self.Y_test = np.array(self.Y_test)

        self.Y_train = to_categorical(self.Y_train)
        self.Y_test = to_categorical(self.Y_test)
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)
        self.X_train = self.X_train / 255.
        self.X_test = self.X_test / 255.

    def alexNet(self):

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=96, kernel_size=(11, 11),
                                      strides=(4, 4), activation="relu",
                                      input_shape=(227, 227, 1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5),
                                      strides=(1, 1), activation="relu",
                                      padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3),
                                      strides=(1, 1), activation="relu",
                                      padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3),
                                      strides=(1, 1), activation="relu",
                                      padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                      strides=(1, 1), activation="relu",
                                      padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(16, activation="softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.SGD(lr = 0.001),
                      metrics=['accuracy'])

        return model

    def model_summary(self, model):

        return model.summary()


    def training_model(self, model):

        aug = ImageDataGenerator(zoom_range=0.1,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05)

        hist = model.fit(aug.flow(self.X_train, self.Y_train, batch_size=128), batch_size=128, epochs=1,
                         validation_data=(self.X_test, self.Y_test))

        return hist

    def loss_and_accuracy_plot(self, hist):
        figure = plt.figure(figsize=(10, 10))
        plt.plot(hist.history['accuracy'], label='Train Set Accuracy')
        plt.plot(hist.history['val_accuracy'], label='Test Set Accuracy')
        plt.title('Accuracy Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        plt.show()

        figure2 = plt.figure(figsize=(10, 10))
        plt.plot(hist.history['loss'], label='Train Set Loss')
        plt.plot(hist.history['val_loss'], label='Test Set Loss')
        plt.title('Loss Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.legend(loc='upper right')
        plt.show()

    def classification_report(self, model):

        ypred = model.predict(self.X_test)
        ypred = np.argmax(ypred, axis=1)
        Y_test_hat = np.argmax(self.Y_test, axis=1)
        reports = [ypred, Y_test_hat]
        print(classification_report(Y_test_hat, ypred))
        return reports

    def cnfs_matrix(self, reports):

        matrix = confusion_matrix(reports[1], reports[0])
        df_cm = pd.DataFrame(matrix, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                             columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        figure = plt.figure(figsize=(20, 10))
        sn.heatmap(df_cm, annot=True, fmt='d')

    def saving_model(self, model):
        model.save('alexNet_ann_hw.h5')

    @staticmethod
    def test_equation(image_path, model):

        chars = []
        img = cv2.imread(image_path)
        img = cv2.resize(img, (800, 800))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        edged = cv2.Canny(img_gray, 30, 150)
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sort_contours(contours, method="left-to-right")[0]
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'mul', 'sub', 'z left', 'z right']

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if 20 <= w and 30 <= h:
                roi = img_gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                (th, tw) = thresh.shape
                if tw > th:
                    thresh = imutils.resize(thresh, width=227)
                if th > tw:
                    thresh = imutils.resize(thresh, height=227)
                (th, tw) = thresh.shape
                dx = int(max(0, 227 - tw) / 2.0)
                dy = int(max(0, 227 - th) / 2.0)
                padded = cv2.copyMakeBorder(thresh, top=dy, bottom=dy, left=dx, right=dx,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
                padded = cv2.resize(padded, (227, 227))
                padded = np.array(padded)
                padded = padded / 255.
                padded = np.expand_dims(padded, axis=0)
                padded = np.expand_dims(padded, axis=-1)
                pred = model.predict(padded)
                pred = np.argmax(pred, axis=1)
                #         print(pred)
                label = labels[pred[0]]
                chars.append(label)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, label, (x - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        figure = plt.figure(figsize=(10, 10))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        e = ''
        for i in chars:
            if i == 'add':
                e += '+'
            elif i == 'sub':
                e += '-'
            elif i == 'mul':
                e += '*'
            elif i == 'div':
                e += '/'
            elif i == 'z left':
                e += '('
            elif i == 'z right':
                e += ')'
            else:
                e += i
        v = eval(e)
        print('Value of the expression {} : {}'.format(e, v))
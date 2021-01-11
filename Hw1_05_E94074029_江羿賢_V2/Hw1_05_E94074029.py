from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model
import random
import pandas as pd
from skimage.transform import resize

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def click1():

    def show_label(img_data, label_data):
        label_array = ["airplane", "automobile", "bird", "cat",
                       "deer", "dog", "frog", "horse", "ship", "truck"]
        fig = plt.gcf()
        for i in range(0, 10):
            the_random = random.randint(0, 50000)
            to_show = plt.subplot(2, 5, i + 1)
            to_show.imshow(img_data[the_random], cmap='binary')
            the_label = label_array[label_data[the_random][0]]
            to_show.set_title(the_label)
            to_show.set_xticks([])
            to_show.set_yticks([])
        plt.show()
    show_label(x_train, y_train)


def click2():
    print("hyperparameters:")
    print("batch size: 64")
    print("learning rate: 0.01")
    print("optimizer: SGD")


def click3():
    img_model = cv2.imread("model.png")
    cv2.imshow("model", img_model)


def click4():
    img_accuracy = cv2.imread("accuracy.png")
    cv2.imshow("accuracy", img_accuracy)


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(367, 300)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(110, 30, 131, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(110, 70, 131, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(110, 110, 131, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(110, 150, 131, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(110, 190, 131, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(110, 230, 131, 23))
        self.pushButton_5.setObjectName("pushButton_5")

        def click5():
            label_array = ["plane", "car", "bird", "cat",
                           "deer", "dog", "frog", "horse", "ship", "truck"]
            model = load_model('transfer_cifar10.h5')
            get_input = self.lineEdit.text()
            the_input = int(get_input)
            img = resize(x_test[the_input], (512, 512))
            cv2.namedWindow('Image')
            cv2.imshow('Image', img)

            img_p = np.array(x_test[the_input])
            img_p = resize(img_p, (64, 64))
            img_p = img_p.reshape(-1, 64, 64, 3)
            val = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            prediction = model.predict(img_p)
            Final_prediction = [result.argmax() for result in prediction][0]
            Final_prediction = label_array[Final_prediction]
            count = 0
            for i in prediction[0]:
                val[count] = i
                count = count+1
            x = np.arange(len(label_array))
            plt.bar(x, val, tick_label=label_array)
            plt.ylim([0, 1])
            plt.show()

        self.pushButton.clicked.connect(click1)
        self.pushButton_2.clicked.connect(click2)
        self.pushButton_3.clicked.connect(click3)
        self.pushButton_4.clicked.connect(click4)
        self.pushButton_5.clicked.connect(click5)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Hw1_05"))
        self.pushButton.setText(_translate("Form", "1. Show Train Images"))
        self.pushButton_2.setText(_translate(
            "Form", "2. Show Hyperparameters"))
        self.pushButton_3.setText(_translate(
            "Form", "3. Show Model Structure"))
        self.pushButton_4.setText(_translate("Form", "4. Show Accuracy"))
        self.pushButton_5.setText(_translate("Form", "5. Test"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Form()

    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

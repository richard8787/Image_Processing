# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
import glob
from PyQt5.QtWidgets import QApplication
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(399, 215)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(20, 30, 361, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 70, 361, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 110, 361, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(20, 150, 361, 23))
        self.pushButton_4.setObjectName("pushButton_4")

        def click1():
            img1 = cv2.imread('data/loss_accuracy.png')
            cv2.imshow('loss and accuarcy', img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def click2():
            img2 = cv2.imread('data/TensorBoard.png')
            cv2.imshow('TensorBoard', img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def click3():
            datapath1 = 'data/sample/test/dogs/'
            datapath2 = 'data/sample/test/cats/'
            rare = '.jpg'
            choose = random.randint(1, 2)
            if choose == 1:
                datapath = datapath1
                img_title = 'Class:dog'
            else:
                datapath = datapath2
                img_title = 'Class:cat'
            choose = random.randint(1401, 1500)

            img_datapath = datapath + str(choose) + rare
            img = Image.open(os.path.join(img_datapath))
            plt.figure('Randomly Select')
            plt.imshow(img)
            plt.axis('on')
            plt.title(img_title)
            plt.show()

        def click4():
            plt.figure('comparison table of accuracy')
            plt.bar(['Before Resize', 'After Resize'], [97.12, 98.87], 0.8)
            plt.title('Resize augmentation comparison')
            plt.show()
        self.pushButton.clicked.connect(click1)
        self.pushButton_2.clicked.connect(click2)
        self.pushButton_3.clicked.connect(click3)
        self.pushButton_4.clicked.connect(click4)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Hw2_05"))
        self.pushButton.setText(_translate(
            "Form", "5.1 show loss and accuracy (code is in source_code folder)"))
        self.pushButton_2.setText(_translate(
            "Form", "5.2 show screenshot of TensorBoard"))
        self.pushButton_3.setText(_translate(
            "Form", "5.3 show Randomly select picture from the test set"))
        self.pushButton_4.setText(_translate(
            "Form", "5.4 show the comparison table of accuracy (code is in source_code folder)"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Form()

    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

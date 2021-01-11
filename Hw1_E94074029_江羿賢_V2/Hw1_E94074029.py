from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication


def click1():  # 1_1
    img1_1 = cv2.imread("Uncle_Roger.jpg")
    cv2.imshow('Uncle_Roger', img1_1)
    height = img1_1.shape[0]
    width = img1_1.shape[1]
    print('Height = ', height)
    print('Width = ', width)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click2():  # 1_2
    img1_2 = cv2.imread("Flower.jpg")

    imgB = img1_2.copy()
    imgB[:, :, 1] = 0
    imgB[:, :, 2] = 0

    imgG = img1_2.copy()
    imgG[:, :, 0] = 0
    imgG[:, :, 2] = 0

    imgR = img1_2.copy()
    imgR[:, :, 0] = 0
    imgR[:, :, 1] = 0

    cv2.imshow('Original Image', img1_2)
    cv2.imshow('B', imgB)
    cv2.imshow('G', imgG)
    cv2.imshow('R', imgR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click3():  # 1_3
    img1_3_origin = cv2.imread("Uncle_Roger.jpg")
    img1_3_flip = cv2.flip(img1_3_origin, 1)
    cv2.imshow('Original Image', img1_3_origin)
    cv2.imshow('Result', img1_3_flip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click4():  # 1_4
    cv2.namedWindow('BLENDING')
    cv2.createTrackbar('BLEND', 'BLENDING', 0, 255, click4_function)
    cv2.setTrackbarPos('BLEND', 'BLENDING', 128)
    while (1):
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def click4_function(x):  # 1_4 function
    img1_4_origin = cv2.imread("Uncle_Roger.jpg")
    img1_4_flip = cv2.flip(img1_4_origin, 1)
    para1 = cv2.getTrackbarPos('BLEND', 'BLENDING') / 255
    para2 = 1 - para1
    result = cv2.addWeighted(img1_4_origin, para1, img1_4_flip, para2, 0)
    cv2.imshow('BLENDING', result)


def click5():  # 2_1
    img2_1 = cv2.imread("Cat.png")
    img2_1_median = cv2.medianBlur(img2_1, 7)
    cv2.imshow('Original Image', img2_1)
    cv2.imshow('median', img2_1_median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click6():  # 2_2
    img2_2 = cv2.imread("Cat.png")
    img2_2_gaussian = cv2.GaussianBlur(img2_2, (3, 3), 0)
    cv2.imshow('Original Image', img2_2)
    cv2.imshow('Gaussian', img2_2_gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click7():  # 2_3
    img2_3 = cv2.imread("Cat.png")
    img2_3_bilateral = cv2.bilateralFilter(img2_3, 9, 90, 90)
    cv2.imshow('Original Image', img2_3)
    cv2.imshow('Bilateral', img2_3_bilateral)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click8():  # 3_1
    img3_1 = cv2.imread("Chihiro.jpg")
    img3_1_gray = cv2.imread("Chihiro.jpg", 0)
    x, y = np.mgrid[-1:2, -1:2]
    kernel_g = np.exp(-(x*x+y*y))
    kernel_g = kernel_g / kernel_g.sum()
    result_g = cv2.filter2D(img3_1_gray, -1, kernel_g)

    cv2.imshow('Chihiro.jpg', img3_1)
    cv2.imshow('Grayscale', img3_1_gray)
    cv2.imshow('Gaussian Blur', result_g)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click9():  # 3_2
    img3_2 = cv2.imread("Chihiro.jpg", 0)

    x, y = np.mgrid[-1:2, -1:2]
    kernel_g = np.exp(-(x*x+y*y))
    kernel_g = kernel_g / kernel_g.sum()
    result_g = cv2.filter2D(img3_2, -1, kernel_g)

    kernel = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="float32")
    result = cv2.filter2D(result_g, -1, kernel)

    cv2.imshow('Gaussian Blur', result_g)
    cv2.imshow('Sobel X', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click10():  # 3_3
    img3_3 = cv2.imread("Chihiro.jpg", 0)

    x, y = np.mgrid[-1:2, -1:2]
    kernel_g = np.exp(-(x*x+y*y))
    kernel_g = kernel_g / kernel_g.sum()
    result_g = cv2.filter2D(img3_3, -1, kernel_g)

    kernel = np.array((
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]), dtype="float32")
    result = cv2.filter2D(result_g, -1, kernel)

    cv2.imshow('Gaussian Blur', result_g)
    cv2.imshow('Sobel Y', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def click11():  # 3_4
    img3_4 = cv2.imread("Chihiro.jpg", 0)

    x, y = np.mgrid[-1:2, -1:2]
    kernel_g = np.exp(-(x*x+y*y))
    kernel_g = kernel_g / kernel_g.sum()
    result_g = cv2.filter2D(img3_4, -1, kernel_g)

    kernel_x = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="float32")
    sobelx = cv2.filter2D(result_g, -1, kernel_x)

    kernel_y = np.array((
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]), dtype="float32")
    sobely = cv2.filter2D(result_g, -1, kernel_y)

    result = cv2.addWeighted(sobelx, 1, sobely, 1, 0)

    cv2.imshow('Sobel X', sobelx)
    cv2.imshow('Sobel Y', sobely)
    cv2.imshow('Magnitude', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ui


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(737, 344)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 40, 160, 291))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_2.addWidget(self.pushButton_4)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_2.setGeometry(
            QtCore.QRect(180, 40, 160, 291))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_3.addWidget(self.pushButton_5)
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_3.addWidget(self.pushButton_6)
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_3.addWidget(self.pushButton_7)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_3.setGeometry(
            QtCore.QRect(350, 40, 160, 291))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_3)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton_8 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout_4.addWidget(self.pushButton_8)
        self.pushButton_9 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout_4.addWidget(self.pushButton_9)
        self.pushButton_10 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.pushButton_10.setObjectName("pushButton_10")
        self.verticalLayout_4.addWidget(self.pushButton_10)
        self.pushButton_11 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.pushButton_11.setObjectName("pushButton_11")
        self.verticalLayout_4.addWidget(self.pushButton_11)
        self.gridLayoutWidget = QtWidgets.QWidget(Form)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(520, 40, 192, 291))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 3, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 2, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 3, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.pushButton_12 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_12.setObjectName("pushButton_12")
        self.gridLayout.addWidget(self.pushButton_12, 4, 0, 1, 3)
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setGeometry(QtCore.QRect(40, 10, 111, 31))
        self.label_8.setObjectName("label_8")
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setGeometry(QtCore.QRect(380, 10, 111, 31))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(560, 10, 111, 31))
        self.label_11.setObjectName("label_11")
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setGeometry(QtCore.QRect(210, 10, 111, 31))
        self.label_9.setObjectName("label_9")

        def click12():  # 4
            img4 = cv2.imread("Parrot.png")
            get_degree = self.lineEdit.text()
            get_scale = self.lineEdit_2.text()
            get_tx = self.lineEdit_3.text()
            get_ty = self.lineEdit_4.text()
            degree = float(get_degree)
            scale = float(get_scale)
            tx = float(get_tx)
            ty = float(get_ty)
            row = img4.shape[0]
            col = img4.shape[1]

            T_set = np.float32([[1, 0, tx], [0, 1, ty]])
            img_T = cv2.warpAffine(img4, T_set, (col, row))

            R_S_set = cv2.getRotationMatrix2D((160+tx, 84+ty), degree, scale)
            result = cv2.warpAffine(img_T, R_S_set, (col, row))

            cv2.imshow('Original Image', img4)
            cv2.imshow('Image RST', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.pushButton.clicked.connect(click1)
        self.pushButton_2.clicked.connect(click2)
        self.pushButton_3.clicked.connect(click3)
        self.pushButton_4.clicked.connect(click4)
        self.pushButton_5.clicked.connect(click5)
        self.pushButton_6.clicked.connect(click6)
        self.pushButton_7.clicked.connect(click7)
        self.pushButton_8.clicked.connect(click8)
        self.pushButton_9.clicked.connect(click9)
        self.pushButton_10.clicked.connect(click10)
        self.pushButton_11.clicked.connect(click11)
        self.pushButton_12.clicked.connect(click12)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "2020 Opencvdl HW1"))
        self.pushButton.setText(_translate("Form", "1.1 Load Image"))
        self.pushButton_2.setText(_translate("Form", "1.2 Color seperation"))
        self.pushButton_3.setText(_translate("Form", "1.3 Image Flipping"))
        self.pushButton_4.setText(_translate("Form", "1.4 Blending"))
        self.pushButton_5.setText(_translate("Form", "2.1 Median Filter"))
        self.pushButton_6.setText(_translate("Form", "2.2 Gaussian Blur"))
        self.pushButton_7.setText(_translate("Form", "2.3 Bilateral Filter"))
        self.pushButton_8.setText(_translate("Form", "3.1 Gaussian Blur"))
        self.pushButton_9.setText(_translate("Form", "3.2 Sobel X"))
        self.pushButton_10.setText(_translate("Form", "3.3 Sobel Y"))
        self.pushButton_11.setText(_translate("Form", "3.4 Magnitude"))
        self.label_4.setText(_translate("Form", "Tx:"))
        self.label.setText(_translate("Form", "Rotation:"))
        self.label_12.setText(_translate("Form", "pixel"))
        self.label_3.setText(_translate("Form", "Scaling:"))
        self.label_7.setText(_translate("Form", "pixel"))
        self.label_2.setText(_translate("Form", "deg"))
        self.label_5.setText(_translate("Form", "Ty:"))
        self.pushButton_12.setText(_translate("Form", "4. Transformation"))
        self.label_8.setText(_translate("Form", "1. Image Processing"))
        self.label_10.setText(_translate("Form", "3. Edge Detection"))
        self.label_11.setText(_translate("Form", "4. Transformation"))
        self.label_9.setText(_translate("Form", "2.Image Smoothing"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Form()

    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
# Ui end

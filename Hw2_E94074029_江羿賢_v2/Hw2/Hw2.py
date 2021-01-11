# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
choose = 0


class Ui_Form(object):
    def selectionchange(self, i):
        global choose
        choose = i

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(417, 352)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(20, 20, 81, 20))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(50, 50, 101, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 80, 101, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(20, 180, 71, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 210, 101, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(50, 240, 101, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(50, 270, 101, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(200, 210, 91, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(240, 230, 61, 16))
        self.label_4.setObjectName("label_4")
        self.comboBox = QtWidgets.QComboBox(Form)
        self.comboBox.setGeometry(QtCore.QRect(240, 250, 69, 22))
        self.comboBox.setObjectName("comboBox")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(240, 280, 121, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(200, 20, 111, 16))
        self.label_5.setObjectName("label_5")
        self.pushButton_7 = QtWidgets.QPushButton(Form)
        self.pushButton_7.setGeometry(QtCore.QRect(240, 50, 131, 23))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(200, 100, 121, 16))
        self.label_6.setObjectName("label_6")
        self.pushButton_8 = QtWidgets.QPushButton(Form)
        self.pushButton_8.setGeometry(QtCore.QRect(240, 130, 131, 23))
        self.pushButton_8.setObjectName("pushButton_8")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(30, 120, 161, 20))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setGeometry(QtCore.QRect(30, 140, 151, 16))
        self.label_8.setObjectName("label_8")

        def click1():
            image = cv2.imread('Datasets/Q1_Image/coin01.jpg')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            canny = cv2.Canny(blurred, 30, 150)
            (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            contours = image.copy()
            cv2.drawContours(contours, cnts, -1, (0, 0, 255), 2)
            cv2.imshow("Draw Contours 01", contours)

            image = cv2.imread('Datasets/Q1_Image/coin02.jpg')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            canny = cv2.Canny(blurred, 30, 150)
            (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            contours = image.copy()
            cv2.drawContours(contours, cnts, -1, (0, 0, 255), 2)
            cv2.imshow("Draw Contours 02", contours)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def click2():
            image = cv2.imread('Datasets/Q1_Image/coin01.jpg')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            canny = cv2.Canny(blurred, 30, 150)
            (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            self.label_7.setText(
                "There are " + str(len(cnts)) + " coins in coin01.jpg")

            image = cv2.imread('Datasets/Q1_Image/coin02.jpg')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            canny = cv2.Canny(blurred, 30, 150)
            (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            self.label_8.setText(
                "There are " + str(len(cnts)) + " coins in coin02.jpg")

        def click3():
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((11*8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            objpoints = []
            imgpoints = []
            images = glob.glob('Datasets/Q2_Image/*.bmp')
            i = 1
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
                    img_resize = cv2.resize(img, (1024, 1024))
                    cv2.imshow('Corner Detection' + str(i), img_resize)
                    i = i+1
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def click4():
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((11*8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            objpoints = []
            imgpoints = []
            images = glob.glob('Datasets/Q2_Image/*.bmp')
            i = 1
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
                    i = i+1
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            print(mtx)

        def click5():
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((11*8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            objpoints = []
            imgpoints = []
            images = glob.glob('Datasets/Q2_Image/*.bmp')
            i = 1
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
                    i = i+1
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            print(dist)

        def click6():
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((11*8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            objpoints = []
            imgpoints = []
            images = glob.glob('Datasets/Q2_Image/*.bmp')
            i = 1
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
                    i = i+1

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
            rv = np.zeros(shape=(3, 1))
            tv = np.zeros(shape=(3, 1))
            ex = np.zeros(shape=(3, 4))
            rv = rvecs[choose]
            tv = tvecs[choose]
            R, _ = cv2.Rodrigues(rv)

            for i in range(3):
                for j in range(4):
                    if (j != 3):
                        ex[i][j] = R[i][j]
                    else:
                        ex[i][j] = tv[i]
            print(ex)

        def click7():
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((11*8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            objpoints = []
            imgpoints = []
            images = glob.glob('Datasets/Q2_Image/*.bmp')
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

            images = glob.glob('Datasets/Q3_Image/*.bmp')
            axis = np.float32([[3, 3, -3], [1, 1, 0], [3, 5, 0], [5, 1, 0]])
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
                if ret == True:
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    retval, rvecs, tvecs = cv2.solvePnP(
                        objp, corners2, mtx, dist)
                    imgpts, _ = cv2.projectPoints(
                        axis, rvecs, tvecs, mtx, dist)
                    imgpts = np.int32(imgpts).reshape(-1, 2)
                    img = cv2.line(img, tuple(imgpts[0]),
                                   tuple(imgpts[1]), (0, 0, 255), 5)
                    img = cv2.line(img, tuple(imgpts[0]),
                                   tuple(imgpts[2]), (0, 0, 255), 5)
                    img = cv2.line(img, tuple(imgpts[0]),
                                   tuple(imgpts[3]), (0, 0, 255), 5)
                    img = cv2.line(img, tuple(imgpts[1]),
                                   tuple(imgpts[2]), (0, 0, 255), 5)
                    img = cv2.line(img, tuple(imgpts[1]),
                                   tuple(imgpts[3]), (0, 0, 255), 5)
                    img = cv2.line(img, tuple(imgpts[2]),
                                   tuple(imgpts[3]), (0, 0, 255), 5)
                    img_resize = cv2.resize(img, (1024, 1024))
                    cv2.imshow('pyramid', img_resize)
                    cv2.waitKey(500)
            cv2.destroyAllWindows()

        def click8():
            imgL = cv2.imread('Datasets/Q4_Image/imgL.png', 0)
            imgR = cv2.imread('Datasets/Q4_Image/imgR.png', 0)
            stereo = cv2.StereoBM_create(numDisparities=240, blockSize=21)
            disparity = stereo.compute(imgL, imgR)
            disparity = cv2.normalize(disparity, None, alpha=0,
                                      beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.namedWindow('disparity', 0)

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    disparity_get = disparity[y][x]
                    depth = int((2826 * 178) / (disparity_get + 123))
                    cv2.rectangle(disparity, (2420, 1800),
                                  (2820, 1920), (255, 255, 255), -1)
                    cv2.putText(disparity, 'Disparity: '+str(disparity_get)+' pixels',
                                (2420, 1850), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(disparity, 'Depth: '+str(depth)+' mm',
                                (2420, 1900), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('disparity', disparity)

            cv2.imshow('disparity', disparity)
            cv2.setMouseCallback('disparity', on_mouse)
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
        self.comboBox.addItems(
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])
        self.comboBox.currentIndexChanged.connect(self.selectionchange)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Hw2"))
        self.label.setText(_translate("Form", "1.Find Contour"))
        self.pushButton.setText(_translate("Form", "1.1 Draw Contour"))
        self.pushButton_2.setText(_translate("Form", "1.2 Count Coins"))
        self.label_2.setText(_translate("Form", "2. Calibration"))
        self.pushButton_3.setText(_translate("Form", "2.1 Find Corners"))
        self.pushButton_4.setText(_translate("Form", "2.2 Find Intrinsic"))
        self.pushButton_5.setText(_translate("Form", "2.4 Find Distortion"))
        self.label_3.setText(_translate("Form", "2.3 Find Extrinsic"))
        self.label_4.setText(_translate("Form", "Select image"))
        self.pushButton_6.setText(_translate("Form", "2.3 Find Extrinsic"))
        self.label_5.setText(_translate("Form", "3. Augmented Reality"))
        self.pushButton_7.setText(_translate("Form", "3.1 Augmented Reality"))
        self.label_6.setText(_translate("Form", "4. Stereo Disparity Map"))
        self.pushButton_8.setText(_translate(
            "Form", "4.1 Stereo Disparity Map"))
        self.label_7.setText(_translate(
            "Form", "There are __ coins in coin01.jpg"))
        self.label_8.setText(_translate(
            "Form", "There are __ coins in coin02.jpg"))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Form()

    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

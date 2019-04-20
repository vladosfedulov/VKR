import sys
import re
import threading
import queue
import timeit
import time

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from fpdf import FPDF

import design


event = threading.Event()


class Fil:

    def red_eye(img):
        eyes = []
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
        if eyes != []:
            for (ex, ey, ew, eh) in eyes:
                eye = img[ey:ey + eh, ex:ex + ew, :]
                b, g, r = cv2.split(img[ey:ey + eh, ex:ex + ew, :])
                bg = cv2.add(b, g)
                mask = (r > 145) & (r > bg)
                mask = mask.astype(np.uint8) * 255
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                mask = cv2.dilate(mask, None, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
                mean = bg / 2
                mean = np.around(mean, decimals=0)
                mean = np.array(mean, dtype=np.uint8)
                mask = mask.astype(np.bool)[:, :, np.newaxis]
                mean = mean[:, :, np.newaxis]
                eye_out = eye.copy()
                eye_out = np.where(mask, mean, eye_out)

                img[ey:ey + eh, ex:ex + ew] = eye_out
        return img

    def for_pdf(filenames, file_to_save):
        pdf = FPDF()
        for f in filenames:
            try:
                pdf.add_page()
                pdf.image(f, w=190)
            except Exception as err:
                print(err)
        try:
            pdf.output(file_to_save[0], "F")
        except Exception as err:
            print(err)

    def sharpness(img, arg):
        clahe = cv2.createCLAHE(clipLimit=arg, tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels

        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv2.merge((l2, a, b))  # merge channels
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        return img

    def brigth(img, arg):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = img[:, :, 2]
        if arg >= 0:
            v = np.where(v <= 255 - arg, v + arg, 255)
        if arg < 0:
            v = np.where(0 <= v + arg, v + arg, 0)
        img[:, :, 2] = v
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def channel(img, arg):
        if arg == 0:
            img[:, :, 0] = 0
            img[:, :, 1] = 0
        return img

    def saturation(img, arg):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = img[:, :, 1]
        if arg >= 0:
            v = np.where(v <= 255 - arg, v + arg, 255)
        if arg < 0:
            v = np.where(0 <= v + arg, v + arg, 0)
        img[:, :, 1] = v
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def white_ballance(wb, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if round(wb/10) > 0:
            img[:, :, 2] += round(wb / 10)
        if round(wb/10) < 0:
            img[:, :, 1] += round(wb / 10 * -1)
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        return img

    def bl_filter(img, arg, filter_name):
        cv2.setUseOptimized(True)
        if filter_name == 'Двустороннее':
            # сигма
            start_time = timeit.default_timer()
            img = cv2.bilateralFilter(img, 5, arg, arg)
            print(timeit.default_timer() - start_time)
        if filter_name == 'Гауссовское':
            # ядро
            start_time = timeit.default_timer()
            if arg % 2 != 0:
                img = cv2.GaussianBlur(img, (arg, arg), 0, 0)
            else:
                img = cv2.GaussianBlur(img, (arg - 1, arg - 1), 0, 0)
            print(timeit.default_timer() - start_time)
        if filter_name == 'Медианное':
            start_time = timeit.default_timer()
            if arg % 2 != 0:
                img = cv2.medianBlur(img, arg)
            else:
                img = cv2.medianBlur(img, arg - 1)
            print(timeit.default_timer() - start_time)
        return img


class Worker(threading.Thread):

    def __init__(self, work_queue_f, wire_f, params_f):

        super(Worker, self).__init__()
        self.work_queue = work_queue_f
        self.wire = wire_f
        self.params = params_f
        self.image = None
        self.time = time.time()
        print(time.time(), "start")

    def run(self):
        event.clear()
        while self.work_queue.qsize() != 0:
            filename_f = self.work_queue.get()
            print(filename_f)

            if filename_f:
                self.process(filename_f)
        else:
            print(time.time(), " end ", time.time() - self.time)

    def process(self, filename_p):
        self.image = cv2.imread(filename_p)
        for filter_f in self.wire:
            list_proc = re.split(r' = ', filter_f)
            if list_proc[0] == 'WB':
                self.image = Fil.white_ballance(int(list_proc[1]), self.image)
            if (list_proc[0] == 'Filter: Гауссовское') | (list_proc[0] == 'Filter: Медианное') | \
                    (list_proc[0] == 'Filter: Двухстороннее'):
                fn = re.sub(r'Filter: ', '', list_proc[0])
                self.image = Fil.bl_filter(self.image, int(list_proc[1]), fn)
            if list_proc[0] == 'Brigth':
                self.image = Fil.brigth(self.image, int(list_proc[1]))
            if list_proc[0] == 'Sat':
                self.image = Fil.saturation(self.image, int(list_proc[1]))
            if list_proc[0] == 'Sharpness':
                self.image = Fil.sharpness(self.image, int(list_proc[1]))
            if list_proc[0] == 'Red eye':
                self.image = Fil.red_eye(self.image)
            if filter_f == 'Save':
                self.pic_save(filename_p)

        # print(filename_p, '\t', self.wire, '\t', threading.current_thread())
        event.set()

    def pic_save(self, filename_p):
        file_name = ''
        if self.params[3] == 'Оставить':
            sp_filename = re.split(r'/', filename_p)
            f_name = re.split(r'\.', sp_filename[-1])
            file_name = self.params[1] + '/' + f_name[0] + '.' + self.params[0]
        else:
            sp_filename = re.split(r'/', filename_p)
            sp_filename = re.split(r'\.', sp_filename[-1])
            file_name = self.params[1] + '/' + sp_filename[0] + '_NEW.' + self.params[0]

        size = self.image.shape
        qformat = QImage.Format_RGB888
        img = QImage(self.image, size[1], size[0], qformat)
        img = img.rgbSwapped()
        pixmap = QPixmap.fromImage(img)
        pixmap.save(file_name, format=self.params[0], quality=int(self.params[2]))


class ExampleApp2(QtWidgets.QMainWindow, design.Ui_MainWindow2):
    def __init__(self, arg1, arg2):

        super().__init__()
        self.setupUi(self)
        self.files = arg1
        self.wire = arg2
        self.hz_pos = 1

        self.pushButton.clicked.connect(self.btn_clk)
        self.pushButton_2.clicked.connect(self.btn2_clk)
        self.horizontalSlider.valueChanged.connect(self.hzslider)

    def hzslider(self, int_t):
        self.hz_pos = int_t
        self.label_6.setText(str(int_t))

    def btn_clk(self):
        file_name = QFileDialog.getExistingDirectory()
        self.textEdit.setText(file_name)

    def btn2_clk(self):
        params = list([])
        params.append(self.comboBox.currentText())
        params.append(self.textEdit.toPlainText())
        params.append(str(self.spinBox.value()))
        params.append(self.comboBox_2.currentText())
        print(params)
        if params[1] == '':
            self.textEdit.setStyleSheet("QTextEdit {background-color:red}")
            self.textEdit.setText('Выберите каталог для сохранения')
        else:
            # threads
            work_queue = queue.Queue()
            for filename in self.files:
                work_queue.put(filename)
            for i in range(0, self.hz_pos):
                print(i)
                worker = Worker(work_queue, self.wire, params)
                worker.start()


class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.window2 = None
        self.files = ['jpg', 'JPG', 'PNG', 'png', 'BMP', 'bmp']
        self.wb = 0
        self.wire = []
        self.image = []
        self.image_copy = []
        self.file_name = []
        self.image_not_compressed = []
        self.arg = 0
        self.brigth = 0
        self.saturation = 0
        self.sharpness = 0
        self.xy_press = [0, 0]
        self.xy_released = [0, 0]
        self.size_pixmap = [0, 0]
        self.size_picture = [0, 0]
        self.scal = 1
        self.roi = []

        self.pushButton.clicked.connect(self.btn_clk)
        self.pushButton_2.clicked.connect(self.btn2_clk)
        self.pushButton_3.clicked.connect(self.btn3_clk)
        self.pushButton_4.clicked.connect(self.btn4_clk)
        self.pushButton_5.clicked.connect(self.btn5_clk)
        self.pushButton_6.clicked.connect(self.btn6_clk)
        self.horizontalSlider.valueChanged.connect(self.hzslider)
        self.horizontalSlider_2.valueChanged.connect(self.hzslider_2)
        self.horizontalSlider.sliderReleased.connect(self.hzslider_ev)
        self.horizontalSlider_2.sliderReleased.connect(self.hzslider2_ev)
        self.comboBox.activated.connect(self.cb)
        self.horizontalSlider_3.valueChanged.connect(self.hzslider_3)
        self.horizontalSlider_3.sliderReleased.connect(self.hzslider3_ev)
        self.horizontalSlider_4.valueChanged.connect(self.hzslider_4)
        self.horizontalSlider_4.sliderReleased.connect(self.hzslider4_ev)
        self.horizontalSlider_5.valueChanged.connect(self.hzslider_5)
        self.horizontalSlider_5.sliderReleased.connect(self.hzslider5_ev)
        self.horizontalSlider_6.valueChanged.connect(self.hzslider_6)
        self.horizontalSlider_7.valueChanged.connect(self.hzslider_7)
        self.horizontalSlider_6.sliderReleased.connect(self.hzslider67_ev)
        self.horizontalSlider_7.sliderReleased.connect(self.hzslider67_ev)

    def mousePressEvent(self, e):
        x = e.x()
        y = e.y()
        self.xy_press[:] = round((y - 10) * self.scal), round((x - 10) * self.scal)
        if self.xy_press[0] < 0:
            self.xy_press[0] = 0
        if self.xy_press[0] > self.size_picture[0]:
            self.xy_press[0] = self.size_picture[0]
        if self.xy_press[1] < 0:
            self.xy_press[1] = 0
        if self.xy_press[1] > self.size_picture[1]:
            self.xy_press[1] = self.size_picture[1]

    def mouseReleaseEvent(self, e):
        x = e.x()
        y = e.y()
        self.xy_released[:] = round((y - 10) * self.scal), round((x - 10) * self.scal)
        if self.xy_released[0] < 0:
            self.xy_released[0] = 0
        if self.xy_released[0] > self.size_picture[0]:
            self.xy_released[0] = self.size_picture[0]
        if self.xy_released[1] < 0:
            self.xy_released[1] = 0
        if self.xy_released[1] > self.size_picture[1]:
            self.xy_released[1] = self.size_picture[1]
        if self.xy_press != self.xy_released:
            self.load_image(self.image)

    def cb(self, filter_name):
        if filter_name == 0:  # Gauss
            self.label_7.setText('15')
            self.label_8.setText('Sigma')
            self.horizontalSlider_2.setMinimum(1)
            self.horizontalSlider_2.setMaximum(15)
            self.horizontalSlider_2.setSingleStep(2)
        if filter_name == 1:  # Median
            self.label_7.setText('7')
            self.label_8.setText('Sigma')
            self.horizontalSlider_2.setMinimum(1)
            self.horizontalSlider_2.setMaximum(7)
            self.horizontalSlider_2.setSingleStep(2)
        if filter_name == 2:  # Bilateral
            self.label_7.setText('100')
            self.label_8.setText('Aperture')
            self.horizontalSlider_2.setMinimum(1)
            self.horizontalSlider_2.setMaximum(100)
            self.horizontalSlider_2.setSingleStep(2)

    def hzslider67_ev(self):
        self.image = self.image_copy.copy()

    def hzslider_7(self, int_t):
        int_t *= -1
        rgb = 0
        if self.comboBox2.currentText() == 'R':
            rgb = 2
        if self.comboBox2.currentText() == 'G':
            rgb = 1
        if self.comboBox2.currentText() == 'B':
            rgb = 0

        if self.image != []:
            self.image_copy = self.image.copy()
            if int_t > 0:
                if self.size_picture[0] > self.xy_released[0]+int_t:
                    self.image_copy[self.xy_press[0]:self.xy_released[0], self.xy_press[1]:self.xy_released[1], rgb] = self.image[self.xy_press[0]+int_t:self.xy_released[0]+int_t, self.xy_press[1]:self.xy_released[1], rgb]
            if int_t < 0:
                if self.xy_press[0]-int_t > 0:
                    self.image_copy[self.xy_press[0]-int_t:self.xy_released[0]-int_t, self.xy_press[1]:self.xy_released[1], rgb] = self.image[self.xy_press[0]:self.xy_released[0], self.xy_press[1]:self.xy_released[1], rgb]
            self.load_image(self.image_copy)

    def hzslider_6(self, int_t):
        int_t *= -1
        rgb = 0
        if self.comboBox2.currentText() == 'R':
            rgb = 2
        if self.comboBox2.currentText() == 'G':
            rgb = 1
        if self.comboBox2.currentText() == 'B':
            rgb = 0

        if self.image != []:
            self.image_copy = self.image.copy()
            if int_t < 0:
                if self.size_picture[1] > self.xy_released[1]+int_t:
                    self.image_copy[self.xy_press[0]:self.xy_released[0], self.xy_press[1]:self.xy_released[1], rgb] = self.image[self.xy_press[0]:self.xy_released[0], self.xy_press[1]+int_t:self.xy_released[1]+int_t, rgb]
            if int_t > 0:
                if self.xy_press[1]-int_t > 0:
                    self.image_copy[self.xy_press[0]:self.xy_released[0], self.xy_press[1]-int_t:self.xy_released[1]-int_t, rgb] = self.image[self.xy_press[0]:self.xy_released[0], self.xy_press[1]:self.xy_released[1], rgb]
            self.load_image(self.image_copy)

    def hzslider_5(self, int_t):
        self.sharpness = int_t
        if self.image != []:
            self.image_copy = Fil.sharpness(self.image, int_t / 20)
            self.load_image(self.image_copy)

    def hzslider5_ev(self):
        self.textEdit.append('Sharpness = ' + str(self.sharpness))
        self.image = self.image_copy.copy()

    def hzslider_3(self, int_t):
        self.brigth = int_t
        if self.image != []:
            self.image_copy = Fil.brigth(self.image, int_t)
            self.load_image(self.image_copy)

    def hzslider3_ev(self):
        self.textEdit.append('Brigth = ' + str(self.brigth))
        self.image = self.image_copy.copy()

    def hzslider_4(self, int_t):
        self.saturation = int_t
        if self.image != []:
            self.image_copy = Fil.saturation(self.image, int_t)
            self.load_image(self.image_copy)

    def hzslider4_ev(self):
        self.textEdit.append('Sat = ' + str(self.saturation))
        self.image = self.image_copy.copy()

    def hzslider_ev(self):
        self.textEdit.append('WB = ' + str(self.wb))
        self.image = self.image_copy.copy()

    def hzslider2_ev(self):
        self.textEdit.append('Filter: ' + self.comboBox.currentText() + ' = ' + str(self.arg))
        self.image = self.image_copy.copy()

    def hzslider_2(self, int_t):
        self.arg = int_t
        if self.image != []:
            self.image_copy = Fil.bl_filter(self.image, int_t, self.comboBox.currentText())
            self.load_image(self.image_copy)

    def hzslider(self, int_t):
        self.wb = int_t
        self.image_copy = Fil.white_ballance(int_t, self.image)
        self.load_image(self.image_copy)

    def btn4_clk(self):
        nmfiles = ''
        self.roi = []
        self.xy_released = []
        self.xy_press = []
        try:
            self.file_name = QFileDialog.getOpenFileNames(filter='*.jpg *.png *.bmp')
            if self.file_name[0]:
                if len(self.file_name[0]) == 1:
                    tpfile = re.split(r'\.', self.file_name[0][0])
                    self.textEdit.clear()
                    self.wire = []
                    self.horizontalSlider.setSliderPosition(0)
                    self.horizontalSlider_2.setSliderPosition(1)
                    if tpfile[1] in self.files:
                        self.image = cv2.imread(self.file_name[0][0])
                        self.size_picture[:] = self.image.shape[0], self.image.shape[1]
                        self.load_image(self.image)
                if len(self.file_name[0]) > 1:
                    for file in self.file_name[0]:
                        nmfile = re.split(r'/', file)
                        nmfiles += nmfile[-1] + '\n'

                    self.label_5.setGeometry(QtCore.QRect(10, 10, 100, 100))
                    self.label_5.setText('Файлов открыто: ' + str(len(nmfiles)))
        except Exception as err:
            print(err)

    def btn_clk(self):
        file_name = QFileDialog.getSaveFileName(filter='*.txt')
        if file_name[0]:
            file = open(file_name[0], 'w')
            text = self.textEdit.toPlainText()
            file.write(text)
            file.close()

    def btn2_clk(self):
        file_name = QFileDialog.getOpenFileName(filter='*.txt')
        if file_name[0]:
            self.textEdit.clear()
            file = open(file_name[0], 'rt')
            lst = [line.strip() for line in file]
            for element in lst:
                self.textEdit.append(element)

    def btn3_clk(self):
        if len(self.file_name[0]) == 1:
            self.textEdit.toPlainText()
            lst = re.split(r'\n', self.textEdit.toPlainText())
            self.image_proc(lst)
        if len(self.file_name[0]) > 1:
            lst = re.split(r'\n', self.textEdit.toPlainText())
            lst.append('Save')
            self.window2 = ExampleApp2(arg1=self.file_name[0], arg2=lst)
            self.window2.show()

    def btn6_clk(self):
        self.image = Fil.red_eye(self.image)
        self.textEdit.append('Red eye = true')
        self.load_image(self.image)

    def image_proc(self, lst):
        for element in lst:
            list_proc = re.split(r' = ', element)
            if list_proc[0] == 'WB':
                self.image = Fil.white_ballance(int(list_proc[1]), self.image)
                self.load_image(self.image)
            if (list_proc[0] == 'Filter: Гауссовское') | (list_proc[0] == 'Filter: Медианное') | \
                    (list_proc[0] == 'Filter: Двухстороннее'):
                fn = re.sub(r'Filter: ', '', list_proc[0])
                self.image = Fil.bl_filter(self.image, int(list_proc[1]), fn)
                self.load_image(self.image)
            if list_proc[0] == 'Brigth':
                self.image = Fil.brigth(self.image, int(list_proc[1]))
                self.load_image(self.image)
            if list_proc[0] == 'Sat':
                self.image = Fil.saturation(self.image, int(list_proc[1]))
                self.load_image(self.image)
            if list_proc[0] == 'Sharpness':
                self.image = Fil.sharpness(self.image, int(list_proc[1]))
                self.load_image(self.image)
            if list_proc[0] == 'Red eye':
                self.image = Fil.red_eye(self.image)
                self.load_image(self.image)

    def load_image(self, image):

        size = image.shape
        qformat = QImage.Format_RGB888
        try:
            img = QImage(image, size[1], size[0], qformat)
            img = img.rgbSwapped()
        except Exception as err:
            print(err)
        pixmap = QPixmap.fromImage(img)

        if (size[0] > 410) | (size[1] > 580):
            if pixmap.width() > pixmap.height():
                scal = 580
            else:
                scal = 410

            try:
                pixmap = pixmap.scaled(scal, scal, QtCore.Qt.KeepAspectRatio)

            except Exception as err:
                print(err)
        self.size_pixmap[:] = pixmap.height(), pixmap.width()
        self.scal = self.size_picture[0]/self.size_pixmap[0]

        if self.xy_released != []:
            if self.xy_press != self.xy_released:
                self.roi = image[self.xy_press[0]:self.xy_released[0], self.xy_press[1]:self.xy_released[1], :]
                cv2.imshow('ROI', self.roi)
        self.label_5.setPixmap(pixmap)
        self.label_5.resize(pixmap.width(), pixmap.height())

    def btn5_clk(self):
        format_image = ''
        if self.radioButton.isChecked():
            format_image = 'BMP'
        if self.radioButton_2.isChecked():
            format_image = 'JPG'
        if self.radioButton_3.isChecked():
            format_image = 'PNG'
        if self.radioButton_4.isChecked():

            file_to_save = QFileDialog.getSaveFileName(filter='*.pdf')
            print(file_to_save)
            print(self.file_name[0])
            if file_to_save[0]:
                Fil.for_pdf(self.file_name[0], file_to_save)
        else:
            self.image = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            image = self.image
            file_name = QFileDialog.getSaveFileName(filter='*.'+format_image)
            if file_name[0]:
                size = image.shape
                qformat = QImage.Format_RGB888
                img = QImage(image, size[1], size[0], qformat)
                img = img.rgbSwapped()
                pixmap = QPixmap.fromImage(img)
                pixmap.save(file_name[0], format=format_image, quality=self.spinBox.value())


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()

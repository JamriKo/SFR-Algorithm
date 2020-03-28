import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sfr_window
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import pyqtgraph as pg


class MyWindow(QMainWindow, sfr_window.Ui_SFR):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setupUi(self)

        self.load_image.clicked.connect(self.get_image)
        self.load_ocef_lut.clicked.connect(self.get_ocef_lut)




    def get_image(self):
        image_file, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', 'Image files (*.jpg *.gif *.png *.jpeg)')
        I = cv2.imread(image_file)  # 给定棋盘格图片尺寸为：weight:1444, height:1084
        # cv2.rectangle(I, (685, 420), (745, 480), (255, 0, 0), 3)
        # I_resize = cv2.resize(I, (361, 271))
        # I_resize = QImage(I_resize, 361, 271, QImage.Format_RGB888)
        # self.img_show.setPixmap(QPixmap.fromImage(I_resize))
        self.PHI = cv2.cvtColor(I[420:480, 685:745, :], cv2.COLOR_RGB2GRAY)
        self.DN = np.array(I[420:480, 685:745, :], dtype=float)
        self.DN_gray = QImage(self.PHI, 60, 60, QImage.Format_Indexed8)
        self.roi.setPixmap(QPixmap.fromImage(self.DN_gray))
        self.roi_title.setText('ROI')
        rows = self.DN.shape[0]
        cols = self.DN.shape[1]

        ## roi oecf变换
        a = -1.018283983130144
        b = 0.014549152785510017
        c = 0.9999931691119089
        DN_red = self.DN[:, :, 2]
        DN_green = self.DN[:, :, 1]
        DN_blue = self.DN[:, :, 0]
        self.phi = np.zeros((rows, cols))
        for row in range(rows):
            for col in range(cols):
                # 公式D.2中的r,g,b权重分别设置为0.25,0.5,0.25
                self.phi[row, col] = 0.25 * (a * np.exp(-b * DN_red[row, col]) + c) + 0.5 * (
                            a * np.exp(-b * DN_green[row, col]) + c) + 0.25 * (a * np.exp(-b * DN_blue[row, col]) + c)
        self.phi_show = np.array(self.phi, dtype=int)
        self.phi_show = QImage(self.phi_show, 60, 60, QImage.Format_Indexed8)
        self.ocef_result.setPixmap(QPixmap.fromImage(self.phi_show))
        self.ocef_result_title.setText('OCEF变换')

        ## roi水平方向一阶微分图
        hor_diff = np.zeros((rows, cols - 1))
        for col in range(cols - 1):
            hor_diff[:, col] = self.PHI[:, col + 1] - self.PHI[:, col]
        ## 计算每行centroids，公式D.3
        cr = []
        for row in range(hor_diff.shape[0]):
            conv = 0
            for col in range(hor_diff.shape[1]):
                conv += (hor_diff[row, col] * col)
            conv = conv / sum(hor_diff[row])
            cr.append(conv - 0.5)
        ## 对centroids进行线性回归拟合，公式D.4
        [m, c_avg] = np.polyfit(np.linspace(0, 59, 60), cr, 1)  # centroid = m * row + c_avg

        self.hor_diff_show = np.array(hor_diff, dtype=int)
        self.hor_diff_show = QImage(self.hor_diff_show, 60, 59, QImage.Format_Indexed8)
        self.roi_diff.setPixmap(QPixmap.fromImage(self.hor_diff_show))
        self.roi_diff_title.setText('水平一阶微分图')

        self.centroids_title.setText('centroids拟合方程：')
        self.centroids_formula.setText('col = ' + str(m) + ' * row + ' + str(c_avg))

        ## ESF，公式D6
        J = np.linspace(0, cols - 0.25, cols * 4)
        x = np.linspace(1, cols * 4, cols * 4)
        A = 0
        B = 0
        esf = []
        for j in range(len(J)):
            for row in range(rows):
                for col in range(cols):
                    sr = (row - rows / 2) * m + c_avg  # 公式D.5(ISO文档中似乎有错误？)
                    if (col - sr - J[j]) < 0.125 and (col - sr - J[j]) >= -0.125:  # 公式D.7
                        alpha = 1
                    else:
                        alpha = 0
                    A += (self.phi[row, col] * alpha)
                    B += alpha
            esf.append(A / B)

        ## LSF
        LSF = []
        X = cols
        for index in range(1, len(esf) - 1):
            Wj = 0.54 + 0.46 * math.cos(2 * math.pi * (index - X * 2) / (X * 4))  # 汉明窗滤波，公式D.9
            lsf = Wj * (esf[index + 1] - esf[index - 1]) / 2  # 公式D.8
            LSF.append(lsf)
        LSF.insert(0, LSF[0])
        LSF.insert(cols * 4 - 1, LSF[cols * 4 - 2])

        def DFT(N, k):
            # 快速傅里叶变换
            dft_m, dft_d = 0, 0
            for j in range(N):
                dft_m += LSF[j] * (math.e ** (-2 * math.pi * k * j / N))
                dft_d += LSF[j]
            dft = dft_m / dft_d
            return dft

        ## sfr，公式D.1
        N = len(LSF)
        SFR = []
        e_SFR = []
        SFR_x = np.linspace(1, N / 2, N / 2)
        for k in range(int(N / 2)):
            Dj = min(1 / math.sin(math.pi * (k + 0.00001) / N), 1)  # +0.00001是为了防止除0，将原公式Dj中的10改为了1
            dft = DFT(N, k)
            esfr = Dj * abs(dft)
            e_SFR.append(esfr)

            f = k / X
            Dj = min(1 / math.sin(math.pi * (f + 0.00001) / N), 1)  # +0.00001是为了防止除0，将原公式Dj中的10改为了1
            dft = DFT(N, f)
            sfr = Dj * abs(dft)
            SFR.append(sfr)

        self.plot = pg.PlotWidget(enableAutoRange=True)
        self.plot_ESF.addWidget(self.plot)
        self.curve = self.plot.plot(x, esf)
        self.plot_ESF_title.setText('ESF曲线')

        self.plot = pg.PlotWidget(enableAutoRange=True)
        self.plot_LSF.addWidget(self.plot)
        self.curve = self.plot.plot(x, LSF)
        self.plot_LSF_title.setText('LSF曲线')

        self.plot = pg.PlotWidget(enableAutoRange=True)
        self.plot_e_SFR.addWidget(self.plot)
        self.curve = self.plot.plot(SFR_x, e_SFR)
        self.plot_e_SFR_title.setText('e-SFR曲线')

        self.plot = pg.PlotWidget(enableAutoRange=True)
        self.plot_SFR.addWidget(self.plot)
        self.curve = self.plot.plot(SFR_x, e_SFR)
        self.plot_SFR_title.setText('SFR曲线')




    def index_f(self, x, a, b, c):
        # 构造指数方程
        return a * np.exp(-b * x) + c

    def get_ocef_lut(self):
        dig = QFileDialog()
        # 设置可以打开任何文件
        dig.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        dig.setFilter(QDir.Files)
        if dig.exec_():
            # 接受选中文件的路径，默认为列表
            filenames = dig.selectedFiles()
            # 列表中的第一个元素即是文件路径，以只读的方式打开文件
            f = open(filenames[0], 'r')
            with f:
                # 接受读取的内容，并显示到多行文本框中
                data = f.read()
                self.ocef_lut.setText(data)
        with open(filenames[0], 'r') as g:
            data = list()
            lines = g.readlines()
            for line in lines:
                oecf = np.zeros((2))
                oecf[0] = line.split('\n')[0].split('\t')[0]
                oecf[1] = line.split('\n')[0].split('\t')[1]
                data.append(oecf)
        array_data = np.array(data)
        xdata = array_data[:, 1]
        ydata = array_data[:, 0]
        [a, b, c], _ = curve_fit(self.index_f, xdata, ydata)
        # self.ocef_formula.setText(str(a))

        self.ocef_formula_title.setText('OCEF拟合方程：')
        self.ocef_formula.setText('[oecf] = ' + str(a)[:7] + ' * exp(-' + str(b)[:7] + ' * [rgb]) + ' + str(c)[:8])

        self.plot = pg.PlotWidget(enableAutoRange=True)
        self.plot_ocef.addWidget(self.plot)
        self.curve = self.plot.plot(xdata, ydata)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())

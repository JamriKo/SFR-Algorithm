import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import exposure
from scipy.optimize import curve_fit


def load_data(file):
    data = list()
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            oecf = np.zeros((2))
            oecf[0] = line.split('\n')[0].split('\t')[0]
            oecf[1] = line.split('\n')[0].split('\t')[1]
            data.append(oecf)
    array_data = np.array(data)
    return array_data

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

## 指数拟合oecf
oecf = load_data('./oecf.txt')
xdata = oecf[:,1]
ydata = oecf[:,0]
popt, pcov = curve_fit(func, xdata, ydata)
# popt数组中，三个值分别是待求参数a,b,c
y2 = [func(i, popt[0], popt[1], popt[2]) for i in xdata]


## 载入图像，获取roi，并灰度化
I = cv2.imread('./1.png')
# cv2.rectangle(I,(685,420),(745,480),(255,0,0),3)
# plt.imshow(I)
# plt.show()
I_gray = cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
roi = np.array(I_gray[420:480,685:745], dtype=float)
rows = roi.shape[0]
cols = roi.shape[1]

## roi逆伽马变换
roi_invgamma = np.zeros((rows, cols))
for row in range(rows):
    for col in range(cols):
        roi_invgamma[row, col] = func(roi[row, col], popt[0], popt[1], popt[2])

## 水平方向一阶微分图
hor_diff = np.zeros((rows, cols-1))
for col in range (cols - 1):
    hor_diff[:,col] = roi[:, col + 1] - roi[:, col]

## 计算每行质心
cr = []
for row in range(hor_diff.shape[0]):
    conv = 0
    for col in range(hor_diff.shape[1]):
        conv += (hor_diff[row, col] * col)
    conv = conv / sum(hor_diff[row])
    cr.append(conv)

## 对质心进行线性回归拟合
[k, b] = np.polyfit(np.linspace(0, 59, 60), cr, 1)

## ESF
J = np.linspace(0, roi.shape[1], roi.shape[1]*4+1)
x = np.linspace(0, roi.shape[1]*4, roi.shape[1]*4+1)
A = 0
B = 0
esf = []
for j in range(len(J)):
    for row in range(rows):
        for col in range(cols):
            sr = (row - rows/2) * k + b
            if (col - sr - J[j]) < 0.125 and (col - sr - J[j]) >= -0.125:
                alpha = 1
            else:
                alpha = 0
            A += (roi_invgamma[row, col] * alpha)
            B += alpha
    esf.append(A / B)

## LSF
lsf = []
for index in range(len(esf)-2):
    lsf.append((esf[index + 2] - esf[index]) / 2)
lsf.insert(0, lsf[0])
lsf.insert(roi.shape[1]*4, lsf[roi.shape[1]*4 - 1])

## 绘制ESF和LSF曲线
plt.plot(x, esf, 'r-')
plt.title('ESF')
plt.show()
plt.plot(x, lsf, 'g-')
plt.title('LSF')
plt.show()
### 参考ISO12233-2014中Annex D部分给出的e-SFR算法编写
### ocef变换的lut(ocef.txt)来源于ISO 14524-OECF

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math


def load_data(file):
    # 载入oecf数据
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

def index_f(x, a, b, c):
    # 构造指数方程
    return a * np.exp(-b * x) + c

## 指数拟合oecf
oecf = load_data('./oecf.txt')
xdata = oecf[:,1]
ydata = oecf[:,0]
[a, b, c], _ = curve_fit(index_f, xdata, ydata)

## 载入图像，获取roi，并灰度化
I = cv2.imread('./test.png')   # 给定棋盘格图片尺寸为：weight:1444, height:1084
DN = np.array(I[420:480,685:745,:], dtype=float) # 截取roi
DN_red = DN[:,:,2]
DN_green = DN[:,:,1]
DN_blue = DN[:,:,0]
PHI = cv2.cvtColor(I[420:480,685:745,:],cv2.COLOR_RGB2GRAY)
rows = DN.shape[0]
cols = DN.shape[1]
cv2.rectangle(I,(685,420),(745,480),(255,0,0),3)
plt.imshow(I)
plt.title('test_img')
plt.show()

## roi oecf变换
phi = np.zeros((rows, cols))
for row in range(rows):
    for col in range(cols):
        # 公式D.2中的r,g,b权重分别设置为0.25,0.5,0.25
        phi[row, col] = 0.25 * index_f(DN_red[row, col], a, b, c) + 0.5 * index_f(DN_green[row, col], a, b, c) + 0.25 * index_f(DN_blue[row, col], a, b, c)

## roi水平方向一阶微分图
hor_diff = np.zeros((rows, cols-1))
for col in range (cols - 1):
    hor_diff[:,col] = PHI[:, col + 1] - PHI[:, col]

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

## ESF，公式D6
J = np.linspace(0, cols - 0.25, cols * 4)
x = np.linspace(1, cols * 4, cols * 4)
A = 0
B = 0
esf = []
for j in range(len(J)):
    for row in range(rows):
        for col in range(cols):
            sr = (row - rows / 2) * m + c_avg   # 公式D.5(ISO文档中似乎有错误？)
            if (col - sr - J[j]) < 0.125 and (col - sr - J[j]) >= -0.125:   #公式D.7
                alpha = 1
            else:
                alpha = 0
            A += (phi[row, col] * alpha)
            B += alpha
    esf.append(A / B)

## LSF
LSF = []
X = cols
for index in range(1, len(esf)-1):
    Wj = 0.54 + 0.46 * math.cos(2 * math.pi * (index - X * 2) / (X * 4))  # 汉明窗滤波，公式D.9
    lsf = Wj * (esf[index + 1] - esf[index - 1]) / 2    # 公式D.8
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
for k in range(int(N/2)):
    Dj = min(1 / math.sin(math.pi * (k + 0.00001) / N), 1) # +0.00001是为了防止除0，将原公式Dj中的10改为了1
    dft = DFT(N, k)
    esfr = Dj * abs(dft)
    e_SFR.append(esfr)

    f = k / X
    Dj = min(1 / math.sin(math.pi * (f + 0.00001) / N), 1)  # +0.00001是为了防止除0，将原公式Dj中的10改为了1
    dft = DFT(N, f)
    sfr = Dj * abs(dft)
    SFR.append(sfr)

## 绘制ESF和LSF曲线
plt.plot(x, esf, 'r-')
plt.title('ESF')
plt.show()
plt.plot(x, LSF, 'g-')
plt.title('LSF')
plt.show()
plt.plot(SFR_x, e_SFR, 'b-')
plt.title('e-SFR')
plt.show()
plt.plot(SFR_x, SFR, 'r-')
plt.title('SFR')
plt.show()
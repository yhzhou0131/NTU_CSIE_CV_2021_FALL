import cv2
import numpy as np
import math
import random
import os

path = "./lena.bmp"

kernel_E = [
             [-2, -1], [-2, 0], [-2, 1],
    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
    [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
    [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
             [2, -1], [2, 0], [2, 1]
]

def GetGaussianNoise(img, threshold):
    return img + threshold * np.random.normal(0, 1, img.shape)

def GetSaltAndPepper(img, threshold):
    res = np.copy(img)
    randomValue = np.random.uniform(0, 1, res.shape)
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if randomValue[i][j] < threshold:
                res[i][j] = 0
            elif randomValue[i][j] > 1 - threshold:
                res[i][j] = 255
            else:
                continue
    return res

def boxFilter(img, size):
    kernel = []
    for i in range(-size // 2, size // 2):
        for j in range(-size // 2, size // 2):
            kernel.append([i, j])
    normalize = size * size
    row, col = img.shape
    res = np.zeros(img.shape)

    for i in range(row):
        for j in range(col):
            sum = 0
            for ele in kernel:
                eleI, eleJ = ele
                if i + eleI >= 0 and i + eleI < row and j + eleJ >= 0 and j + eleJ < col:
                    sum += img[i + eleI, j + eleJ]
            res[i][j] = sum / normalize
    return res

def medianFilter(img, size):
    kernel = []
    for i in range(-size // 2, size // 2):
        for j in range(-size // 2, size // 2):
            kernel.append([i, j])
    
    res = np.zeros(img.shape)
    row, col = img.shape

    for i in range(row):
        for j in range(col):
            medianSet = []
            for k in kernel:
                ki, kj = k
                if i + ki >= 0 and i + ki < row and j + kj >= 0 and j + kj < col:
                    medianSet.append(img[i+ki][j+kj])
            res[i][j] = np.median(medianSet)
    return res

def SNR(img, noiseImg):
    img = img / 255
    noiseImg = noiseImg / 255

    if img.shape != noiseImg.shape:
        print("Image size must be same.")
        return
    
    us = 0
    VS = 0
    uNoise = 0
    VN = 0
    row, col = img.shape

    for i in range(row):
        for j in range(col):
            us = us + img[i][j]
    us = us / (row * col)

    for i in range(row):
        for j in range(col):
            VS = VS + math.pow(img[i][j] - us, 2)
    VS = VS / (row * col)

    for i in range(row):
        for j in range(col):
            uNoise = uNoise + (noiseImg[i][j] - img[i][j])
    uNoise = uNoise / (row * col)

    for i in range(row):
        for j in range(col):
            VN = VN + math.pow(noiseImg[i][j] - img[i][j] - uNoise, 2)
    VN = VN / (row * col)

    return 20 * math.log(math.sqrt(VS) / math.sqrt(VN), 10)

def dilation(a, b):
    row_a, col_a = a.shape
    img = np.zeros(a.shape, dtype='int32')

    for a_i in range(row_a):
        for a_j in range(col_a):
            max_value = 0
            for lis in kernel_E:
                b_i, b_j = lis
                if (a_i + b_i) < row_a and (a_i + b_i) >= 0 and \
                    (a_j + b_j) < col_a and (a_j + b_j) >= 0:
                    max_value = max(max_value, a[a_i+b_i, a_j+b_j]) 
            img[a_i, a_j] = max_value
    return img

def erosion(a, b):
    row_a, col_a = a.shape
    img = np.zeros(a.shape, dtype='int32')

    for a_i in range(row_a):
        for a_j in range(col_a):
            min_value = 255
            
            for lis in b:
                b_i, b_j = lis
                if (a_i + b_i) < row_a and (a_i + b_i) >= 0 and \
                    (a_j + b_j) < col_a and (a_j + b_j) >= 0:
                    min_value = min(min_value, a[a_i+b_i, a_j+b_j])
            img[a_i, a_j] = min_value
    return img   

def opening(a, b):
    return dilation(erosion(a, b), b)

def closing(a, b):
    return erosion(dilation(a, b), b)

if __name__ == "__main__":
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if not os.path.exists("partA"):
        os.mkdir("partA")
    if not os.path.exists("partB"):
        os.mkdir("partB")
    if not os.path.exists("partC"):
        os.mkdir("partC")
    if not os.path.exists("partD"):
        os.mkdir("partD")
    if not os.path.exists("partE"):
        os.mkdir("partE")
    
    # a
    gaussianImage_10 = GetGaussianNoise(img, 10)
    cv2.imwrite("./partA/gaussianImage_10.bmp", gaussianImage_10)

    gaussianImage_30 = GetGaussianNoise(img, 30)
    cv2.imwrite("./partA/gaussianImage_30.bmp", gaussianImage_30)

    print("gaussianImage_10_SNR = {}".format(SNR(img, gaussianImage_10)))
    print("gaussianImage_30_SNR = {}".format(SNR(img, gaussianImage_30)))

    # b
    saltAndPepper_005 = GetSaltAndPepper(img, 0.05)
    cv2.imwrite("./partB/saltAndPepper_005.bmp", saltAndPepper_005)

    saltAndPepper_01 = GetSaltAndPepper(img, 0.1)
    cv2.imwrite("./partB/saltAndPepper_01.bmp", saltAndPepper_01)

    print("saltAndPepper_0.05_SNR = {}".format(SNR(img, saltAndPepper_005)))
    print("saltAndPepper_0.1_SNR = {}".format(SNR(img, saltAndPepper_01)))

    # c
    gaussianImage_10_BF3 = boxFilter(gaussianImage_10, 3)
    cv2.imwrite("./partC/gaussianImage_10_BF3.bmp", gaussianImage_10_BF3)
    gaussianImage_10_BF5 = boxFilter(gaussianImage_10, 5)
    cv2.imwrite("./partC/gaussianImage_10_BF5.bmp", gaussianImage_10_BF5)

    gaussianImage_30_BF3 = boxFilter(gaussianImage_30, 3)
    cv2.imwrite("./partC/gaussianImage_30_BF3.bmp", gaussianImage_30_BF3)
    gaussianImage_30_BF5 = boxFilter(gaussianImage_30, 5)
    cv2.imwrite("./partC/gaussianImage_30_BF5.bmp", gaussianImage_30_BF5)

    print("gaussianImage_10_BF3_SNR = {}".format(SNR(img, gaussianImage_10_BF3)))
    print("gaussianImage_10_BF5_SNR = {}".format(SNR(img, gaussianImage_10_BF5)))
    print("gaussianImage_30_BF3_SNR = {}".format(SNR(img, gaussianImage_30_BF3)))
    print("gaussianImage_30_BF5_SNR = {}".format(SNR(img, gaussianImage_30_BF5)))

    saltAndPepper_005_BF3 = boxFilter(saltAndPepper_005, 3)
    cv2.imwrite("./partC/saltAndPepper_005_BF3.bmp", saltAndPepper_005_BF3)
    saltAndPepper_005_BF5 = boxFilter(saltAndPepper_005, 5)
    cv2.imwrite("./partC/saltAndPepper_005_BF5.bmp", saltAndPepper_005_BF5)

    saltAndPepper_01_BF3 = boxFilter(saltAndPepper_01, 3)
    cv2.imwrite("./partC/saltAndPepper_01_BF3.bmp", saltAndPepper_01_BF3)
    saltAndPepper_01_BF5 = boxFilter(saltAndPepper_01, 5)
    cv2.imwrite("./partC/saltAndPepper_01_BF5.bmp", saltAndPepper_01_BF5)

    print("saltAndPepper_0.05_BF3_SNR = {}".format(SNR(img, saltAndPepper_005_BF3)))
    print("saltAndPepper_0.05_BF5_SNR = {}".format(SNR(img, saltAndPepper_005_BF5)))
    print("saltAndPepper_0.1_BF3_SNR = {}".format(SNR(img, saltAndPepper_01_BF3)))
    print("saltAndPepper_0.1_BF5_SNR = {}".format(SNR(img, saltAndPepper_01_BF5)))

    # d
    gaussianImage_10_MF3 = medianFilter(gaussianImage_10, 3)
    cv2.imwrite("./partD/gaussianImage_10_MF3.bmp", gaussianImage_10_MF3)
    gaussianImage_10_MF5 = medianFilter(gaussianImage_10, 5)
    cv2.imwrite("./partD/gaussianImage_10_MF5.bmp", gaussianImage_10_MF5)
    gaussianImage_30_MF3 = medianFilter(gaussianImage_30, 3)
    cv2.imwrite("./partD/gaussianImage_30_MF3.bmp", gaussianImage_30_MF3)
    gaussianImage_30_MF5 = medianFilter(gaussianImage_30, 5)
    cv2.imwrite("./partD/gaussianImage_30_MF5.bmp", gaussianImage_30_MF5)

    print("gaussianImage_10_MF3_SNR = {}".format(SNR(img, gaussianImage_10_MF3)))
    print("gaussianImage_10_MF5_SNR = {}".format(SNR(img, gaussianImage_10_MF5)))
    print("gaussianImage_30_MF3_SNR = {}".format(SNR(img, gaussianImage_30_MF3)))
    print("gaussianImage_30_MF5_SNR = {}".format(SNR(img, gaussianImage_30_MF5)))

    saltAndPepper_005_MF3 = medianFilter(saltAndPepper_005, 3)
    cv2.imwrite("./partD/saltAndPepper_005_MF3.bmp", saltAndPepper_005_MF3)
    saltAndPepper_005_MF5 = medianFilter(saltAndPepper_005, 5)
    cv2.imwrite("./partD/saltAndPepper_005_MF5.bmp", saltAndPepper_005_MF5)
    saltAndPepper_01_MF3 = medianFilter(saltAndPepper_01, 3)
    cv2.imwrite("./partD/saltAndPepper_01_MF3.bmp", saltAndPepper_01_MF3)
    saltAndPepper_01_MF5 = medianFilter(saltAndPepper_01, 5)
    cv2.imwrite("./partD/saltAndPepper_01_MF5.bmp", saltAndPepper_01_MF5)

    print("saltAndPepper_0.05_MF3_SNR = {}".format(SNR(img, saltAndPepper_005_MF3)))
    print("saltAndPepper_0.05_MF5_SNR = {}".format(SNR(img, saltAndPepper_005_MF5)))
    print("saltAndPepper_0.1_MF3_SNR = {}".format(SNR(img, saltAndPepper_01_MF3)))
    print("saltAndPepper_0.1_MF5_SNR = {}".format(SNR(img, saltAndPepper_01_MF5)))

    # e
    gaussianImage_10_OC = closing(opening(gaussianImage_10, kernel_E), kernel_E)
    cv2.imwrite("./partE/gaussianImage_10_OC.bmp", gaussianImage_10_OC)
    gaussianImage_30_OC = closing(opening(gaussianImage_30, kernel_E), kernel_E)
    cv2.imwrite("./partE/gaussianImage_30_OC.bmp", gaussianImage_30_OC)
    gaussianImage_10_CO = opening(closing(gaussianImage_10, kernel_E), kernel_E)
    cv2.imwrite("./partE/gaussianImage_10_CO.bmp", gaussianImage_10_CO)
    gaussianImage_30_CO = opening(closing(gaussianImage_30, kernel_E), kernel_E)
    cv2.imwrite("./partE/gaussianImage_30_CO.bmp", gaussianImage_30_CO)

    print("gaussianImage_10_OC_SNR = {}".format(SNR(img, gaussianImage_10_OC)))
    print("gaussianImage_30_OC_SNR = {}".format(SNR(img, gaussianImage_30_OC)))
    print("gaussianImage_10_CO_SNR = {}".format(SNR(img, gaussianImage_10_CO)))
    print("gaussianImage_30_CO_SNR = {}".format(SNR(img, gaussianImage_30_CO)))

    saltAndPepper_005_OC = closing(opening(saltAndPepper_005, kernel_E), kernel_E)
    cv2.imwrite("./partE/saltAndPepper_005_OC.bmp", saltAndPepper_005_OC)
    saltAndPepper_01_OC = closing(opening(saltAndPepper_01, kernel_E), kernel_E)
    cv2.imwrite("./partE/saltAndPepper_01_OC.bmp", saltAndPepper_01_OC)
    saltAndPepper_005_CO = opening(closing(saltAndPepper_005, kernel_E), kernel_E)
    cv2.imwrite("./partE/saltAndPepper_005_CO.bmp", saltAndPepper_005_CO)
    saltAndPepper_01_CO = opening(closing(saltAndPepper_01, kernel_E), kernel_E)
    cv2.imwrite("./partE/saltAndPepper_01_CO.bmp", saltAndPepper_01_CO)

    print("saltAndPepper_0.05_OC_SNR = {}".format(SNR(img, saltAndPepper_005_OC)))
    print("saltAndPepper_0.1_OC_SNR = {}".format(SNR(img, saltAndPepper_01_OC)))
    print("saltAndPepper_0.05_CO_SNR = {}".format(SNR(img, saltAndPepper_005_CO)))
    print("saltAndPepper_0.1_CO_SNR = {}".format(SNR(img, saltAndPepper_01_CO)))
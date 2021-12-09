import cv2
import numpy as np
import math

from numpy.lib.function_base import gradient

path = "./lena.bmp"

def padding(img):
    row, col = img.shape
    res_img = np.zeros((row+2, col+2))

    res_row, res_col = res_img.shape

    for ri in range(res_row):
        for rj in range(res_col):
            # adapt row
            i = 0; j = 0
            if ri != 0 and ri != row + 1:
                i = ri - 1
            if ri == row + 1:
                i = row - 1
            # adapt col
            if rj != 0 and rj != col + 1:
                j = rj - 1
            if rj == col + 1:
                j = col - 1

            res_img[ri][rj] = img[i][j]
    return res_img

def Robert(img, threshold):
    row, col = img.shape
    res_img = np.zeros((row - 2, col - 2))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            r1 = img[i+1][j+1] - img[i][j]
            r2 = img[i+1][j] - img[i][j+1]
            gradient_magnitude = math.sqrt(r1 ** 2 + r2 ** 2)

            if gradient_magnitude < threshold:
                res_img[i-1][j-1] = 255
    return res_img

def Perwitts(img, threshold):
    row, col = img.shape
    res_img = np.zeros((row - 2, col - 2))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            r1 = (img[i+1][j+1] + img[i+1][j] + img[i+1][j-1]) - (img[i-1][j+1] + img[i-1][j] + img[i-1][j-1])
            r2 = (img[i+1][j+1] + img[i][j+1] + img[i-1][j+1]) - (img[i+1][j-1] + img[i][j-1] + img[i-1][j-1])
            gradient_magnitude = math.sqrt(r1 ** 2 + r2 ** 2)

            if gradient_magnitude < threshold:
                res_img[i-1][j-1] = 255
    return res_img

def Sobel(img, threshold):
    row, col = img.shape
    res_img = np.zeros((row - 2, col - 2))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            r1 = (img[i+1][j+1] + 2 * img[i+1][j] + img[i+1][j-1]) - (img[i-1][j+1] + 2 * img[i-1][j] + img[i-1][j-1])
            r2 = (img[i+1][j+1] + 2 * img[i][j+1] + img[i-1][j+1]) - (img[i+1][j-1] + 2 * img[i][j-1] + img[i-1][j-1])
            gradient_magnitude = math.sqrt(r1 ** 2 + r2 ** 2)

            if gradient_magnitude < threshold:
                res_img[i-1][j-1] = 255
    return res_img

def Frei_and_Chen(img, threshold):
    row, col = img.shape
    res_img = np.zeros((row - 2, col - 2))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            r1 = (img[i+1][j+1] + math.sqrt(2) * img[i+1][j] + img[i+1][j-1]) - (img[i-1][j+1] + math.sqrt(2) * img[i-1][j] + img[i-1][j-1])
            r2 = (img[i+1][j+1] + math.sqrt(2) * img[i][j+1] + img[i-1][j+1]) - (img[i+1][j-1] + math.sqrt(2) * img[i][j-1] + img[i-1][j-1])
            gradient_magnitude = math.sqrt(r1 ** 2 + r2 ** 2)

            if gradient_magnitude < threshold:
                res_img[i-1][j-1] = 255
    return res_img

def Kirsch(img, threshold):
    row, col = img.shape
    res_img = np.zeros((row - 2, col - 2))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            k0 = -3 * (img[i-1][j] + img[i-1][j-1] + img[i][j-1] + img[i+1][j-1] + img[i+1][j]) + 5 * (img[i-1][j+1] + img[i][j+1] + img[i+1][j+1])
            k1 = -3 * (img[i+1][j+1] + img[i-1][j-1] + img[i][j-1] + img[i+1][j-1] + img[i+1][j]) + 5 * (img[i-1][j+1] + img[i][j+1] + img[i-1][j])
            k2 = -3 * (img[i+1][j+1] + img[i][j+1] + img[i][j-1] + img[i+1][j-1] + img[i+1][j]) + 5 * (img[i-1][j+1] + img[i-1][j-1] + img[i-1][j])
            k3 = -3 * (img[i+1][j+1] + img[i][j+1] + img[i-1][j+1] + img[i+1][j-1] + img[i+1][j]) + 5 * (img[i][j-1] + img[i-1][j-1] + img[i-1][j])
            k4 = -3 * (img[i+1][j+1] + img[i][j+1] + img[i-1][j+1] + img[i-1][j] + img[i+1][j]) + 5 * (img[i][j-1] + img[i-1][j-1] + img[i+1][j-1])
            k5 = -3 * (img[i+1][j+1] + img[i][j+1] + img[i-1][j+1] + img[i-1][j] + img[i-1][j-1]) + 5 * (img[i][j-1] + img[i+1][j] + img[i+1][j-1])
            k6 = -3 * (img[i][j-1] + img[i][j+1] + img[i-1][j+1] + img[i-1][j] + img[i-1][j-1]) + 5 * (img[i+1][j+1] + img[i+1][j] + img[i+1][j-1])
            k7 = -3 * (img[i][j-1] + img[i+1][j-1] + img[i-1][j+1] + img[i-1][j] + img[i-1][j-1]) + 5 * (img[i+1][j+1] + img[i+1][j] + img[i][j+1])
            arr = [k0, k1, k2, k3, k4, k5, k6, k7]
            gradient_magnitude = max(arr)

            if gradient_magnitude < threshold:
                res_img[i-1][j-1] = 255
    return res_img

def Robinson(img, threshold):
    row, col = img.shape
    res_img = np.zeros((row - 2, col - 2))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            r0 = img[i-1][j+1] + 2*img[i][j+1] + img[i+1][j+1] - (img[i-1][j-1] + 2*img[i][j-1] + img[i+1][j-1])
            r1 = img[i-1][j] + 2*img[i-1][j+1] + img[i][j+1] - (img[i][j-1] + 2*img[i+1][j-1] + img[i+1][j])
            r2 = img[i-1][j-1] + 2*img[i-1][j] + img[i-1][j+1] - (img[i+1][j-1] + 2*img[i+1][j] + img[i+1][j+1])
            r3 = img[i-1][j] + 2*img[i-1][j-1] + img[i][j-1] - (img[i+1][j] + 2*img[i+1][j+1] + img[i][j+1])
            r4 = img[i-1][j-1] + 2*img[i][j-1] + img[i+1][j-1] - (img[i-1][j+1] + 2*img[i][j+1] + img[i+1][j+1])
            r5 = img[i][j-1] + 2*img[i+1][j-1] + img[i+1][j] - (img[i][j+1] + 2*img[i-1][j+1] + img[i-1][j])
            r6 = img[i+1][j-1] + img[i+1][j] + img[i+1][j+1] - (img[i-1][j-1] + 2*img[i-1][j] + img[i-1][j+1])
            r7 = img[i-1][j] + 2*img[i-1][j-1] + img[i][j-1] - (img[i+1][j] + 2*img[i+1][j+1] + img[i][j+1])
            arr = [r0, r1, r2, r3, r4, r5, r6, r7]
            gradient_magnitude = max(arr)

            if gradient_magnitude < threshold:
                res_img[i-1][j-1] = 255
    return res_img

def padding_5x5(img):
    row, col = img.shape
    res_img = np.zeros((row+4, col+4))

    res_row, res_col = res_img.shape

    for ri in range(res_row):
        for rj in range(res_col):
            # adapt row
            i = 0; j = 0
            if ri > 1 and ri < row + 2:
                i = ri - 2
            if ri >= row + 2:
                i = row - 1
            # adapt col
            if rj > 1 and rj < col + 2:
                j = rj - 2
            if rj >= col + 2:
                j = col - 1

            res_img[ri][rj] = img[i][j]
    return res_img

def Nevatia_Babu(img, threshold):
    row, col = img.shape
    res_img = np.zeros((row-4, col-4))

    for i in range(2, row-2):
        for j in range(2, col-2):
            N0 = 100 * (img[i-2][j-2] + img[i-2][j-1] + img[i-2][j] + img[i-2][j+1] + img[i-2][j+2] + \
                        img[i-1][j-2] + img[i-1][j-1] + img[i-1][j] + img[i-1][j+1] + img[i-1][j+2]) - \
                 100 * (img[i+1][j-2] + img[i+1][j-1] + img[i+1][j] + img[i+1][j+1] + img[i+1][j+2] + \
                        img[i+2][j-2] + img[i+2][j-1] + img[i+2][j] + img[i+2][j+1] + img[i+2][j+2])
            N1 = 100 * (img[i-2][j-2] + img[i-2][j-1] + img[i-2][j] + img[i-2][j+1] + img[i-2][j+2] + \
                        img[i-1][j-2] + img[i-1][j-1] + img[i-1][j] + img[i][j-2]) + 78*img[i-1][j+1] + 92*img[i][j-1] + 32*img[i+1][j-2] - \
                 100 * (img[i+1][j] + img[i+1][j+1] + img[i+1][j+2] + img[i][j+2] + \
                        img[i+2][j-2] + img[i+2][j-1] + img[i+2][j] + img[i+2][j+1] + img[i+2][j+2]) - 32*img[i-1][j+2] - 92*img[i][j+1] - 78*img[i+1][j-1]
            N2 = 100 * (img[i-2][j-2] + img[i-1][j-2] + img[i][j-2] + img[i+1][j-2] + img[i+2][j-2] + \
                        img[i-2][j-1] + img[i-1][j-1] + img[i][j-1] + img[i-2][j]) + 32*img[i-2][j+1] + 92*img[i-1][j] + 78*img[i+1][j-1] - \
                 100 * (img[i-2][j+2] + img[i-1][j+2] + img[i][j+2] + img[i+1][j+2] + img[i+2][j+2] + \
                        img[i][j+1] + img[i+1][j+1] + img[i+2][j+1] + img[i+2][j]) - 78*img[i-1][j+1] - 92*img[i+1][j] - 32*img[i+2][j-1]
            N3 = 100 * (img[i-2][j+1] + img[i-1][j+1] + img[i][j+1] + img[i+1][j+1] + img[i+2][j+1] + \
                        img[i-2][j+2] + img[i-1][j+2] + img[i][j+2] + img[i+1][j+2] + img[i+2][j+2]) - \
                 100 * (img[i-2][j-1] + img[i-1][j-1] + img[i][j-1] + img[i+1][j-1] + img[i+2][j-1] + \
                        img[i-2][j-2] + img[i-1][j-2] + img[i][j-2] + img[i+1][j-2] + img[i+2][j-2])
            N4 = 100 * (img[i-2][j+2] + img[i-1][j+2] + img[i][j+2] + img[i+1][j+2] + img[i+2][j+2] + \
                        img[i-2][j+1] + img[i-1][j+1] + img[i][j+1] + img[i-2][j]) + 32*img[i-2][j-1] + 92*img[i-1][j] + 78*img[i+1][j+1] - \
                 100 * (img[i-2][j-2] + img[i-1][j-2] + img[i][j-2] + img[i+1][j-2] + img[i+2][j-2] + \
                        img[i][j-1] + img[i+1][j-1] + img[i+2][j-1] + img[i+2][j]) - 78*img[i-1][j-1] - 92*img[i+1][j] - 32*img[i+2][j+1]
            N5 = 100 * (img[i-2][j-2] + img[i-2][j-1] + img[i-2][j] + img[i-2][j+1] + img[i-2][j+2] + \
                        img[i-1][j] + img[i-1][j+1] + img[i-1][j+2] + img[i][j+2]) + 78*img[i-1][j-1] + 92*img[i][j+1] + 32*img[i+1][j+2] - \
                 100 * (img[i+2][j-2] + img[i+2][j-1] + img[i+2][j] + img[i+2][j+1] + img[i+2][j+2] + \
                        img[i+1][j-2] + img[i+1][j-1] + img[i+1][j] + img[i][j-2]) - 32*img[i-1][j-2] - 92*img[i][j-1] - 78*img[i+1][j+1]
            arr = [N0, N1, N2, N3, N4, N5]
            gradient_magnitude = max(arr)

            if gradient_magnitude < threshold:
                res_img[i-2][j-2] = 255
    return res_img

if __name__ == "__main__":
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    pad_img = padding(img)

    # a
    Robert_img = Robert(pad_img, 12)
    print("finish robert.")
    cv2.imwrite("Robert_img.bmp", Robert_img)

    # b
    Perwitts_img = Perwitts(pad_img, 24)
    print("finish perwitts.")
    cv2.imwrite("Perwitts_img.bmp", Perwitts_img)

    # c
    Sobel_img = Sobel(pad_img, 38)
    print("finish sobel.")
    cv2.imwrite("Sobel_img.bmp", Sobel_img)

    # d
    Frei_and_Chen_img = Frei_and_Chen(pad_img, 30)
    print("finish frei&chen.")
    cv2.imwrite("Frei_and_Chen_img.bmp", Frei_and_Chen_img)

    # e
    Kirsch_img = Kirsch(pad_img, 135)
    print("finish kirsch.")
    cv2.imwrite("Kirsch_img.bmp", Kirsch_img)

    # f
    Robinson_img = Robinson(pad_img, 43)
    print("finish robinson.")
    cv2.imwrite("Robinson_img.bmp", Robinson_img)

    # g
    padding_5x5_img = padding_5x5(img)
    Nevatia_Babu_img = Nevatia_Babu(padding_5x5_img, 12500)
    print("finish nevatiaBabu.")
    cv2.imwrite("Nevatia_Babu_img.bmp", Nevatia_Babu_img)
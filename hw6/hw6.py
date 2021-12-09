import cv2
import numpy as np

path = "./lena.bmp"

def binarize(img):
    return (img > 0x7f) * 0xff

def h(b, c, d, e):
    if(b == c and b == d and b == e):
        return 'r'
    elif(b == c and (b != d or b != e)):
        return 'q'
    return 's'

def showText(img):
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i][j] != 0:
                print(img[i][j], end='')
            else:
                print(' ', end='')
        print()

if __name__ == '__main__':
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bin_img = binarize(img)

    DS_lena = np.zeros((64, 64), dtype='int32')
    row, col = DS_lena.shape

    for i in range(row):
        for j in range(col):
            DS_lena[i][j] = bin_img[8*i][8*j]
    
    yokoi_lena = np.zeros((64, 64), dtype='int32')
    for i in range(row):
        for j in range(col):
            if DS_lena[i][j] != 0:
                if i == 0:
                    if j == 0:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, DS_lena[i][j], DS_lena[i][j+1]
                        x8, x4, x5 = 0, DS_lena[i+1][j], DS_lena[i+1][j+1]
                    elif j == col - 1:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = DS_lena[i][j-1], DS_lena[i][j], 0
                        x8, x4, x5 = DS_lena[i+1][j-1], DS_lena[i+1][j], 0
                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = DS_lena[i][j-1], DS_lena[i][j], DS_lena[i][j+1]
                        x8, x4, x5 = DS_lena[i+1][j-1], DS_lena[i+1][j], DS_lena[i+1][j+1]
                elif i == row - 1:
                    if j == 0:
                        x7, x2, x6 = 0, DS_lena[i-1][j], DS_lena[i-1][j+1]
                        x3, x0, x1 = 0, DS_lena[i][j], DS_lena[i][j+1]
                        x8, x4, x5 = 0, 0, 0
                    elif j == col - 1:
                        x7, x2, x6 = DS_lena[i-1][j-1], DS_lena[i-1][j], 0
                        x3, x0, x1 = DS_lena[i][j-1], DS_lena[i][j], 0
                        x8, x4, x5 = 0, 0, 0
                    else:
                        x7, x2, x6 = DS_lena[i-1][j-1], DS_lena[i-1][j], DS_lena[i-1][j+1]
                        x3, x0, x1 = DS_lena[i][j-1], DS_lena[i][j], DS_lena[i][j+1]
                        x8, x4, x5 = 0, 0, 0
                else:
                    if j == 0:
                        x7, x2, x6 = 0, DS_lena[i-1][j], DS_lena[i-1][j+1]
                        x3, x0, x1 = 0, DS_lena[i][j], DS_lena[i][j+1]
                        x8, x4, x5 = 0, DS_lena[i+1][j], DS_lena[i+1][j+1]
                    elif j == col - 1:
                        x7, x2, x6 = DS_lena[i-1][j-1], DS_lena[i-1][j], 0
                        x3, x0, x1 = DS_lena[i][j-1], DS_lena[i][j], 0
                        x8, x4, x5 = DS_lena[i+1][j-1], DS_lena[i+1][j], 0
                    else:
                        x7, x2, x6 = DS_lena[i-1][j-1], DS_lena[i-1][j], DS_lena[i-1][j+1]
                        x3, x0, x1 = DS_lena[i][j-1], DS_lena[i][j], DS_lena[i][j+1]
                        x8, x4, x5 = DS_lena[i+1][j-1], DS_lena[i+1][j], DS_lena[i+1][j+1]

                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)
                aList = [a1, a2, a3, a4]
                
                if aList.count('r') == 4:
                    yokoi_lena[i][j] = 5
                else:
                    yokoi_lena[i][j] = aList.count('q')
            else:
                yokoi_lena[i][j] = 0
    showText(yokoi_lena)
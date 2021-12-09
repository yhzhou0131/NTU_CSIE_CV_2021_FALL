import cv2
import numpy as np
import copy

path = "./lena.bmp"

def binarize(img):
    return (img > 0x7f) * 0xff

def down_sample(img):
    res_img = np.zeros((64, 64), dtype='int32')
    row, col = res_img.shape
    for i in range(row):
        for j in range(col):
            res_img[i][j] = img[8*i][8*j]
    return res_img

    def h(c, d):
        if c == d:
            return c
        return 'b'

    res_img = np.zeros(img.shape, dtype='int32')
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            x1, x2, x3, x4 = 0, 0, 0, 0
            if i == 0:
                if j == 0:
                    x1, x4 = img[i][j+1], img[i+1][j]
                elif j == col - 1:
                    x3, x4 = img[i][j-1], img[i+1][j]
                else:
                    x1, x3, x4 = img[i][j+1], img[i][j-1], img[i+1][j]
            elif i == row - 1:
                if j == 0:
                    x1, x2 = img[i][j+1], img[i-1][j]
                elif j == col - 1:
                    x2, x3 = img[i-1][j], img[i][j-1]
                else:
                    x1, x2, x3 = img[i][j+1], img[i-1][j], img[i][j-1]
            else:
                if j == 0:
                    x1, x2, x4 = img[i][j+1], img[i-1][j], img[i+1][j]
                elif j == col - 1:
                    x2, x3, x4 = img[i-1][j], img[i][j-1], img[i+1][j]
                else:
                    x1, x2, x3, x4 = img[i][j+1], img[i-1][j], img[i][j-1], img[i+1][j]
            x1 /= 255
            x2 /= 255
            x3 /= 255
            x4 /= 255
            a1 = h(1, x1)
            a2 = h(a1, x2)
            a3 = h(a2, x3)
            a4 = h(a3, x4)
            if a4 == 'b':
                res_img[i][j] = 2
            else:
                res_img[i][j] = 1
    return res_img

def pair_relationship(img): # 3: p, 4: q
    yokoi_img = yokoi(img)
    def h(a, m):
        if a == m:
            return 1
        else:
            return 0
    res_img = np.zeros(img.shape, dtype='int32')
    row, col = res_img.shape
    for i in range(row):
        for j in range(col):
            x1, x2, x3, x4 = 0, 0, 0, 0
            if i == 0:
                if j == 0:
                    x1, x4 = yokoi_img[i][j+1], yokoi_img[i+1][j]
                elif j == col - 1:
                    x3, x4 = yokoi_img[i][j-1], yokoi_img[i+1][j]
                else:
                    x1, x3, x4 = yokoi_img[i][j+1], yokoi_img[i][j-1], yokoi_img[i+1][j]
            elif i == row - 1:
                if j == 0:
                    x1, x2 = yokoi_img[i][j+1], yokoi_img[i-1][j]
                elif j == col - 1:
                    x2, x3 = yokoi_img[i-1][j], yokoi_img[i][j-1]
                else:
                    x1, x2, x3 = yokoi_img[i][j+1], yokoi_img[i-1][j], yokoi_img[i][j-1]
            else:
                if j == 0:
                    x1, x2, x4 = yokoi_img[i][j+1], yokoi_img[i-1][j], yokoi_img[i+1][j]
                elif j == col - 1:
                    x2, x3, x4 = yokoi_img[i-1][j], yokoi_img[i][j-1], yokoi_img[i+1][j]
                else:
                    x1, x2, x3, x4 = yokoi_img[i][j+1], yokoi_img[i-1][j], yokoi_img[i][j-1], yokoi_img[i+1][j]
            
            if h(x1, 1) + h(x2, 1) + h(x3, 1) + h(x4, 1) >= 1 and yokoi_img[i][j] == 1:
                res_img[i, j] = 3
            else:
                res_img[i, j] = 4
    return res_img

def yokoi(img):
    def h(b, c, d, e):
        if(b == c and b == d and b == e):
            return 'r'
        elif(b == c and (b != d or b != e)):
            return 'q'
        return 's'
    
    yokoi_lena = np.zeros((64, 64), dtype='int32')
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i][j] != 0:
                if i == 0:
                    if j == 0:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, img[i][j], img[i][j+1]
                        x8, x4, x5 = 0, img[i+1][j], img[i+1][j+1]
                    elif j == col - 1:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img[i][j-1], img[i][j], 0
                        x8, x4, x5 = img[i+1][j-1], img[i+1][j], 0
                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img[i][j-1], img[i][j], img[i][j+1]
                        x8, x4, x5 = img[i+1][j-1], img[i+1][j], img[i+1][j+1]
                elif i == row - 1:
                    if j == 0:
                        x7, x2, x6 = 0, img[i-1][j], img[i-1][j+1]
                        x3, x0, x1 = 0, img[i][j], img[i][j+1]
                        x8, x4, x5 = 0, 0, 0
                    elif j == col - 1:
                        x7, x2, x6 = img[i-1][j-1], img[i-1][j], 0
                        x3, x0, x1 = img[i][j-1], img[i][j], 0
                        x8, x4, x5 = 0, 0, 0
                    else:
                        x7, x2, x6 = img[i-1][j-1], img[i-1][j], img[i-1][j+1]
                        x3, x0, x1 = img[i][j-1], img[i][j], img[i][j+1]
                        x8, x4, x5 = 0, 0, 0
                else:
                    if j == 0:
                        x7, x2, x6 = 0, img[i-1][j], img[i-1][j+1]
                        x3, x0, x1 = 0, img[i][j], img[i][j+1]
                        x8, x4, x5 = 0, img[i+1][j], img[i+1][j+1]
                    elif j == col - 1:
                        x7, x2, x6 = img[i-1][j-1], img[i-1][j], 0
                        x3, x0, x1 = img[i][j-1], img[i][j], 0
                        x8, x4, x5 = img[i+1][j-1], img[i+1][j], 0
                    else:
                        x7, x2, x6 = img[i-1][j-1], img[i-1][j], img[i-1][j+1]
                        x3, x0, x1 = img[i][j-1], img[i][j], img[i][j+1]
                        x8, x4, x5 = img[i+1][j-1], img[i+1][j], img[i+1][j+1]

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
    return yokoi_lena

def equal_img(img1, img2):
    row1, col1 = img1.shape
    row2, col2 = img2.shape

    if row1 != row2 or col1 != col2:
        return False
    else:
        flag = True
        for i in range(row1):
            for j in range(col1):
                if(img1[i][j] != img2[i][j]):
                    flag = False
                    break
            if(not flag):
                break
        return flag

if __name__ == "__main__":
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bin_img = binarize(img)
    down_img = down_sample(bin_img)
    
    thin_lena = np.copy(down_img)
    old_thin = np.copy(down_img)

    def h(b, c, d, e):
        if b == c and (b != d or b != e):
            return 1
        return 0
    cnt = 1
    while True:
        print(f"iteration : {cnt}")
        old_thin = np.copy(thin_lena)
        pair_img = pair_relationship(thin_lena)
        yokoi_img = yokoi(thin_lena)
        row, col = thin_lena.shape
        
        for i in range(row):
            for j in range(col):    
                if pair_img[i][j] == 3:
                    if i == 0:
                        if j == 0:
                            x7, x2, x6 = 0, 0, 0
                            x3, x0, x1 = 0, thin_lena[i][j], thin_lena[i][j+1]
                            x8, x4, x5 = 0, thin_lena[i+1][j], thin_lena[i+1][j+1]
                        elif j == col - 1:
                            x7, x2, x6 = 0, 0, 0
                            x3, x0, x1 = thin_lena[i][j-1], thin_lena[i][j], 0
                            x8, x4, x5 = thin_lena[i+1][j-1], thin_lena[i+1][j], 0
                        else:
                            x7, x2, x6 = 0, 0, 0
                            x3, x0, x1 = thin_lena[i][j-1], thin_lena[i][j], thin_lena[i][j+1]
                            x8, x4, x5 = thin_lena[i+1][j-1], thin_lena[i+1][j], thin_lena[i+1][j+1]
                    elif i == row - 1:
                        if j == 0:
                            x7, x2, x6 = 0, thin_lena[i-1][j], thin_lena[i-1][j+1]
                            x3, x0, x1 = 0, thin_lena[i][j], thin_lena[i][j+1]
                            x8, x4, x5 = 0, 0, 0
                        elif j == col - 1:
                            x7, x2, x6 = thin_lena[i-1][j-1], thin_lena[i-1][j], 0
                            x3, x0, x1 = thin_lena[i][j-1], thin_lena[i][j], 0
                            x8, x4, x5 = 0, 0, 0
                        else:
                            x7, x2, x6 = thin_lena[i-1][j-1], thin_lena[i-1][j], thin_lena[i-1][j+1]
                            x3, x0, x1 = thin_lena[i][j-1], thin_lena[i][j], thin_lena[i][j+1]
                            x8, x4, x5 = 0, 0, 0
                    else:
                        if j == 0:
                            x7, x2, x6 = 0, thin_lena[i-1][j], thin_lena[i-1][j+1]
                            x3, x0, x1 = 0, thin_lena[i][j], thin_lena[i][j+1]
                            x8, x4, x5 = 0, thin_lena[i+1][j], thin_lena[i+1][j+1]
                        elif j == col - 1:
                            x7, x2, x6 = thin_lena[i-1][j-1], thin_lena[i-1][j], 0
                            x3, x0, x1 = thin_lena[i][j-1], thin_lena[i][j], 0
                            x8, x4, x5 = thin_lena[i+1][j-1], thin_lena[i+1][j], 0
                        else:
                            x7, x2, x6 = thin_lena[i-1][j-1], thin_lena[i-1][j], thin_lena[i-1][j+1]
                            x3, x0, x1 = thin_lena[i][j-1], thin_lena[i][j], thin_lena[i][j+1]
                            x8, x4, x5 = thin_lena[i+1][j-1], thin_lena[i+1][j], thin_lena[i+1][j+1]
                    a1 = h(x0, x1, x6, x2)
                    a2 = h(x0, x2, x7, x3)
                    a3 = h(x0, x3, x8, x4)
                    a4 = h(x0, x4, x5, x1)
                    if a1 + a2 + a3 + a4 == 1:
                        thin_lena[i][j] = 0
        if (thin_lena == old_thin).all():
            break        
        cnt += 1

    cv2.imwrite('thin_lena.bmp', thin_lena)
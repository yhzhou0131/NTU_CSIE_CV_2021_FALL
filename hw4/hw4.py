import matplotlib.pyplot as plt
import numpy as np
import cv2

path = './lena.bmp'

kernel = [
             [-2, -1], [-2, 0], [-2, 1],
    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
    [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
    [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
             [2, -1], [2, 0], [2, 1]
]

kernel_j = [
    [0, -1], [0, 0], [1, 0]
]

kernel_k = [
    [-1, 0], [-1, 1], [0, 1]
]

def binarize(img):
    return (img > 127) * 0xff

def dilation(a, b):
    row_a, col_a = a.shape
    img = np.zeros(a.shape, dtype='int32')

    for a_i in range(row_a):
        for a_j in range(col_a):
            if a[a_i, a_j] == 0xff:
                img[a_i, a_j] = 0xff
                
                for lis in kernel:
                    b_i, b_j = lis
                    if (a_i + b_i) in np.arange(row_a) and (a_j + b_j) in np.arange(col_a):
                        img[a_i+b_i, a_j+b_j] = 0xff 
    return img

def erosion(a, b, mode):
    row_a, col_a = a.shape
    img = np.zeros(a.shape, dtype='int32')

    for a_i in range(row_a):
        for a_j in range(col_a):
            if a[a_i, a_j] == 0xff and not mode:
                img[a_i, a_j] = 0xff
                flag = 1
                
                for lis in b:
                    b_i, b_j = lis
                    if (a_i + b_i) >= row_a or (a_i + b_i) < 0 or \
                        (a_j + b_j) >= col_a or (a_j + b_j) < 0 or \
                        a[a_i+b_i, a_j+b_j] != 0xff:
                        flag = 0
                        break
                if flag == 0:
                    img[a_i, a_j] = 0
                    
            elif mode == 1:
                flag = True
                
                for lis in b:
                    b_i, b_j = lis
                    if (a_i + b_i) >= row_a or (a_i + b_i) < 0 or \
                        (a_j + b_j) >= col_a or (a_j + b_j) < 0 or \
                        a[a_i+b_i, a_j+b_j] != 0xff:
                        flag = False
                        break
                if flag:
                    img[a_i, a_j] = 0xff
    return img   

def opening(a, b):
    cv2.imwrite("opening_lena.bmp", dilation(erosion(a, b, 0), b))

def closing(a, b):
    cv2.imwrite("closing_lena.bmp", erosion(dilation(a, b), b, 0))

def hit_and_miss(img, j, k):
    img_comp = -img + 0xff
    return (((erosion(img, j, 0) + erosion(img_comp, k, 1)) / 2) == 0xff) * 0xff

if __name__ == '__main__':
    bmp_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    binary_img = binarize(bmp_img)

    # part(a)
    img = dilation(binary_img, kernel)
    cv2.imwrite("dilation_lena.bmp", img) 

    # part(b)
    img = erosion(binary_img, kernel, 0)
    cv2.imwrite("erosion_lena.bmp", img)
    
    # part(c)
    opening(binary_img, kernel)

    # part(d)
    closing(binary_img, kernel)

    # part(e)
    ham_img = hit_and_miss(binary_img, kernel_j, kernel_k)
    cv2.imwrite("hitAndMiss_lena.bmp", ham_img)
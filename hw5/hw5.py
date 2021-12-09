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

def dilation(a, b):
    row_a, col_a = a.shape
    img = np.zeros(a.shape, dtype='int32')

    for a_i in range(row_a):
        for a_j in range(col_a):
            max_value = 0
            for lis in kernel:
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
    cv2.imwrite("opening_gray_lena.bmp", dilation(erosion(a, b), b))

def closing(a, b):
    cv2.imwrite("closing_gray_lena.bmp", erosion(dilation(a, b), b))

if __name__ == '__main__':
    bmp_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # part(a)
    img = dilation(bmp_img, kernel)
    cv2.imwrite("dilation_gray_lena.bmp", img) 

    # part(b)
    img = erosion(bmp_img, kernel)
    cv2.imwrite("erosion_gray_lena.bmp", img)
    
    # part(c)
    opening(bmp_img, kernel)

    # part(d)
    closing(bmp_img, kernel)
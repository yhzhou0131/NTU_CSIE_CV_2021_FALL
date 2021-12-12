import cv2
import numpy as np
from numpy.lib.arraypad import pad


def Laplacian(img, kernel, threshold):
    row, col = img.shape
    res_img = np.zeros((row - 2, col - 2))

    for i in range(1, row-1):
        for j in range(1, col-1):
            rk, ck = kernel.shape
            magnitude_gradient = 0
            for ki in range(-rk // 2 + 1, rk // 2 + 1):
                for kj in range(-ck // 2 + 1, ck // 2 + 1):
                    magnitude_gradient += img[i+ki][j+kj] * kernel[ki+(rk//2)][kj+(ck//2)]

            if magnitude_gradient >= threshold:
                res_img[i-1][j-1] = 1
            elif magnitude_gradient <= -threshold:
                res_img[i-1][j-1] = -1
            else:
                res_img[i-1][j-1] = 0
    return res_img

def zero_crossing(pad_img, t):
    row, col = pad_img.shape
    res_img = np.zeros((row-2, col-2))

    for i in range(1, row-1):
        for j in range(1, col-1):
            res_img[i-1][j-1] = 255
            if pad_img[i][j] >= t:
                for ki in range(-1, 2):
                    for kj in range(-1, 2):
                        if pad_img[i+ki][j+kj] <= -t:
                            res_img[i-1][j-1] = 0
                            break
    return res_img

def Laplacian_of_Gaussian(pad_img, threshold):
    kernel = np.array([
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
            [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
            [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
            [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
            [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
            [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
        ])
    row, col = pad_img.shape
    res_img = np.zeros((row - 10, col - 10))
    rk, ck = kernel.shape

    for i in range(5, row-5):
        for j in range(5, col-5):
            magnitude_gradient = 0
            for ki in range(-rk // 2 + 1, rk // 2 + 1):
                for kj in range(-ck // 2 + 1, ck // 2 + 1):
                    magnitude_gradient += pad_img[i+ki][j+kj] * kernel[ki+(rk//2)][kj+(ck//2)]

            if magnitude_gradient >= threshold:
                res_img[i-5][j-5] = 1
            elif magnitude_gradient <= -threshold:
                res_img[i-5][j-5] = -1
            else:
                res_img[i-5][j-5] = 0
        print(i)
    return res_img

def Difference_of_Gaussian(pad_img, threshold):
    kernel = np.array([
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
            [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
            [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
            [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
            [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
            [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        ])
    row, col = pad_img.shape
    res_img = np.zeros((row - 10, col - 10))
    rk, ck = kernel.shape

    for i in range(5, row-5):
        for j in range(5, col-5):
            magnitude_gradient = 0
            for ki in range(-rk // 2 + 1, rk // 2 + 1):
                for kj in range(-ck // 2 + 1, ck // 2 + 1):
                    magnitude_gradient += pad_img[i+ki][j+kj] * kernel[ki+(rk//2)][kj+(ck//2)]

            if magnitude_gradient >= threshold:
                res_img[i-5][j-5] = 1
            elif magnitude_gradient <= -threshold:
                res_img[i-5][j-5] = -1
            else:
                res_img[i-5][j-5] = 0
        print(i)
    return res_img
    

if __name__ == "__main__":
    
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    pad_img = np.pad(img, ((1, 1), (1, 1)), 'edge')
    # # a
    # kernel = [
    #     [0, 1, 0],
    #     [1, -4, 1],
    #     [0, 1, 0]
    # ]
    # kernel = np.array(kernel)
    # threshold = 15
    # a_img = Laplacian(pad_img, kernel, threshold)
    # pad_a_img = np.pad(a_img, ((1, 1), (1, 1)), 'edge')
    # out_img = zero_crossing(pad_a_img, 1)
    # cv2.imwrite("Laplace_mask1.bmp", out_img)

    # # b
    # kernel = [
    #     [1/3, 1/3, 1/3],
    #     [1/3, -8/3, 1/3],
    #     [1/3, 1/3, 1/3]
    # ]
    # kernel = np.array(kernel)
    # threshold = 15
    # b_img = Laplacian(pad_img, kernel, threshold)   
    # pad_b_img = np.pad(b_img, ((1, 1), (1, 1)), 'edge')
    # out_img = zero_crossing(pad_b_img, 1) 
    # cv2.imwrite("Laplace_mask2.bmp", out_img)

    # # c
    # kernel = [
    #     [2/3, -1/3, 2/3],
    #     [-1/3, -4/3, -1/3],
    #     [2/3, -1/3, 2/3]
    # ]
    # kernel = np.array(kernel)
    # threshold = 20
    # c_img = Laplacian(pad_img, kernel, threshold)
    # pad_c_img = np.pad(c_img, ((1, 1), (1, 1)), 'edge')
    # out_img = zero_crossing(pad_c_img, 1)
    # cv2.imwrite("MinVar_Laplacian.bmp", out_img)

    # # d
    # threshold = 3000
    # pad_img = np.pad(img, ((5, 5), (5, 5)), 'edge')
    # d_img = Laplacian_of_Gaussian(pad_img, threshold)
    # pad_d_img = np.pad(d_img, ((1, 1), (1, 1)), 'edge')
    # out_img = zero_crossing(pad_d_img, 1)
    # cv2.imwrite("LG_lena.bmp", out_img)

    # e
    threshold = 1
    pad_img = np.pad(img, ((5, 5), (5, 5)), 'edge')
    e_img = Difference_of_Gaussian(pad_img, threshold)
    pad_e_img = np.pad(e_img, ((1, 1), (1, 1)), 'edge')
    out_img = zero_crossing(pad_e_img, 1)
    cv2.imwrite("DG_lena.bmp", out_img)
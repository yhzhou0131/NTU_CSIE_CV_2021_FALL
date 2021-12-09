import matplotlib.pyplot as plt
import numpy as np
import cv2

path = './lena.bmp'

def image_histogram(img, name):
    hist = [0 for _ in range(256)]
    for c in range(np.size(img, axis=1)):
        for r in range(np.size(img, axis=0)):
            values = img[c, r]
            hist[values] += 1

    x = np.arange(len(hist))
    plt.bar(x, hist)
    plt.xlim(0, 256)
    plt.savefig(name + '_hist.png')
    plt.show()
    return hist

def div_3(img):
    return img // 3

def hist_equal(img, hist):
    cdf_min = 512
    cdf_max = 0
    cdf = [0 for _ in range(len(hist))]
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1] + hist[i]     
        if cdf_max < cdf[i]:
            cdf_max = cdf[i]
        if cdf_min > cdf[i]:
            cdf_min = cdf[i]
    dic = {}
    for pixel in range(len(hist)):
        dic[pixel] = round( (cdf[pixel] - cdf_min) / (cdf_max - cdf_min) * 255 )
    for c in range(np.size(img, axis=1)):
        for r in range(np.size(img, axis=0)):
            img[c, r] = dic[img[c, r]]
    return img

if __name__ == '__main__':
    bmp_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # part(a)
    histo = image_histogram(bmp_img, "ori")

    # part(b)
    div_3_img = div_3(bmp_img)
    cv2.imwrite("div_3_img.png", div_3_img)
    div_3_histo = image_histogram(div_3_img, "div_3")

    # part(c)
    equal_img = hist_equal(div_3_img, div_3_histo)
    cv2.imwrite("equal_img.png", equal_img)
    equal_img_hist = image_histogram(equal_img, "equal_img")
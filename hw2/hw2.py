# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:07:55 2021

@author: naive
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv, cv2

path = './lena.bmp'

class Stack:
    def __init__(self):
        self.list = []

    def push(self,item):
        self.list.append(item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return len(self.list) == 0

def binary_image_threshold(img, threshold):
    for c in range(img.width):
        for r in range(img.height):
            values = img.getpixel((c, r))
            if(values < threshold):
                values = 0
            else:
                values = 255
            img.putpixel((c, r), values)
    img.save("lena_binary.bmp")
    return img

def image_histogram(img):
    hist = [0 for _ in range(256)]
    for c in range(img.width):
        for r in range(img.height):
            values = img.getpixel((c, r))
            hist[values] += 1
    
    histFile = open('lena_hist.csv', "w")
    wri = csv.writer(histFile)
    wri.writerow(hist)
    histFile.close()

    x = np.arange(len(hist))
    plt.bar(x, hist)
    plt.xlim(0, 256)
    plt.savefig('lena_hist.png')
    plt.show()

def image_connected_component(binary_img, region_threshold):
    width, height = binary_img.size
    visited = np.zeros((width, height))
    label_image_array = np.zeros((width, height))
    region_cnt = 1
    number_of_label = np.zeros(width * height)

    for c in range(width):
        for r in range(height):
            if binary_img.getpixel((c, r)) == 0:
                visited[c, r] = 1
            elif visited[c, r] == 0:
                stack = Stack()
                stack.push((c, r))
                while not stack.isEmpty():
                    col, row = stack.pop()

                    if visited[col, row] == 1:
                        continue
                    visited[col, row] = 1
                    label_image_array[col, row] = region_cnt

                    number_of_label[region_cnt] = number_of_label[region_cnt] + 1

                    for x in [col - 1, col, col + 1]:
                        for y in [row - 1, row, row + 1]:
                            if (0 <= x < width) and (0 <= y < height):
                                if (binary_img.getpixel((x, y)) != 0) and (visited[x, y] == 0):
                                    stack.push((x, y))
                region_cnt += 1

    rect = []
    centroid = []
    for regionID, n in enumerate(number_of_label):
        if n >= region_threshold:
            rect_left = width
            rect_right = 0
            rect_top = height
            rect_bottom = 0
            sum_c = 0
            sum_r = 0
            cnt = 0
            for c in range(width):
                for r in range(height):
                    if (label_image_array[c, r] == regionID):
                        sum_c += c
                        sum_r += r
                        cnt += 1
                        if (c < rect_left):
                            rect_left = c
                        if (c > rect_right):
                            rect_right = c
                        if (r < rect_top):
                            rect_top = r
                        if (r > rect_bottom):
                            rect_bottom = r
            # Push rectangle's information to stack.
            rect.append([rect_left, rect_right, rect_top, rect_bottom])
            centroid.append([int(sum_c/cnt), int(sum_r/cnt)])

    cc_img_arr = np.array(binary_image)
    for lst1, lst2 in zip(rect, centroid):
        cv2.rectangle(cc_img_arr, (lst1[0], lst1[2]), (lst1[1], lst1[3]), (150, 0, 0), 3)
        cv2.circle(cc_img_arr, (lst2[0], lst2[1]), 5, (150, 0, 0), -1)

    cv2.imshow('connected_lena', cc_img_arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('connected_lena.bmp', cc_img_arr)

if __name__ == '__main__':
    bmp_image = Image.open(path)
     
    # # part a
    threshold = 128
    binary_image_threshold(bmp_image.copy(), threshold)
    
    #part b
    image_histogram(bmp_image)
    
    #part c
    region_threshold = 500
    binary_image = binary_image_threshold(bmp_image.copy(), threshold)
    image_connected_component(binary_image.copy(), region_threshold)
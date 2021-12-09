from PIL import Image



path = './lena.bmp'

    
if __name__ == '__main__':
    #bmp_image.save("testBMP.bmp")
    bmp_image = Image.open(path)
    rgb_img = bmp_image.convert('RGB')
    upside_down_img = Image.new("RGB", (bmp_image.height, bmp_image.width))
    right_side_left_img = Image.new("RGB", (bmp_image.height, bmp_image.width))
    diagonally_flip_img = Image.new("RGB", (bmp_image.height, bmp_image.width))
    
    for x in range(bmp_image.height):
        for y in range(bmp_image.width):
            (r, g, b) = rgb_img.getpixel((x, y))
            upside_down_img.putpixel((x, bmp_image.height-y-1), (r, g, b))
            right_side_left_img.putpixel((bmp_image.width-x-1, y), (r, g, b))
            diagonally_flip_img.putpixel((y, x), (r, g, b))
    
    upside_down_img.save('upside_down_lena.bmp')
    right_side_left_img.save('right_side_left_lena.bmp')
    diagonally_flip_img.save('diagonally_flip_lena.bmp')
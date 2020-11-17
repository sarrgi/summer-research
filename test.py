import glob
import os
from math import floor
from PIL import Image

def resizeImg(image, location):
    """Resize image, note: all originals currently 5184x3456"""
    compressed = image.resize((10,10),Image.ANTIALIAS) #518,345
    compressed.save(location, optimize=True, quality=95)

def createGreyscale(location):
    """Create and save greyscale version of specified folder of images."""
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            img = img.convert("L")
            print(filename[0:-4]+"_grey.jpg")
            img.save(filename[0:-4]+"_grey.jpg")


def loadAllImages(location):
    """Iterate over all images in specified location,
       open and return them inside a list"""
    img_arr = []
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            pix = img.load()
            img_arr.append(pix)
    return img_arr


def slideWindow(image, window):
    """Notes for tomorrow:
        - currently just slides along and reads each pixel values
        - ignores edge pixels
        - todo: features -> gp"""
    # get image size
    width, height = (10,10)#image.size
    window_x, window_y = window
    # get sizes to iterate window over
    buf_x = floor(window_x/2)
    buf_y = floor(window_y/2)
    x = width - buf_x
    y = height - buf_y

    # print(buf_x,buf_y,x,y)

    #TODO: iterate without index, grey mode
    for i in range(buf_x, x):
        for j in range(buf_y, y):
            # sum = [0,0,0]
            sum = 0
            for k in range(-buf_x, buf_x+1):
                for l in range(-buf_y, buf_y+1):
                    # print("(",i+k,", ", j+l,")", end=' ', sep='')
                    # rgb = image[i+k, j+l] # each pixel value
                    grey = image[i+k, j+l]
                    sum += grey
            #         sum[0] += rgb[0]
            #         sum[1] += rgb[1]
            #         sum[2] += rgb[2]
            # avg = (sum[0]/(window_x*window_y), sum[1]/(window_x*window_y), sum[2]/(window_x*window_y))
            avg = sum/(window_x*window_y)
            print(i,j, sum, avg)


if __name__ == "__main__":
    # img1 = Image.open('images/MNZ_Oyster_0034.jpg')
    # img2 = Image.open('images/MNZ_Oyster_0043.jpg')
    # img3 = Image.open('images/MNZ_Oyster_0061.jpg')
    #
    # # # resize and save images
    # resizeImg(img1, "images/compressed/tiny1.jpg")
    # resizeImg(img2, "images/compressed/tiny2.jpg")
    # resizeImg(img3, "images/compressed/tiny3.jpg")

    # createGreyscale("images/compressed/tiny*.jpg")

    # load pixel values of all images
    imgs = loadAllImages("images/compressed/tiny*_grey.jpg")

    #get image size - NOTE: all images same size
    #

    window = (5,5)
    for i in imgs:
        slideWindow(i, (5,5))

import glob
import os
import itertools
import operator
from math import floor
from PIL import Image

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


def resizeImg(image, location, dimensions):
    """Resize an image to (x,y) dimensions, and save it to a specified location."""
    compressed = image.resize(dimensions,Image.ANTIALIAS) #518,345
    compressed.save(location, optimize=True, quality=95)


def createGreyscale(location):
    """Create and save greyscale version of specified folder of images."""
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            img = img.convert("L")
            img.save(filename[0:-5]+"_grey.jpg") #note 5 as ".jpeg", likely change for actual data


def getDimensions(location):
    """Return the width and height of the first found image in a loction."""
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            # return dimensions of first found image
            return img.size
    # error state
    return -1,-1


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


def mergeInputData(dataList, targetList):
    """Merge two lists of x different classes into a single data list and single target list."""
    data_arr = itertools.chain.from_iterable(dataList)
    target_arr = itertools.chain.from_iterable(targetList)
    return list(data_arr), list(target_arr)



def slideWindow(image, window, img_dimensions):
    """Notes for tomorrow:
        - currently just slides along and reads each pixel values
        - ignores edge pixels
        - todo: features -> gp"""
    # get image size
    width, height = img_dimensions
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
    # load all images
    jute = loadAllImages("images/archive/crop_images/jute/*.jpg")
    maize = loadAllImages("images/archive/crop_images/maize/*.jpg")
    rice = loadAllImages("images/archive/crop_images/rice/*.jpg")
    sugarcane = loadAllImages("images/archive/crop_images/sugarcane/*.jpg")
    wheat = loadAllImages("images/archive/crop_images/wheat/*.jpg")

    # create target lists for all images
    jute_targets = ["jute"] * len(jute)
    maize_targets = ["maize"] * len(maize)
    rice_targets = ["rice"] * len(rice)
    sugarcane_targets = ["sugarcane"] * len(sugarcane)
    wheat_targets = ["wheat"] * len(wheat)

    # get image dimensions - NOTE: all images appear to be 224x224
    width, height = getDimensions("images/archive/crop_images/maize/*.jpg")

    # split into test and train
    # note: only getting first two images of each class for training
    jute_train = jute[0:2]
    maize_train = maize[0:2]
    rice_train = rice[0:2]
    sugarcane_train = sugarcane[0:2]
    wheat_train = wheat[0:2]

    jute_test = jute[2:]
    maize_test = maize[2:]
    rice_test = rice[2:]
    sugarcane_test = sugarcane[2:]
    wheat_test = wheat[2:]

    # get training data
    train_data, train_target = mergeInputData(
        [jute_train, maize_train, rice_train, sugarcane_train, wheat_train],
        [jute_targets[0:2], maize_targets[0:2], rice_targets[0:2], sugarcane_targets[0:2], wheat_targets[0:2]]
    )

    # get test data
    test_data, test_target = mergeInputData(
        [jute_test, maize_test, rice_test, sugarcane_test, wheat_test],
        [jute_targets[2:], maize_targets[2:], rice_targets[2:], sugarcane_targets[2:], wheat_targets[2:]]
    )


    # window = (5,5)
    # for i in imgs:
    #     slideWindow(i, window, (width,height))

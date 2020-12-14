import glob
import os
import itertools
import operator
import numpy
import csv
from math import floor
from PIL import Image


def resize_img(image, location, dimensions):
    """
    Resize an image to (x,y) dimensions, and save it to a specified location.
    """
    compressed = image.resize(dimensions,Image.ANTIALIAS) #518,345
    compressed.save(location, optimize=True, quality=95)


def create_greyscale(location):
    """
    Create and save greyscale version of specified folder of images.
    """
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            img = img.convert("L")
            img.save(filename[0:-5]+"_grey.jpg") #note 5 as ".jpeg", likely change for actual data


def get_dimensions(location):
    """
    Return the width and height of the first found image in a loction.
    """
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            # return dimensions of first found image
            return img.size
    # error state
    return -1,-1


def load_all_images(location):
    """
    Iterate over all images in specified location, open and return them inside a list.
    """
    img_arr = []
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            pix = img.load()
            img_arr.append(pix) #TODO: pix
    return img_arr


def merge_input_data(dataList, targetList):
    """
    Merge two lists of x different classes into a single data list and single target list.
    """
    data_arr = itertools.chain.from_iterable(dataList)
    target_arr = itertools.chain.from_iterable(targetList)
    return list(data_arr), list(target_arr)



def slide_window(image, window, img_dimensions, train_data, train_target):
    """
    Notes:
        - currently just slides along and creates the terminal set for each window
        - ignores edge pixels
    """
    # get image size
    width, height = img_dimensions
    window_x, window_y = window

    # get sizes to iterate window over
    buf_x = floor(window_x/2)
    buf_y = floor(window_y/2)
    x = width - buf_x
    y = height - buf_y

    t_list = []
    # iterate over every image
    for i in range(buf_x, x):
        for j in range(buf_y, y):

            terminal_set = []
            # iterate over window
            for k in range(-buf_x, buf_x+1):
                for l in range(-buf_y, buf_y+1):
                    terminal_set.append(image[i+k, j+l])

            t_list.append(terminal_set)

    return t_list

import mygp
import slide
import time
import random
import sys
import glob
import re
import os
from PIL import Image
import test

# def load_images():
#     jute = slide.load_all_images("/vol/grid-solar/sgeusers/sargisfinl/tiny_crop_images/jute/*.jpg")
#     maize = slide.load_all_images("/vol/grid-solar/sgeusers/sargisfinl/tiny_crop_images/maize/*.jpg")
#     rice = slide.load_all_images("/vol/grid-solar/sgeusers/sargisfinl/tiny_crop_images/rice/*.jpg")
#     sugarcane = slide.load_all_images("/vol/grid-solar/sgeusers/sargisfinl/tiny_crop_images/sugarcane/*.jpg")
#     wheat = slide.load_all_images("/vol/grid-solar/sgeusers/sargisfinl/tiny_crop_images/wheat/*.jpg")
#
#     return [jute, maize, rice, sugarcane, wheat]
#
#
# def create_targets(image_classes, image_lengths):
#     """
#     Create a repeating list of image_class labels a specified length.
#     List generated will have image_lengths[i] long lists of image_class[i].
#     Note: len(images_classes) >= len(image_lengths)
#
#     Args:
#         image_classes (`list` of str) : list of image class labels
#         image_lengths (`list` of int) : corresponding list of image lengths.
#     """
#     targets = []
#     for i in range(len(image_lengths)):
#         targets.append([image_classes[i]] * image_lengths[i])
#
#     return targets
class Obj:
    def __init__(self, i, n):
        self.image = i
        self.name = n


def loadAllImagesLocal(location):
    """
    Iterate over all images in specified location, open and return them inside a list.
    """
    img_arr = []
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            # pix = img.load()
            img_arr.append(Obj(img, filename))
    return img_arr



def split_dataset(set, n):
    train = [x[0:n] for x in set]
    test = [x[n:] for x in set]

    return train, test


if __name__ == "__main__":
    # check and set seed
    if len(sys.argv) != 2:
        sys.exit("Incorrect parameter amount.")

    seed_val = sys.argv[1]
    random.seed(seed_val)
    print("Running with seed", seed_val)

    # define window size
    window = (5,5)

    # load all images
    directories = glob.glob("/vol/grid-solar/sgeusers/sargisfinl/data/cropped_and_sorted/*/")
    images = []
    for d in directories:
        dir = d.strip("\\")
        dir = dir.replace("\\", "/")
        dir = dir + "/*.jpg"
        # print(dir)
        # print(dir)
        images.append(loadAllImagesLocal(dir))

    # create all targets
    targets = []
    for c in images:
        for i, im in enumerate(c):
            # get target name
            class_type = re.search("class_[0-9]+", im.name).group(0)
            # create target list
            if i == 0:
                targets.append([class_type] * len(c))

    # get image dimensions
    width, height = test.getDimensions("/vol/grid-solar/sgeusers/sargisfinl/data/cropped_and_sorted/class_4/*.jpg")

    # split targets into test and train
    train_targets = []
    test_targets = []
    for t in range(len(targets)):
        train_targets.append(targets[t][0:2])
        test_targets.append(targets[t][2:])

    # split images into test and train
    train_features = []
    test_features = []
    for i in images:
        train_row_imgs = []
        test_row_imgs = []
        for count, j in enumerate(i):
            if count < 2:
                train_row_imgs.append(i[count].image)
            else:
                test_row_imgs.append(i[count].image)
        train_features.append(train_row_imgs)
        test_features.append(test_row_imgs)


    # merge test and training data
    train_features, train_targets = test.mergeInputData(train_features,train_targets)
    test_features, test_targets = test.mergeInputData(test_features,test_targets)

    # load images
    for i, t in enumerate(train_features):
        train_features[i] = t.load()
    for i, t in enumerate(test_features):
        test_features[i] = t.load()

    # get terminal set from training data
    print("Getting Training Set.")
    train_terminal_set = []
    for image in train_features:
        sw = slide.slide_window(image, window, (width,height), train_features, train_targets)
        for i in sw:
            train_terminal_set.append(i)

    # get terminal set from test data
    print("Getting Test Set.")
    test_terminal_set = []
    for image in test_features:
        sw = slide.slide_window(image, window, (width,height), test_features, test_targets)
        for i in sw:
            test_terminal_set.append(i)

    # remove duplicates from targets
    train_targets = mygp.remove_duplicates(train_targets)
    test_targets = mygp.remove_duplicates(test_targets)

    # normalize input
    train_features = mygp.normalize_input(train_terminal_set)
    test_features = mygp.normalize_input(test_terminal_set)

    start_time = time.time()

    # create toolbox (GP structure)
    print("Evaluating GP.")
    toolbox = mygp.create_toolbox(train_targets, train_features)
    # evaluate the GP
    mygp.evaluate(toolbox, train_features, train_targets, test_features, test_targets, "output.txt", seed_val, start_time)


    # # load all (tiny) images
    # images = load_images()
    #
    # # create target lists for all images
    # image_lengths = [len(x) for x in images]
    # targets = create_targets(["jute", "maize", "rice", "sugarcane", "wheat"], image_lengths)
    #
    # # get image dimensions
    # width, height = slide.get_dimensions("/vol/grid-solar/sgeusers/sargisfinl/tiny_crop_images/maize/*.jpg")
    #
    # # split into test and train
    # train_data, test_data = split_dataset(images, 2)
    # train_targs, test_targs = split_dataset(targets, 2)
    #
    # # merge training and test data
    # train_features, train_targets = slide.merge_input_data(train_data, train_targs)
    # test_features, test_targets = slide.merge_input_data(test_data, test_targs)
    #
    # # get terminal set from training data
    # train_terminal_set = []
    # for image in train_features:
    #     sw = slide.slide_window(image, window, (width,height), train_features, train_targets)
    #     for i in sw:
    #         train_terminal_set.append(i)
    #
    # # get terminal set from test data
    # test_terminal_set = []
    # for image in test_features:
    #     sw = slide.slide_window(image, window, (width,height), test_features, test_targets)
    #     for i in sw:
    #         test_terminal_set.append(i)
    #
    # # remove duplicates from targets
    # train_targets = mygp.remove_duplicates(train_targets)
    # test_targets = mygp.remove_duplicates(test_targets)
    #
    # # normalize input
    # train_features = mygp.normalize_input(train_terminal_set)
    # test_features = mygp.normalize_input(test_terminal_set)
    #
    # start_time = time.time()
    #
    # # create toolbox (GP structure)
    # toolbox = mygp.create_toolbox(train_targets, train_features)
    # # evaluate the GP
    # mygp.evaluate(toolbox, train_features, train_targets, test_features, test_targets, "output.txt", seed_val, start_time)

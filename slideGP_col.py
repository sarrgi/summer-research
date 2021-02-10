import mygp
import slide_col
import time
import random
import sys
import glob
import re
import os
from PIL import Image
import test
import csv

class Obj:
    def __init__(self, i, n):
        self.image = i
        self.name = n
        self.dimensions = -1

    def set_dimensions(self, d):
        self.dimensions = d


def load_all_images_local(location):
    """
    Iterate over all images in specified location, open and return them inside a list.
    """
    img_arr = []
    for filename in glob.glob(location):
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            img = Image.open(filename)
            pix = img.load()
            img_arr.append(Obj(pix, filename))
        f.close()
    return img_arr



def split_dataset(set, n):
    """
    Split a dataset into two at n point.
    """
    train = [x[0:n] for x in set]
    test = [x[n:] for x in set]

    return train, test


def load_image_data(split_dir):
    """
    Loads images in a directory. (Used for split dataset)
    """
    images = []
    directories = glob.glob(split_dir)
    for dir in directories:
        # get jpg files
        dir = dir + "/*.jpg"
        # load all images and their dimensions
        ims = load_all_images_local(dir)
        dimensions = test.get_all_dimensions(dir)
        # store images
        for i in range(len(ims)):
            ims[i].set_dimensions(dimensions[i])
        # store images
        images.append(ims)
    return images


def create_targets(image_set):
    """
    Create targets list from an image set.
    """
    targets = []
    for c in image_set:
        for i, im in enumerate(c):
        # get target name
            class_type = re.search("class_[0-9]+", im.name).group(0)
            # create target list
            if i == 0:
                targets.append([class_type] * len(c))

    return targets


def split_images_with_dimensions(image_set, n):
    train_features = []
    train_feat_dims = []
    test_features = []
    test_feat_dims = []
    for i in image_set:
        train_row_imgs = []
        test_row_imgs = []
        for count, j in enumerate(i):
            if count < n:
                train_row_imgs.append(i[count].image)
                train_feat_dims.append(i[count].dimensions)
            else:
                test_row_imgs.append(i[count].image)
                test_feat_dims.append(i[count].dimensions)
        train_features.append(train_row_imgs)
        test_features.append(test_row_imgs)

    return train_features, train_feat_dims, test_features, test_feat_dims


if __name__ == "__main__":

    mygp.set_code_node_children(8)

    # check and set seed
    if len(sys.argv) != 2:
        sys.exit("Incorrect parameter amount.")

    seed_val = sys.argv[1]
    random.seed(seed_val)
    print("Running with seed", seed_val)

    # define window size
    window = (5,5)

    # get class directories
    # "images/archive/tiny_crops_uneven/*/"
    # "data/sorted_by_class_shell_0_grey/*/"
    # "images/archive/grayscale_crops/*/"
    # "/vol/grid-solar/sgeusers/sargisfinl/data/sorted_by_class_shell_0_grey/*/"
    # "/vol/grid-solar/sgeusers/sargisfinl/data/test/grayscale_crops/*/"
    # "/vol/grid-solar/sgeusers/sargisfinl/data/bovw_0/*/"
    # "/vol/grid-solar/sgeusers/sargisfinl/data/bovw_1/*/"

    main_dir = "images/archive/tiny_crops_uneven_split/"
    train_dir = main_dir + "train/*/"
    test_dir = main_dir + "test/*/"

    print("Dataset: ", main_dir)

    # load images
    train_images = load_image_data(train_dir)
    test_images = load_image_data(test_dir)

    # create all targets
    train_targets = create_targets(train_images)
    test_targets = create_targets(test_images)

    # split train targets for two images for gp
    gp_train_targets, gp_test_targets = split_dataset(train_targets, 2)

    # split train images for two images for gp
    gp_train_features, gp_train_features_dims, gp_test_features, gp_test_features_dims = split_images_with_dimensions(train_images, 2)

    # merge test and training data
    gp_train_features, gp_train_targets = test.mergeInputData(gp_train_features, gp_train_targets)
    gp_test_features, gp_test_targets = test.mergeInputData(gp_test_features, gp_test_targets)

    # keep track of size
    for i, t in enumerate(gp_train_features):
        gp_train_features[i] = (gp_train_features[i], gp_train_features_dims[i])
    for i, t in enumerate(gp_test_features):
        gp_test_features[i] = (gp_test_features[i], gp_test_features_dims[i])

    # get terminal set from training data
    print("Getting Training Set.")
    gp_train_terminal_set = []
    for i in range(len(gp_train_features)):
        sw = slide_col.slide_window(gp_train_features[i][0], window, gp_train_features[i][1])
        for i in sw:
            gp_train_terminal_set.append(i)

    # remove duplicates from targets
    gp_train_targets_copy = gp_train_targets
    gp_train_targets = mygp.remove_duplicates(gp_train_targets)

    # normalize input
    gp_train_features = mygp.normalize_input(gp_train_terminal_set)

    # Start recording time (at creation of gp tree)
    start_time = time.time()

    # create toolbox (GP structure)
    print("Creating Toolbox.")
    toolbox, pset = mygp.create_toolbox(gp_train_targets, gp_train_features, gp_train_features_dims)

    # train gp algorithms
    best_tree = mygp.train(toolbox)

    # evaluate training set
    print("Evaluating Training Set.")
    correct = 0
    incorrect = 0
    for i in range(len(gp_train_features_dims)):
        curr_size = (gp_train_features_dims[i][0] - 4) * (gp_train_features_dims[i][1] - 4)

        previous_sizes = 0
        for j in range(i):
            previous_sizes += (gp_train_features_dims[j][0] - 4) * (gp_train_features_dims[j][1] - 4)

        current_features = gp_train_features[previous_sizes: (previous_sizes+curr_size)]
        pred = mygp.singular_eval(best_tree, toolbox, current_features, gp_train_features_dims[i], gp_train_features, gp_train_features_dims, gp_train_targets)

        if pred == gp_train_targets[i%len(gp_train_targets)]:
            correct += 1
        else:
            incorrect += 1
    print(correct, "/", (correct + incorrect), " acc: ", correct/(correct + incorrect), sep = "")

    # evaluate test set
    correct = 0
    incorrect = 0
    print("Evaluating Test Set.")
    for i in range(len(gp_test_features)):
        sw = slide_col.slide_window(gp_test_features[i][0], window, gp_test_features[i][1])
        test_terminals = []
        for s in sw:
            test_terminals.append(s)
        normalized = mygp.normalize_input(test_terminals)

        pred = mygp.singular_eval(best_tree, toolbox, normalized, gp_test_feature_dims[i], gp_train_features, gp_train_feature_dims,
        gp_train_targets)

        # print(pred, "vs", test_targets[i])
        if pred == test_targets[i]:
            correct += 1
        else:
            incorrect += 1
    print(correct, "/", len(test_features), " acc: ", correct/(correct + incorrect), sep = "")


    output_file = open("output.txt", "w")
    output_file.write("".join(("Code Node Children: ", str(mygp.get_code_node_children()), "\n")))
    output_file.write("".join(("Seed Value:", str(seed_val), "\n")))
    output_file.write("Best tree: \n")
    output_file.write(str(best_tree))
    output_file.write("".join(("\n-----------------------\n", str(correct), "/", str(len(test_features)), " acc: ", str(correct/incorrect))))

    output_file.close()

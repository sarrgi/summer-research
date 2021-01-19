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


class Obj:
    def __init__(self, i, n):
        self.image = i
        self.name = n
        self.dimensions = -1

    def set_dimensions(self, d):
        self.dimensions = d


def loadAllImagesLocal(location):
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


if __name__ == "__main__":
    mygp.set_code_node_children(3)

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
    directories = glob.glob("/vol/grid-solar/sgeusers/sargisfinl/data/test/grayscale_crops/*/")

    images = []
    for d in directories:
        # create directory link
        dir = d.strip("\\")
        dir = dir.replace("\\", "/")
        dir = dir + "/*.jpg"
        # get all images
        ims = loadAllImagesLocal(dir)
        # get all dimensions
        dimensions = test.getAllDimensions(dir)
        # add image dimensions
        for i in range(len(ims)):
            ims[i].set_dimensions(dimensions[i])
        # store images
        images.append(ims)

    # create all targets
    targets = []
    for c in images:
        for i, im in enumerate(c):
            # get target name
            class_type = re.search("class_[0-9]+", im.name).group(0)
            # create target list
            if i == 0:
                targets.append([class_type] * len(c))


    # split targets into test and train
    train_targets = []
    test_targets = []
    for t in range(len(targets)):
        train_targets.append(targets[t][0:2])
        test_targets.append(targets[t][2:])

    # split images into test and train
    train_features = []
    train_feat_dims = []
    test_features = []
    test_feat_dims = []
    for i in images:
        train_row_imgs = []
        test_row_imgs = []
        for count, j in enumerate(i):
            if count < 2:
                train_row_imgs.append(i[count].image)
                train_feat_dims.append(i[count].dimensions)
            else:
                test_row_imgs.append(i[count].image)
                test_feat_dims.append(i[count].dimensions)
        train_features.append(train_row_imgs)
        test_features.append(test_row_imgs)

    # merge test and training data
    train_features, train_targets = test.mergeInputData(train_features, train_targets)
    test_features, test_targets = test.mergeInputData(test_features, test_targets)

    # keep track of size
    for i, t in enumerate(train_features):
        train_features[i] = (train_features[i], train_feat_dims[i])
    for i, t in enumerate(test_features):
        test_features[i] = (test_features[i], test_feat_dims[i])

    # get terminal set from training data
    print("Getting Training Set.")
    train_terminal_set = []
    for i in range(len(train_features)):
        sw = slide.slide_window(train_features[i][0], window, train_features[i][1])
        for i in sw:
            train_terminal_set.append(i)

    # remove duplicates from targets
    train_targets_copy = train_targets
    train_targets = mygp.remove_duplicates(train_targets)
    # test_targets = mygp.remove_duplicates(test_targets)

    # normalize input
    train_features = mygp.normalize_input(train_terminal_set)

    # Start recording time (at creation of gp tree)
    start_time = time.time()

    # create toolbox (GP structure)
    print("Creating Toolbox.")
    toolbox, pset = mygp.create_toolbox(train_targets, train_features, train_feat_dims)

    best_tree = mygp.train(toolbox)

    # print("Starting Test Set.")
    correct = 0
    incorrect = 0
    for i in range(len(test_features)):
        sw = slide.slide_window(test_features[i][0], window, test_features[i][1])
        normalized = mygp.normalize_input(sw)

        # print(normalized)
        pred = mygp.singular_eval(best_tree, toolbox, normalized, test_features[i][1], train_features, train_feat_dims, train_targets)

        # print(pred, "vs", test_targets[i])
        if pred == test_targets[i]:
            correct += 1
        else:
            incorrect += 1

    print("Training eval")

    print(correct, "/", len(test_features), " acc: ", correct/incorrect, sep = "")

    output_file = open("output.txt", "w")
    output_file.write("".join(("Code Node Children: ", str(mygp.get_code_node_children()), "\n")))
    output_file.write("".join(("Seed Value:", str(seed_val), "\n")))
    output_file.write("Best tree: \n")
    output_file.write(str(best_tree))
    output_file.write("".join(("\n-----------------------\n", str(correct), "/", str(len(test_features)), " acc: ", str(correct/incorrect))))

    output_file.close()

    # for i in range(len(train_features)):
    #     curr_size = (train_feat_dims[i][0] - 4) * (train_feat_dims[i][1] - 4)
    #     #
    #     previous_sizes = 0
    #     for j in range(i):
    #         previous_sizes += (dimensions[j][0] - 4) * (dimensions[j][1] - 4)
    #
    #     to_eval = train_features[previous_sizes: curr_size]
    #
    #     # print(train_feat_dims[i])
    #     # sw = sw = slide.slide_window(train_features[i], window, train_feat_dims[i])
    #     # # normalized = mygp.normalize_input(sw)
    #     print(train_features[i], train_feat_dims[i])
    #     pred = mygp.singular_eval(best_tree, toolbox, to_eval, train_feat_dims[i], train_features, train_feat_dims, train_targets)
    #
    #     print(pred, "vs", train_targets_copy[i])
    #     if pred == train_targets_copy[i]:
    #         correct += 1
    #     else:
    #         incorrect += 1











    # # get terminal set from test data
    # print("Getting Test Set.")
    # test_terminal_set = []
    # for i in range(len(test_features)):
    #     sw = slide.slide_window(test_features[i][0], window, test_features[i][1])
    #     for i in sw:
    #         test_terminal_set.append(i)


    # test_features = mygp.normalize_input(test_terminal_set)

    # # for t in train_features:
    # #     print(len(train_features), t)

    # # evaluate the GP
    # mygp.evaluate(toolbox, train_features, train_targets, test_features, test_targets, train_feat_dims, test_feat_dims,"output.txt", seed_val, start_time)

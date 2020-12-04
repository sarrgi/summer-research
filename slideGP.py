import mygp
import test

# GP
if __name__ == "__main__":
    window = (5,5)

    # load all (tiny) images
    jute = test.loadAllImages("images/archive/tiny_crop_images/jute/*.jpg")
    maize = test.loadAllImages("images/archive/tiny_crop_images/maize/*.jpg")
    rice = test.loadAllImages("images/archive/tiny_crop_images/rice/*.jpg")
    sugarcane = test.loadAllImages("images/archive/tiny_crop_images/sugarcane/*.jpg")
    wheat = test.loadAllImages("images/archive/tiny_crop_images/wheat/*.jpg")

    # create target lists for all images
    jute_targets = ["jute"] * len(jute)
    maize_targets = ["maize"] * len(maize)
    rice_targets = ["rice"] * len(rice)
    sugarcane_targets = ["sugarcane"] * len(sugarcane)
    wheat_targets = ["wheat"] * len(wheat)

    # get image dimensions
    width, height = test.getDimensions("images/archive/tiny_crop_images/maize/*.jpg")

    # split into test and train - note: only getting first two images of each class for training
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
    train_features, train_targets = test.mergeInputData(
        [jute_train, maize_train, rice_train, sugarcane_train, wheat_train],
        [jute_targets[0:2], maize_targets[0:2], rice_targets[0:2], sugarcane_targets[0:2], wheat_targets[0:2]]
    )

    # get test data
    test_features, test_targets = test.mergeInputData(
        [jute_test, maize_test, rice_test, sugarcane_test, wheat_test],
        [jute_targets[2:], maize_targets[2:], rice_targets[2:], sugarcane_targets[2:], wheat_targets[2:]]
    )

    # get terminal set from training data
    terminal_set = []
    for image in train_features:
        terminal_set.append(test.slideWindow(image, window, (width,height), train_features, train_targets))

    # print(train_targets)
    # print(terminal_set)



    # gp
    toolbox = mygp.createToolbox(train_targets, terminal_set)
    mygp.evaluate(toolbox, train_targets, train_features, train_targets, train_features)
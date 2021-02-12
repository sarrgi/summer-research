import numpy as np
import cv2
import os
import sys

from PIL import Image

from matplotlib import pyplot as plt

from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from info_gain import info_gain


# takes all images and convert them to grayscale.
# return a dictionary that holds all images category by category.
def load_images_from_folder(folder):
    """
    images[class][image_no][image_y][image_x]
    """
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat,0)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category

    return images


def sift_features(images, layers, sig, contrast, edge):
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.SIFT.create(nOctaveLayers=layers, sigma=sig, contrastThreshold=contrast, edgeThreshold=edge)
    for key,value in images.items():
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img,None)


            descriptor_list.extend(des)
            features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]


def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_
    return visual_words


def image_class(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature


def knn(images, tests):
    num_test = 0
    correct_predict = 0
    class_based = {}

    for test_key, test_val in tests.items():
        class_based[test_key] = [0, 0] # [correct, all]
        for tst in test_val:
            predict_start = 0
            #print(test_key)
            minimum = 0
            key = "a" #predicted
            for train_key, train_val in images.items():
                for train in train_val:
                    if(predict_start == 0):
                        minimum = distance.euclidean(tst, train)
                        #minimum = L1_dist(tst,train)
                        key = train_key
                        predict_start += 1
                    else:
                        dist = distance.euclidean(tst, train)
                        #dist = L1_dist(tst,train)
                        if(dist < minimum):
                            minimum = dist
                            key = train_key

            if(test_key == key):
                correct_predict += 1
                class_based[test_key][0] += 1
            num_test += 1
            class_based[test_key][1] += 1
            #print(minimum)
    return [num_test, correct_predict, class_based]


def info_gain(images, tests, model):
    """
    Return features (worst to best)
    """
    train_targets, train_images = convert_to_clf_form(images)
    test_targets, test_images = convert_to_clf_form(tests)

    clf = model
    clf.fit(train_images, train_targets)

    unordered_feats = []
    importance = model.feature_importances_

    for i, v in enumerate(importance):
        unordered_feats.append((i,v))

    sorted_by_feats = sorted(unordered_feats, key=lambda tup: tup[1])
    sorted_by_feats.reverse()

    return sorted_by_feats


def remove_feats(features, feature_rankings, cut_off):
    features_copy = features

    arbitrary_point = int(cut_off*len(feature_rankings))

    # find indexes to keep
    to_keep_ind = ()
    for i, feat in enumerate(feature_rankings):
        if i < arbitrary_point: to_keep_ind += (feat[0],)

    # remove unwanted indexes
    for c in features_copy:
        for ind, feat in enumerate(features_copy[c]):
            removed = [v for i,v in enumerate(feat) if i in frozenset(to_keep_ind)]
            removed_np = np.asarray(removed, dtype=float)
            features_copy[c][ind] = removed_np

    return features_copy



def eval(images, tests, model):
    train_targets, train_images = convert_to_clf_form(images)
    test_targets, test_images = convert_to_clf_form(tests)

    clf = model
    clf.fit(train_images, train_targets)
    predictions = clf.predict(test_images)

    return compare_results(predictions, test_targets), predictions, test_targets


def convert_to_clf_form(dictionary):
    targets = []
    images = []

    for key in dictionary:
        for im in dictionary[key]:
            images.append(im)
            targets.append(key)

    return targets, images


def compare_results(predictions, actual):
    class_results = {}
    correct = 0

    for i in range(len(actual)):
        to_add = 0

        if predictions[i] == actual[i]:
            correct += 1
            to_add = 1

        if actual[i] not in class_results:
            class_results[actual[i]] = [to_add, 1]
        else:
            class_results[actual[i]] = [class_results[actual[i]][0] + to_add, class_results[actual[i]][1] + 1]

    return [len(actual), correct, class_results]


def accuracy(results):
    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy:" + str(avg_accuracy), "%.")
    print("\nClass based accuracies: \n")
    for key,value in results[2].items():
        acc = (value[0] / value[1]) * 100
        print(key + " : %" + str(acc))


def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i])
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind


if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit("Incorrect parameter amount.")
    k_val = int(sys.argv[1])
    # info_gain_thresh = float(sys.argv[2])

    print("Running with k:", k_val)
    print("Running with info thresh:", 0.8)

    # /vol/grid-solar/sgeusers/sargisfinl/data/bovw_0_color_run/train
    # data/gcp/bovw_0_color_run/train
    dir = '/vol/grid-solar/sgeusers/sargisfinl/data/bovw_0_color_run/'
    train = load_images_from_folder(dir + 'train')  # take all images category by category
    test = load_images_from_folder(dir + 'test') # take test images

    print("Dataset:", dir)

    # classes = ["color_1", "color_2", "color_3", "color_4", "color_5"]

    layers = 3
    sig = 2.5
    contrast =  0.065
    edge = 3
    print('Sift with params: Layers - %d, Sigma - %.1f, Contrast - %.3f, Edge - %d,' % (layers, sig, contrast, edge))
    sifts = sift_features(train, layers, sig, contrast, edge)
    # Takes the descriptor list which is unordered one
    descriptor_list = sifts[0] # 53418 long array of 128 long features
    # Takes the sift features that is seperated class by class for train data
    train_bovw_feature = sifts[1]
    # Takes the sift features that is seperated class by class for test data
    test_bovw_feature = sift_features(test, layers, sig, contrast, edge)[1]

    print('kmeans')
    # Takes the central points which is visual words
    visual_words = kmeans(k_val, descriptor_list)

    print('hist')
    # Creates histograms for train data
    bovw_train = image_class(train_bovw_feature, visual_words)
    # Creates histograms for test data
    bovw_test = image_class(test_bovw_feature, visual_words)

    # info gain experiments
    ranked_features = info_gain(bovw_train, bovw_test, RandomForestClassifier())
    bovw_filtered_train = remove_feats(bovw_train, ranked_features, 0.8)
    bovw_filtered_test = remove_feats(bovw_test, ranked_features, 0.8)

    print("-----------------------------------------------")
    print('Random forest (Info Gain)')
    results_bowl, y_pred, y_true = eval(bovw_filtered_train, bovw_filtered_test, RandomForestClassifier())
    accuracy(results_bowl)
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)

    print("-----")
    print('Random forest (Default)')
    results_bowl, y_pred, y_true = eval(bovw_train, bovw_test, RandomForestClassifier())
    accuracy(results_bowl)
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    print("-----")


    exit(1)

    print('Decision Tree')
    results_bowl, y_pred, y_true = eval(bovw_train, bovw_test, DecisionTreeClassifier())
    accuracy(results_bowl)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("-----")

    print('Knn (3)')
    k = KNeighborsClassifier()
    k.set_params(n_neighbors=3)
    results_bowl, y_pred, y_true = eval(bovw_train, bovw_test, k)
    accuracy(results_bowl)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("-----")

    print('Knn (1) sk')
    k = KNeighborsClassifier()
    k.set_params(n_neighbors=1)
    results_bowl, y_pred, y_true = eval(bovw_train, bovw_test, k)
    accuracy(results_bowl)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("-----")


    print("Knn (1) manual")
    results_bowl = knn(bovw_train, bovw_test)
    accuracy(results_bowl)
    print("-----")
    print("-----------------------------------------------")

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

# takes all images and convert them to grayscale.
# return a dictionary that holds all images category by category.
def load_images_from_folder(folder):
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
    kmeans = KMeans(n_clusters = k, n_init=100)
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
    print("Average accuracy: %" + str(avg_accuracy))
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


# def plotConfusionMatrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     cm = confusion_matrix(y_true, y_pred)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax


if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit("Incorrect parameter amount.")
    k_val = int(sys.argv[1])

    print("Running with k: ", k_val)

    # /vol/grid-solar/sgeusers/sargisfinl/data/bovw_0_color_run/train
    # data/gcp/bovw_0_color_run/train
    dir = '/vol/grid-solar/sgeusers/sargisfinl/data/bovw_0_color_run/'
    images = load_images_from_folder(dir + 'train')  # take all images category by category
    test = load_images_from_folder(dir + 'test') # take test images

    print("Dataset:", dir)

    # classes = ["color_1", "color_2", "color_3", "color_4", "color_5"]

    print('Sift with params: Layers - %d, Sigma - %.1f, Contrast - %.3f, Edge - %d,' % (3, 2.5, 0.065, 3))
    sifts = sift_features(images, 3, 1.3, 0.065, 3)
    # Takes the descriptor list which is unordered one
    descriptor_list = sifts[0]
    # Takes the sift features that is seperated class by class for train data
    all_bovw_feature = sifts[1]
    # Takes the sift features that is seperated class by class for test data
    test_bovw_feature = sift_features(test, 3, 1.3, 0.065, 3)[1]

    print('kmeans')
    # Takes the central points which is visual words
    visual_words = kmeans(k_val, descriptor_list)



    #
    vvs = visual_words[0].reshape(8, 16)

    for v in vvs:
        array = np.array(v, dtype=np.uint8)
        img = Image.fromarray(array)
        img.show()

    exit(1)
    #


    print('hist')
    # Creates histograms for train data
    bovw_train = image_class(all_bovw_feature, visual_words)
    # Creates histograms for test data
    bovw_test = image_class(test_bovw_feature, visual_words)

    print("-----------------------------------------------")
    print('Random forest')
    results_bowl, y_pred, y_true = eval(bovw_train, bovw_test, RandomForestClassifier())
    accuracy(results_bowl)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("-----")

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

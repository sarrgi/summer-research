import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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


def sift_features(images):
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.SIFT.create()
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


def eval(images, tests, model):
    train_targets, train_images = convert_to_clf_form(images)
    test_targets, test_images = convert_to_clf_form(tests)
    #
    clf = model
    # print(type(images.items()), images.items())
    # print("--")
    # print(type(images.keys()), images.keys())
    clf.fit(train_images, train_targets)
    predictions = clf.predict(test_images)
    return compare_results(predictions, test_targets)


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
    for i in range(len(predictions)):
        # check if correct
        to_add = 0
        if predictions[i] == actual[i]:
            correct += 1
            to_add = 1

        # add to dict
        if predictions[i] in class_results:
            class_results[predictions[i]] = [class_results[predictions[i]][0] + to_add, class_results[predictions[i]][1] + 1]
        else:
            class_results[predictions[i]] = [to_add, 1]

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



if __name__ == "__main__":

    images = load_images_from_folder('/vol/grid-solar/sgeusers/sargisfinl/data/bovw_0_color_run/train')  # take all images category by category
    test = load_images_from_folder("/vol/grid-solar/sgeusers/sargisfinl/data/bovw_0_color_run/test") # take test images

    print('sift')
    sifts = sift_features(images)
    # Takes the descriptor list which is unordered one
    descriptor_list = sifts[0]
    # Takes the sift features that is seperated class by class for train data
    all_bovw_feature = sifts[1]
    # Takes the sift features that is seperated class by class for test data
    test_bovw_feature = sift_features(test)[1]

    print('kmeans')
    # Takes the central points which is visual words
    visual_words = kmeans(50, descriptor_list)

    print('hist')
    # Creates histograms for train data
    bovw_train = image_class(all_bovw_feature, visual_words)
    # Creates histograms for test data
    bovw_test = image_class(test_bovw_feature, visual_words)

    print("-----------------------------------------------")
    print('Random forest')
    results_bowl = eval(bovw_train, bovw_test, RandomForestClassifier())
    accuracy(results_bowl)
    print("-----")

    print('Decision Tree')
    results_bowl = eval(bovw_train, bovw_test, DecisionTreeClassifier())
    accuracy(results_bowl)
    print("-----")

    print('Knn (3)')
    k = KNeighborsClassifier()
    k.set_params(n_neighbors=3)
    results_bowl = eval(bovw_train, bovw_test, k)
    accuracy(results_bowl)
    print("-----")

    print("Knn (1)")
    results_bowl = knn(bovw_train, bovw_test)
    accuracy(results_bowl)
    print("-----")
    print("-----------------------------------------------")

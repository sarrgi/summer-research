# todo - seperate gp over here
import csv
import itertools
import operator
import numpy
import random
import time
import re
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# from scoop import futures
import multiprocessing

code_node_children = 8

def get_code_node_children():
    return code_node_children

def set_code_node_children(x):
    global code_node_children
    code_node_children = x


def merge_input_data(data_list, target_list):
    """
    Merge two lists of x different classes into a single data list and single target list.
    """
    data_arr = itertools.chain.from_iterable(data_list)
    target_arr = itertools.chain.from_iterable(target_list)
    return list(data_arr), list(target_arr)


def protected_div(left, right):
    """
    Performs regular division, with the exception of safeguarding potential division by 0 issues.
    In this case it will return 0. (Using 0 (and not 1) as this is specified by GP-criptor paper.)

    Args:
        left (number): Numerator.
        right (number): Denominator.

    Returns:
        0 if right == 0, left/right otherwise
    """
    try:
        return left / right
    except:
        return 0


def convert_to_decimal(binary_string):
    """
    Converts a binary string into its corresponding integer value.

    Args:
        binary_string (str): Binary representation of a number.

    Returns:
        Integer value of the binary string.
    """
    return int(binary_string, 2)


def generate_feature_vector(length, func, features):
    """
    Generate a single feature vector from a single image.

    Args:
        individual (registered individual in toolbox): Individual GP tree.
        toolbox (deap.Base.Toolbox): toolbox of set used to compile the individual into a callable method.
        features (`list` of int): list of features of a single image.

    Returns:
        Feature vector.
    """
    # print(len(features), length)
    feature_vector = [0] * length

    for i in range(len(features)):
        pos = int(func(*features[i]))
        feature_vector[pos] += (1/len(features))

    # print(feature_vector)
    return feature_vector


def generate_all_feature_vectors(individual, toolbox, features, targets, dimensions):
    """
    Generate the feature vector for all images in the training set.

    Args:
        individual (registered individual in toolbox): Individual GP tree.
        toolbox (deap.Base.Toolbox): toolbox of set used to compile the individual into a callable method.
        features (`list` of `list` of int): list of features of all images in the set.
        targets  (`list` of `str`): list of all images target class.

    Returns:
        list of feature vectors for all images in the training set.
    """
    feature_vectors = {}
    func = toolbox.compile(expr=individual)

    for i in range(len(dimensions)):
        # get specific class from target name
        target = targets[i%len(targets)]#re.sub(r'[0-9]+', '', targets[i])  count
        # get current image's features from  list - TODO NOT HARDCODED
        # val = (width - floor(window/2)) * (height - floor(window/2))
        curr_size = (dimensions[i][0] - 4) * (dimensions[i][1] - 4)
        #
        previous_sizes = 0
        for j in range(i):
            previous_sizes += (dimensions[j][0] - 4) * (dimensions[j][1] - 4)
        # print(i,curr_size, previous_sizes, len(features))

        current_features = features[previous_sizes: (previous_sizes+curr_size)]
        # create feature vector from current images

        if target in feature_vectors:
            feature_vectors[target].append(generate_feature_vector(pow(2, code_node_children), func, current_features))
        else:
            feature_vectors[target] = [generate_feature_vector(pow(2, code_node_children), func, current_features)]
    # print("-----")
    # print(feature_vectors)

    # print(dimensions, len(features))
    # print("-------------")
    # print(feature_vectors)
    # print("------------")

    return feature_vectors




def distance_vectors(u, v):
    """
    Calculate the distance between to vectors.
    Note: requires len(u) == len(v)
    """
    sum = 0

    # normalize feature vector
    u = normalize_vector(u)
    v = normalize_vector(v)

    for i in range(len(u)):
        if u[i] + v[i] != 0: #avoid divide by 0 errors
            sum += (pow((u[i]-v[i]), 2) / (u[i] + v[i]))

    sum /= 2
    return sum


def distance_between_and_within(set):
    """
    TODO:
        - calc distanc between all feature vectors in training set
        - and then return DB and DW values
    """

    class_count = len(set)
    inst_per_class = len(list(set.values())[0])
    total_inst = class_count * inst_per_class

    dist_within = 0
    dist_between = 0

    # cycle trhough all images in set
    for key in set:
        for count_1, img in enumerate(set[key]):
            curr_img = img

            # compare to all images in set
            for key2 in set:

                for count_2, img in enumerate(set[key2]):
                    compare_img = img
                    # print(key, key2, count_1, count_2)

                    # ensure not comparing object to self
                    if key == key2 and count_1 == count_2:
                        continue

                    # calc distance
                    dist = distance_vectors(curr_img, compare_img)
                    # print("Keys:", key, key2, "- Counts:", count_1, count_2, "- Dist:", dist)

                    # add dist to respective count
                    if key == key2:
                        dist_within += dist
                    else:
                        dist_between += dist

    # print((total_inst * (total_inst - inst_per_class)), (total_inst * (inst_per_class - 1)))
    # print(dist_within, dist_between, (total_inst * (inst_per_class - 1)), (total_inst * (total_inst - inst_per_class)))
    dist_within /= (total_inst * (inst_per_class - 1))
    dist_between /= (total_inst * (total_inst - inst_per_class))

    # print(dist_within, dist_between)

    return dist_within, dist_between


def fitness_func(individual, toolbox, features, targets, dimensions):
    """
    TODO: implement distance based fitness function (chi square only)
    """

    # print("----")
    feature_vectors = generate_all_feature_vectors(individual, toolbox, features, targets, dimensions)

    # print(len(feature_vectors['class_1'][0]))


    dist_within, dist_between = distance_between_and_within(feature_vectors)

    fit = 1 / (1 + pow(math.e, (-5 * (dist_within - dist_between))))
    return 1-fit, #TODO - check 1 minus ?


def code_node(*args):
    """
    Root node of the GP tree.
    Evaluates all children and returns a binary string based on the evaluations.

    Args:
        *args (`list` of children nodes):

    Returns:
        Decimal value as a float.

    NOTE:
        - children appear to be automatically evaluated (can be floats or ints)
        - currently returns binary value (as a float to distinguish this as root node)
    """

    # calculate binary string based on child nodes
    binary_string = ""

    # evaluate binary tree
    for v in args:
        if v < 0.0:
            binary_string = "".join((binary_string, "0"))
        else:
            binary_string = "".join((binary_string, "1"))

    # convert to a decimal value
    d = convert_to_decimal(binary_string)

    return d


def code_node_color(*args):
    """
        Concept Ideas:
        NOTE: *args is the evaluated childrne being passed through

        #1
        - there are three args: r, g, b tree
        - each of these are efively just a code node, each with color cordinated channels
        - each of these produce a binary string of len(2^n)
        - this just combines three string to product binary string len(3*2^n)

        #2
        - orig code node just gets passed through 3x list of args,
          determines on it's own which features from which channels are valuable

        #3
        - some sort of system which give priority to blue input, due to this being likely more substantial in differneing colors

        #4
        - three r, g, b tree which produce three respective histograms
        - new input is compared to each histogram respectively and given a chosen class from each
        - use popular vote to classify


    """


    return -1


def normalize_vector(vec):
    """
    Normalize a list (vector) so all values inside of the vector are between 0 and 1.

    Args:
        vec (list) : Vector to be normalized.
    """
    return [float(i)/sum(vec) for i in vec]


def create_toolbox(train_targets, train_features, train_dims):
    """
    Create a toolbox for evolving and evaluating the GP tree.

    Args:
        train_targets (`list` of str):  list of features of all images in the set.
        train_features (`list` of `list` of int): list of all images target class.

    Returns:
        Toolbox object for evolving and evaluating the GP tree.
    """
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, len(train_features[0])), int, prefix='F')

    # define primitive set
    pset.addPrimitive(operator.add, [float, float], float, name="ADD")
    pset.addPrimitive(operator.sub, [float, float], float, name="SUB")
    pset.addPrimitive(operator.mul, [float, float], float, name="MULT")
    pset.addPrimitive(protected_div, [float, float], float, name="PDIV")

    # pset.addPrimitive(operator.add, [int, int], float, name="FLOAT")
    pset.addPrimitive(code_node, ([float] * code_node_children), int)

    # define terminal set
    for i in range(pow(2, code_node_children)):
        pset.addTerminal(i, int)

    # creates fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # toolbox.register("map", futures.map)
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=10) #min and max of generated trees
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", fitness_func, toolbox=toolbox, features=train_features,targets=train_targets, dimensions=train_dims)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_ = 2, max_ = 10) #TODO check if these constraints are necessary
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset = pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    return toolbox, pset


def train(toolbox):
    """
    Train the toolbox and create the best tree.
    """
    print("Training.")
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.2, 50, stats, halloffame=hof, verbose=True) #TODO: 30 gens

    return hof[0]


def evaluate(toolbox, train_features, train_targets, test_features, test_targets, train_dimensions, test_dimensions, filename, seed_val, start_time):
    """
        TODO - proper evaluation
    """
    pop = toolbox.population(n=300) #TODO: 500
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    print("Training.")
    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.2, 10, stats, halloffame=hof, verbose=True) #TODO: 30 gens
    print("Training done.")
    print("Evaluating.")

    output_file = open(filename, "w")

    output_file.write("".join(("Seed Value:", str(seed_val), "\n")))
    output_file.write("HOF: \n")
    output_file.write(str(hof[0]))

    output_file.write("\nTraining Accuracy:")

    # print(len(dimensions), len(train_features), len(test_features))
    # output_file.write(str(fitness_func(hof[0], toolbox, train_features, train_targets, dimensions)[0]))
    for i in range(len(train_features)):
        # output_file.write(str(fitness_func(hof[0], toolbox, train_features[i], train_targets[i], dimensions[i])))
        fitness_func(hof[0], toolbox, [train_features[i]], train_targets, [train_dimensions[i]])
    output_file.write("\nTest Accuracy:")
    # output_file.write(str(fitness_func(hof[0], toolbox, test_features, test_targets, dimensions)[0]))
    for i in range(len(test_features)):
        # output_file.write(str(fitness_func(hof[0], toolbox, test_features[i], test_targets[i], dimensions[i])))
        fitness_func(hof[0], toolbox, [test_features[i]], test_targets, [test_dimensions[i]])

    output_file.write("".join(("\nTime taken: ", "{:.2f}".format(time.time() - start_time), " seconds.")))
    output_file.write("\n--------------------------")

    output_file.close()
    print("Evaluation Complete.")


def singular_eval(best_tree, toolbox, test_features, test_dimensions, train_features, train_dimensions, train_targets):
    """
    ... yeah this method sucks, TODO: actual AI library

    Evaluate a singular test instance against the entire trainign instance.

    TODO: normalize distance func?
    """
    func = toolbox.compile(expr=best_tree)
    # func(*features)
    test_feat_vec = generate_feature_vector(pow(2, code_node_children), func, test_features)

    train_feat_vecs = generate_all_feature_vectors(best_tree, toolbox, train_features, train_targets, train_dimensions)
    # need to convert feat_vec into an actual class
    predicted = nearest_neighbour(test_feat_vec, train_feat_vecs, train_targets)
    return predicted



def nearest_neighbour(individual, neighbours, neighbour_targets):
    best_dist = distance_vectors(individual, neighbours[neighbour_targets[0]][0])
    best_target = neighbour_targets[0]

    for target in neighbour_targets:
        for i in neighbours[target]:
            dist = distance_vectors(individual, i)
        # dist = distance_vectors(individual, n)
            if dist < best_dist:
                best_dist = dist
                best_target = target


    # print(best_target, best_dist)
    return best_target


def remove_duplicates(list):
    """
    Removes the duplicates of a list.

    Args:
        list (`list` of obj): Any list.

    Returns:
        A list with no duplicate values.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in list if not (x in seen or seen_add(x))]


def read_CSV(file_name):
    """
    Read the csv and split into target and feature lists.

    Args:
        file_name (str): The name (and directory) of file to be opened.
    """
    train_targets = []
    train_features = []

    with open(file_name) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        for row in reader:
            target = row[0]
            features = row[1:]

            train_targets.append(target)
            train_features.append(features)

    return train_targets, train_features


def normalize_input(input):
    """
    Normalize the list of an image input value.
    Normalizes based on the 0-255 range of possible color values.

    Args:
        file_name (`list` of int): The name (and directory) of file to be opened.
    """
    min = 0
    max = 255

    for set in input:
        for i in range(len(set)):
            set[i] = (set[i]-min) / (max-min)

    return input


def convert_tuple_string(s):
    """
    Convert a string of tuples into a list of tuple objects.
    Used for reading in color csv values.

    Args:
        - `list of `str`` : list of strings which are tuples
    """
    pixel_list = []
    for t in s:
        # strip parens
        t = t.strip("()")
        # get values
        split = t.split(",")
        a = [int(x) for x in split]
        # convert to tuples and add to list
        pixels = tuple(a)
        pixel_list.append(pixels)

    return pixel_list


def tuples_to_list(tuples):
    """
    Converts a list of tuples into a singular list.
    List is ordered: R0 -> RN -> G0 -> GN -> B0 -> BN.
    """
    red = [x[0] for x in tuples]
    green = [x[1] for x in tuples]
    blue = [x[2] for x in tuples]

    return red + green + blue


if __name__ == "__main__":

    # read into features and targets
    train_targets, train_features = read_CSV("test_data/5x5trainColor.csv")
    # convert from str to tuple
    for t in range(len(train_features)):
        train_features[t] = convert_tuple_string(train_features[t])
    # remove duplicates
    train_targets = remove_duplicates(train_targets)

    # read into features and targets
    test_targets, test_features = read_CSV("test_data/5x5testColor.csv")
    # convert from str to tuple
    for t in range(len(test_features)):
        test_features[t] = convert_tuple_string(test_features[t])
    # remove duplicates
    test_targets = remove_duplicates(test_targets)


    rgb_train = []
    for t in train_features:
        rgb_train.append(tuples_to_list(t))

    rgb_test = []
    for t in test_features:
        rgb_test.append(tuples_to_list(t))

    # print(rgb_train[0])
    rgb_train = normalize_input(rgb_train)
    rgb_test = normalize_input(rgb_test)
    # print(rgb_train[0])

    # start timing method
    start_time = time.time()

    # create toolbox (GP structure)
    toolbox = create_toolbox(train_targets, rgb_train)
    # evaluate the GP
    evaluate(toolbox, rgb_train, train_targets, rgb_test, test_targets, "output.txt", seed_val)

    # time printout
    print("Time taken: ", "{:.2f}".format(time.time() - start_time), " seconds.", sep='')
    time.sleep(3)






    ###################### GREY SCALE METHOD BELOW #############################


    # # read into features and targets
    # train_targets, train_features = read_CSV("test_data/5x5testUnique.csv")
    # # convert from str to int
    # train_features = [[int(i) for i in j] for j in train_features]
    # #remove duplicates
    # train_targets = remove_duplicates(train_targets)
    #
    # # test
    # test_targets, test_features = read_CSV("test_data/5x5testUnseen.csv")
    # test_features = [[int(i) for i in j] for j in test_features]
    # test_targets = remove_duplicates(test_targets)
    #
    # test_features = normalize_input(test_features)
    # train_features = normalize_input(train_features)
    #
    #
    # # start timing method
    # start_time = time.time()
    #
    # # create toolbox (GP structure)
    # toolbox = create_toolbox(train_targets, train_features)
    # # evaluate the GP
    # evaluate(toolbox, train_features, train_targets, test_features, test_targets)
    #
    # # time printout
    # print("Time taken: ", "{:.2f}".format(time.time() - start_time), " seconds.", sep='')
    # time.sleep(3)

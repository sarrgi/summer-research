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
    feature_vector = [0] * length

    for i in range(len(features)):
        pos = int(func(*features[i]))
        feature_vector[pos] += 1

    return feature_vector


def generate_all_feature_vectors(individual, toolbox, features, targets):
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

    for i in range(len(targets)):
        # get specific class from target name
        target = re.sub(r'[0-9]+', '', targets[i])
        # get current image's features from  list - TODO NOT HARDCODED
        current_features = features[(i*36) : (i+1)*36]
        # create feature vector from current images
        feature_vectors[i] = (target, generate_feature_vector(pow(2, 8), func, current_features))

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
        current_c = set[key][0]
        current_i = set[key][1]

        # compare to all images in set
        for key2 in set:
            compare_c = set[key2][0]
            compare_i = set[key2][1]

            # ensure not comparing object to self
            if key == key2:
                continue

            # calc distance
            dist = distance_vectors(current_i, compare_i)

            # add dist to respective count
            if current_c == compare_c:
                dist_within += dist
            else:
                dist_between += dist

    dist_within /= (total_inst * (total_inst - inst_per_class))
    dist_between /= (total_inst * (inst_per_class - 1))

    return dist_within, dist_between


def fitness_func(individual, toolbox, features, targets):
    """
    TODO: implement distance based fitness function (chi square only)

    """
    feature_vectors = generate_all_feature_vectors(individual, toolbox, features, targets)
    dist_within, dist_between = distance_between_and_within(feature_vectors)

    fit = 1 / (1 + pow(math.e, (-5 * (dist_within - dist_between))))
    return (1-fit), #TODO - check 1 minus ?


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


def create_toolbox(train_targets, train_features):
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
    pset.addPrimitive(code_node, ([float] * 8), int)

    # define terminal set
    for i in range(pow(2, 8)):
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

    toolbox.register("evaluate", fitness_func, toolbox=toolbox, features=train_features,targets=train_targets)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_ = 2, max_ = 10) #TODO check if these constraints are necessary
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset = pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    return toolbox


def evaluate(toolbox, train_features, train_targets, test_features, test_targets, filename, seed_val, start_time):
    """
        TODO - proper evaluation
    """
    pop = toolbox.population(n=100) #TODO: 500
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.2, 10, stats, halloffame=hof, verbose=True) #TODO: 30 gens

    output_file = open(filename, "w")

    output_file.write("".join(("Seed Value:", str(seed_val), "\n")))
    output_file.write("HOF: \n")
    output_file.write(str(hof[0]))

    output_file.write("\nTraining Accuracy:")
    output_file.write(str(fitness_func(hof[0], toolbox, train_features, train_targets)[0]))
    output_file.write("\nTest Accuracy:")
    output_file.write(str(fitness_func(hof[0], toolbox, test_features, test_targets)[0]))

    output_file.write("".join(("\nTime taken: ", "{:.2f}".format(time.time() - start_time), " seconds.")))
    output_file.write("\n--------------------------")

    output_file.close()


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

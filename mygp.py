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


def protectedDiv(left, right):
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


def convertToDecimal(binary_string):
    """
    Converts a binary string into its corresponding integer value.

    Args:
        binary_string (str): Binary representation of a number.

    Returns:
        Integer value of the binary string.
    """
    return int(binary_string, 2)


def generateFeatureVector(length, func, features):
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
        # print("feat vec", i, func(*features[i]), targets[i])
        pos = int(func(*features[i]))

        # print(str(func))

        feature_vector[pos] += 1

    return feature_vector


def generateAllFeatureVectors(individual, toolbox, features, targets):
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
        target = re.sub(r'[0-9]+', ' ', targets[i])

        if target in feature_vectors:
            # add to existing image
            vals = [feature_vectors[target]]
            vals.append(generateFeatureVector(pow(2, 8), func, features))
            feature_vectors[target] = vals
        else:
            #unique entry
            feature_vectors[target] = generateFeatureVector(pow(2, 8), func, features)

    return feature_vectors


def distanceVectors(u, v):
    """
    TODO - calc distance between any given two feature vectors
    """
    e = len(u)
    sum = 0

    for i in range(len(u)):
        if u[i] + v[i] != 0: #avoid divide by 0 errors
            # print(type(u[i]), u[i])
            sum += (pow((u[i]-v[i]), 2) / (u[i] + v[i]))

    sum /= 2
    return sum

def distanceBetweenAndWithin(set):
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

    for current_c in set:
        for current_i in set[current_c]:
            #compare with all in set
            for compare_c in set:
                for compare_i in set[compare_c]:
                    # calculate dist
                    # print(current_c, current_i, compare_c, compare_i)
                    dist = distanceVectors(current_i, compare_i)
                    # print(current_i, compare_i, dist)
                    # add dist to respective count
                    if current_c == compare_c:
                        dist_within += dist
                    else:
                        dist_between += dist

    dist_within /= (total_inst * (total_inst - inst_per_class))
    dist_between /= (total_inst * (inst_per_class - 1))

    # print("dw", dist_within, "db", dist_between)
    return dist_within, dist_between



def fitnessFunc(individual, toolbox, features, targets):
    """
    TODO: implement distance based fitness function (chi square only)

    """

    feature_vectors = generateAllFeatureVectors(individual, toolbox, features, targets)
    # print(feature_vectors)
    # print("very fit:", func(*features[0]))
    dist_within, dist_between = distanceBetweenAndWithin(feature_vectors)

    fit = 1 / (1 + pow(math.e, (-5 * (dist_within - dist_between))))
    # print("fit", fit, "dw", dist_within, "db", dist_between)
    return fit,


    # # essentially random feature count
    # func = toolbox.compile(expr=individual)
    # sum = 0
    # for i in range(len(features)):
    #     if func(*features[i]) == targets[int(i/36)]: #yeah yeah
    #         sum += 1
    #
    # # print(sum/len(targets))
    # return sum/len(targets),


def codeNode(*args):
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
    for v in args:
        if float(v) < 0.0: #/356 as a normalization attempt
            binary_string = "".join((binary_string, "0"))
        else:
            binary_string = "".join((binary_string, "1"))

    d = convertToDecimal(binary_string)
    # print(type(d), d)
    return float(d)


def createToolbox(train_targets, train_features):
    """
    Create a toolbox for evolving and evaluating the GP tree.

    Args:
        train_targets (`list` of str):  list of features of all images in the set.
        train_features (`list` of `list` of int): list of all images target class.

    Returns:
        Toolbox object for evolving and evaluating the GP tree.
    """
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(int, len(train_features[0])), float, prefix='F')

    # define primitive set
    pset.addPrimitive(operator.add, [int, int], int, name="ADD")
    pset.addPrimitive(operator.sub, [int, int], int, name="SUB")
    pset.addPrimitive(operator.mul, [int, int], int, name="MULT")
    pset.addPrimitive(protectedDiv, [int, int], int, name="PDIV")

    # pset.addPrimitive(operator.add, [int, int], float, name="FLOAT")
    pset.addPrimitive(codeNode, ([int] * 8), float)

    # define terminal set (TODO: hard coded to 2^3 as 3 for codenode)
    for i in range(pow(2, 8)):
        pset.addTerminal(i, float)

    # pset.addTerminal("jute", str)
    # pset.addTerminal("maize", str)
    # pset.addTerminal("rice", str)
    # pset.addTerminal("sugarcane", str)
    # pset.addTerminal("wheat", str)

    # creates fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) #TODO what min/max are?
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

     # TODO: fitnessFunction method
    toolbox.register("evaluate", fitnessFunc, toolbox=toolbox, features=train_features,targets=train_targets) # todo: train target?
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint) #TODO verify this is best
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_ = 0, max_ = 2) #TODO half and half?
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset = pset) #TODO ????

    # set max height (TODO: verify min height)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    return toolbox


def evaluate(toolbox, train_features, train_targets, test_features, test_targets):
    """
        TODO - proper evaluation
    """
    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.15, 20, stats, halloffame=hof, verbose=False)

    print("HOF:", hof[0])
    # print(train_targets)
    print("Training Accuracy:", fitnessFunc(hof[0], toolbox, train_features, train_targets)[0], "%.")
    # print("Test Accuracy:", fitnessFunc(hof[0], toolbox, test_features, test_targets)[0], "%.")


def removeDuplicates(list):
    """
    Removes the duplicates of a list.

    Params:
        list (`list` of obj): Any list.

    Returns:
        A list with no duplicate values.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in list if not (x in seen or seen_add(x))]


def readCSV(file_name):
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




if __name__ == "__main__":

    # read into features and targets
    train_targets, train_features = readCSV("test_data/5x5testUnique.csv")


    # convert from str to int
    train_features = [[int(i) for i in j] for j in train_features]
    #remove duplicates
    train_targets = removeDuplicates(train_targets)
    # print(train_targets)

    # test
    test_targets, test_features = readCSV("test_data/5x5testUnseen.csv")
    test_features = [[int(i) for i in j] for j in test_features]
    test_targets = removeDuplicates(test_targets)
    # print(train_targets)

    # hardcoded lol
    img_dimensions = (10,10)

    start_time = time.time()

    # bittaGP(train_targets, train_features, img_dimensions)
    toolbox = createToolbox(train_targets, train_features)
    # print(train_features)
    evaluate(toolbox, train_features, train_targets, test_features, test_targets)

    print("Time taken: ", "{:.2f}".format(time.time() - start_time), " seconds.", sep='')

    time.sleep(3)

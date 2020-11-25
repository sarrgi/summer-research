# todo - seperate gp over here
import csv
import itertools
import operator
import numpy
import random
import time
import re

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Global variable as thi needs to be accessed by codeNode, (can't create inside (cost), or pass throug (deap))
def protectedDiv(left, right):
    """
    Performs regular division, if divide by 0 issues then returns 0
    (Using 0 (and not 1) as this is specified by GP-criptor paper.
    """
    try:
        return left / right
    except:
        return 0


def convertToDecimal(binary_string):
    """
    Converts a binary string in to corresponding integer value.
    """
    return int(binary_string, 2)



def fitnessFunc(individual, toolbox, features, targets):
    """
    - should find the min and max distance between it and other classes

    """


    # print("FEATURING:", features)

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    feature_vectors = {}
    for i in range(len(targets)):
        # get specific class from target name
        target = re.sub(r'[0-9]+', ' ', targets[i])

        if target in feature_vectors:
            # add to existing image
            vals = [feature_vectors[target]]
            vals.append(generateFeatureVector(8, func, features, targets))
            feature_vectors[target] = vals
        else:
            #unique entry
            feature_vectors[target] = generateFeatureVector(8, func, features, targets)

    print(feature_vectors)
    # print("very fit:", func(*features[0]))





    # essentially random feature count
    sum = 0
    for i in range(len(features)):
        if func(*features[i]) == targets[int(i/36)]: #yeah yeah
            sum += 1

    # print(sum/len(targets))
    return sum/len(targets),



def generateFeatureVector(length, func, features, targets):
    """
    TODO
        - implement feature vec for fitness measure
    """
    feature_vector = [0] * length

    for i in range(len(features)):
        # print("feat vec", i, func(*features[i]), targets[i])
        pos = int(func(*features[i]))
        feature_vector[pos] += 1

    return feature_vector

def codeNode(*args):
    """
    NOTE
        - children appear to be automatically evaluated (can be floats or ints)
        - currently returns binary value (as a float to distinguis this as root node)
    """

    # calculate binary string based on child nodes
    binary_string = ""
    for v in args:
        if float(v) < 0.0:
            binary_string = "".join((binary_string, "0"))
        else:
            binary_string = "".join((binary_string, "1"))

    d = convertToDecimal(binary_string)
    # print(type(d), d)
    return float(d)


    # fun evaluating method:
    # r = random.randint(1,5)
    # if r == 1: return "jute"
    # elif r == 2: return "maize"
    # elif r == 3: return "rice"
    # elif r == 4: return "sugarcane"
    # elif r == 5: return "wheat"


def createToolbox(train_targets, train_features):
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(int, len(train_features[0])), float, prefix='F')

    # define primitive set
    pset.addPrimitive(operator.add, [int, int], int, name="ADD")
    pset.addPrimitive(operator.sub, [int, int], int, name="SUB")
    pset.addPrimitive(operator.mul, [int, int], int, name="MULT")
    pset.addPrimitive(protectedDiv, [int, int], int, name="PDIV")

    # pset.addPrimitive(operator.add, [int, int], float, name="FLOAT")
    pset.addPrimitive(codeNode, ([int] * 3), float)

    # define terminal set (TODO: hard coded to 2^3 as 3 for codenode)
    for i in range(8):
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
    toolbox.register("expr_mut", gp.genFull, min_ = 0, max_ = 2) #TODO half and half?
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset = pset) #TODO ????

    # set max height (TODO: verify min height)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    return toolbox


def evaluate(toolbox, train_features, train_targets, test_features, test_targets):
    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.15, 20, stats, halloffame=hof, verbose=False)

    print("HOF:", hof[0])
    # print("Training Accuracy:", fitnessFunc(hof[0], toolbox, train_features, train_targets)[0], "%.")
    # print("Test Accuracy:", fitnessFunc(hof[0], toolbox, test_features, test_targets)[0], "%.")


def removeDuplicates(list):
    seen = set()
    seen_add = seen.add
    return [x for x in list if not (x in seen or seen_add(x))]


def readCSV(file_name):
    """
    Read the csv and split into target and feature lists.
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
    # TODO: split into train and test

    # convert from str to int
    train_features = [[int(i) for i in j] for j in train_features]
    #remove duplicates
    train_targets = removeDuplicates(train_targets)
    print(train_targets)

    # hardcoded lol
    img_dimensions = (10,10)

    start_time = time.time()

    # bittaGP(train_targets, train_features, img_dimensions)
    toolbox = createToolbox(train_targets, train_features)
    # print(train_features)
    evaluate(toolbox, train_targets, train_features, train_targets, train_features)

    print("Time taken: ", "{:.2f}".format(time.time() - start_time), " seconds.", sep='')

    time.sleep(3)

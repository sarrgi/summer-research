# todo - seperate gp over here
import csv
import itertools
import operator
import numpy
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


def protectedDiv(left, right):
    """
    Performs regular division, if divide by 0 issues then returns 0
    (Using 0 (and not 1) as this is specified by GP-criptor paper.
    """
    try:
        return left / right
    except:
        return 0


def fitnessFunc(individual, toolbox, features, targets):
    width, height = img_dimensions
    # print("FEATURING:", features)

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    # print(type(features[0][0]), features[0][0])
    print("very fit:", func(*features[0]))

    return 1,

    # sum = 0
    # for i in range(len(features)):
    #     # print(len(features))
    #     print(func(*features[:25]))
    #     if True:
    #     # if func(*features[:25]) == True: #targets[i]:
    #         sum += 1
    #
    # return sum/len(targets),


def codeNode(a, b, c):
    """
    TODO
    """
    if random.randint(0,1) > 0.5:
        return True
    else:
        return False


def bittaGP(train_targets, train_features, img_dimensions):
    img_w, img_h = img_dimensions
    # print(len(train_features[0]))

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(int, len(train_features[0])), float, prefix='F')

    pset.addPrimitive(operator.add, [int, int], int, name="ADD")
    pset.addPrimitive(operator.sub, [int, int], int, name="MEATBALL_SUB")
    pset.addPrimitive(operator.mul, [int, int], int, name="MULTI")
    pset.addPrimitive(protectedDiv, [int, int], int, name="PROT")

    pset.addPrimitive(operator.add, [int, int], float, name="FLOAT")

    # pset.addPrimitive(operator.and_, [bool, bool], bool)
    # pset.addPrimitive(operator.or_, [bool, bool], bool)
    # pset.addPrimitive(operator.not_, [bool], bool)
    # pset.addPrimitive(codeNode, [int, int, int], bool)

    # pset.addTerminal(True, bool)
    # pset.addTerminal(False, bool)
    # pset.addTerminal(0.0, float)
    # pset.addTerminal(1.0, float)
    # pset.addTerminal(2.0, float)
    # pset.addTerminal(3.0, float)
    # pset.addTerminal("0.0", str)
    # pset.addTerminal("1.0", str)
    # pset.addTerminal("2.0", str)
    # pset.addTerminal("3.0", str)
    for t in tr


    # # define CODENODE as a set wich returns a type that pset cannot, therefore it shall be root (provided tree must return that type)
    # # pset.addPrimitive(codeNode, [c_pset], string) #TODO - check this compiles, also check it isnt randomly insterted into pset
    #
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





    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.15, 20, stats, halloffame=hof, verbose=False)

    print("HOF:", hof[0])

    return -1


def readCSV(fileName):
    """
    Read the csv and split into target and feature lists.
    """
    train_targets = []
    train_features = []

    with open('test_data/5x5.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        for row in reader:
            target = row[0]
            features = row[1:]

            train_targets.append(target)
            train_features.append(features)

    return train_targets, train_features




if __name__ == "__main__":

    # read into features and targets
    train_targets, train_features = readCSV("test_data/5x5.csv")

    # convert from str to int
    converted = [[int(i) for i in j] for j in train_features]

    # hardcoded lol
    img_dimensions = (10,10)


    # bittaGP(train_targets, train_features, img_dimensions)
    bittaGP(train_targets, converted, img_dimensions)

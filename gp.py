# todo - seperate gp over here
import csv

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
    # print(train_features)
    # print(train_targets)


    bittaGP(train_target, train_features)

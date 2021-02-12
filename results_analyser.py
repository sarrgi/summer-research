import re

def extractAcc(str):
    r = re.search('[0-9]+\.[0-9]+', str).group(0)
    r = round(float(r), 4)
    return r


def average(l):
    return sum(l)/len(l)

if __name__ == "__main__":

    # read in file
    file = open("res.txt", "r")
    lines = file.readlines()

    # store accuarcies
    rf = []
    dt = []
    knn1 = []
    knn3 = []

    next_line = ""

    for line in lines:

        # para print outs
        if "Running with k:" in line:
            print(line)
        elif "params" in line:
            print(line)
        elif "Dataset" in line:
            print(line)



        # store accuracy
        if next_line == "rf":
            val = extractAcc(line)
            rf.append(val)
            next_line = ""
        elif next_line == "dt":
            val = extractAcc(line)
            dt.append(val)
            next_line = ""
        elif next_line == "knn1":
            val = extractAcc(line)
            knn1.append(val)
            next_line = ""
        elif next_line == "knn3":
            val = extractAcc(line)
            knn3.append(val)
            next_line = ""

        # detect next line
        if "Random forest" in line:
            next_line = "rf"
        elif "Decision Tree" in line:
            next_line = "dt"
        elif "Knn (1) sk" in line:
            next_line = "knn1"
        elif "Knn (3)" in line:
            next_line = "knn3"

    rf = rf[:-1]
    dt = dt[:-1]
    knn1 = knn1[:-1]
    knn3 = knn3[:-1]

    print("Random Forest:", round(average(rf), 4))
    print("Decision Tree:", round(average(dt), 4))
    print("Knn (1)      :", round(average(knn1), 4))
    print("Knn (3)      :", round(average(knn3), 4))




    # print(":)")

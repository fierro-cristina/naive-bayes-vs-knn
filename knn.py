import csv
import random
import math
import operator
import time

start = time.time()

def loadCSV(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def SplitDataset(dataset, SplitRatio):
    train_size = int(len(dataset) * SplitRatio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]

def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        euc_dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], euc_dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sorted_votes[0][0]

def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set)))*100.0

def dec_format(t):
    return ('%.2f' % t).rstrip('0').rstrip('.')

def main():
    filename = 'diabetes.csv'
    split_ratio = 0.67
    dataset = loadCSV(filename)
    training_set, test_set = SplitDataset(dataset, split_ratio)
    print('Split {0} rows into train = {1} and test = {2} rows.').format(len(dataset), len(training_set), len(test_set))
    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        #print('predicted = ' + repr(result) + ', actual = ' + repr(test_set[x][-1]))
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%').format(accuracy)

main()

end = time.time()
t = end - start
print "Time of execution:", dec_format(t), "miliseconds."

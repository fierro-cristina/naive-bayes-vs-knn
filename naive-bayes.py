import csv
import random
import math
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

def separate_by_class(dataset):
    separate = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separate):
            separate[vector[-1]] = []
        separate[vector[-1]].append(vector)
    return separate

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = input_vector[i]
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for classValue, probability in probabilities.iteritems():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = classValue
    return best_label

def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions

def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
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
    summaries = summarize_by_class(training_set)
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%').format(accuracy)

main()

end = time.time()
t = end - start
print "Time of execution:", dec_format(t), "miliseconds."

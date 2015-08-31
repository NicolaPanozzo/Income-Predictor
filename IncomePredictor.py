import urllib.request
import csv

#the following code retrieve the file from the web and split each line using the csv.reader function

URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
local_file, headers = urllib.request.urlretrieve(URL)  # this is a tuple

fh = open(local_file, "r")
reader = csv.reader(fh)  # each line of the file 'fh' is formatted as a list

#the following code creates a new list of lists (raw_data_set) containing all the rows of the original file

raw_data_set = []

for row in reader:
        raw_data_set.append(row)

fh.close()

# the following function calculates the mid-point for an attribute of a list formatted like raw_data_set
# assuming the data can be converted into numbers. This will be used later on to create the 'classifier'


def midpoint(attribute, data_set):

    total_pos = 0
    total_neg = 0
    count_pos = 0
    count_neg = 0

    for row_1 in data_set:
        try:
            if row_1[14] == " >50K":
                pos = float(row_1[attribute])
                total_pos += pos
                count_pos += 1
            else:
                neg = float(row_1[attribute])
                total_neg += neg
                count_neg += 1
        except IndexError as e1:
            print(e1)
            continue

    mid_point = ((total_pos / count_pos) + (total_neg / count_neg)) / 2
    return mid_point

# the following function calculate the number of occurrances of a word by using a dictionary


def add_word(word, word_count_dict):
    if word in word_count_dict:
        word_count_dict[word] += 1
    else:
        word_count_dict[word] = 1

# the following function takes a set of data formatted like raw_data_set, an attribute of the set and
# 2 empty dictionaries and assigns positive weights to one dictionary
# and negative weights to the other dictionary


def weight(data_set, discrete_attribute, pos_weight, neg_weight):
    count_pos = 0
    count_neg = 0

    for row_2 in data_set:
        try:
            if row_2[14] == " >50K":
                add_word(row_2[discrete_attribute], pos_weight)
                count_pos += 1
            else:
                add_word(row_2[discrete_attribute], neg_weight)
                count_neg += 1
        except IndexError as e2:
            print(e2)
            continue

    for key in pos_weight:
        pos_weight[key] = pos_weight[key] / count_pos
    for key in neg_weight:
        neg_weight[key] = neg_weight[key] / count_neg

#the following loop replaces non-numerical attributes with their positive or negative weights in the raw_data_set

discrete_list = [1, 5, 6, 7, 8, 9]  # this is the list of non-numerical attributes

output_g = open("Weights.txt", "w")

for element in discrete_list:

    try:
        p_weight = {}
        n_weight = {}
        weight(raw_data_set, element, p_weight, n_weight)  # this line uses the function 'weight' defined earlier
                                                            # to create the positive and negative weights
                                                            # for each non-numerical attribute
        print("Positive weights:", p_weight, "\n\nNegative weights:", n_weight, "\n", file=output_g)
        for row_3 in raw_data_set:
            if row_3[14] == " >50K":
                row_3[element] = p_weight[row_3[element]]
            else:
                row_3[element] = n_weight[row_3[element]]
    except IndexError as e3:
        print(e3)
        continue

output_g.close()

#the following code creates the training set of data and test set of data by slicing the raw_data_set
#(a 75% split has been used in the example below)

training_set = raw_data_set[0:(int(len(raw_data_set)*0.75))]
test_set = raw_data_set[(int(len(raw_data_set)*0.75)):]

# the following code creates the 'classifier' as a list and populates it with the mid_points of each attribute
# applying the function mid_point to the training set. The attributes not required are ignored i.e. 2 (fnlwgt),
# 3 (Education) and 13 (Native Country)

range_1 = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
classifier = [midpoint(iteration, training_set) for iteration in range_1]
print(classifier)

#the following code iterate the records in the test set, check the numbers of positive and negative results
#for each attribute by comparing each record with the classifier, predicts if the record is >50K or not
# and tracks the result

correct_counter = 0
wrong_counter = 0
row_counter = 0

output_f = open("Wrong predictions report.txt", "w")

for index1, row in enumerate(test_set):
    try:
        pos_counter = 0
        neg_counter = 0
        for index2, element in enumerate(row[0:2] + row[4:13]):  # each row is sliced to ignore attributes not required
                                                                # I used the function enumerate to be able to compare
                                                                # each element of the row with the corresponding element
                                                                # of the classifier
            if float(element) > classifier[index2]:
                pos_counter += 1
            else:
                neg_counter += 1
        if pos_counter > neg_counter:
            result = " >50K"
        else:
            result = " <=50K"
        if result == row[14]:
            correct_counter += 1
        else:
            wrong_counter += 1
            print("Record {}".format(index1), end="", file=output_f)
            print("\tID = {}".format(row[2]), end="", file=output_f)
            print("\tCorrect outcome = {:>6s}".format(row[-1]), end="", file=output_f)
            print("\tPrediction =", result, file=output_f)
        row_counter += 1
    except IndexError as e4:
        print(e4)
        continue

result_track = correct_counter / row_counter * 100

print("-----------------------------------------------------------")
print("The number of tested records is {}".format(row_counter))
print("The number of correct predictions is {}".format(correct_counter))
print("The number of wrong predictions is {}".format(wrong_counter))
print("The percentage of correct predictions is: {0:.2f} %".format(result_track))
print("-----------------------------------------------------------")
print("You can find a report of the wrong predictions in the file 'Wrong predictions report.txt'")
print("You can find the dictionaries containing the positive and negative weights in the file 'Weights.txt''")

output_f.close()
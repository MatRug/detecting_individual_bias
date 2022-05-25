from __future__ import division
import numpy as np
import tensorflow as tf
from anchor import utils
from anchor import anchor_tabular
import csv
from sklearn.model_selection import train_test_split
import os
import sklearn.ensemble

def classifier(input):
    output = []
    sol = model.predict(input)
    for pre in sol:
        if pre[0] > 0.5:
            output.append(0)    #opposite due to the order in class_name
        else:
            output.append(1)
    return np.array(output)

csv_open = open('train_case-1.csv')
csv_reader = csv.reader(csv_open, delimiter=',')

a = True
features = []
labels = []
for i in csv_reader:
    if a:
        a = False
        continue
    
    sup = [float(j) for j in i]
    features.append(sup[1:])
    if sup[0]:
        labels.append([1, 0])
    else:
        labels.append([0, 1])

features = np.array(features)
labels = np.array(labels)

trainFeatures, valFeatures, trainLabels, valLabels = train_test_split(features, labels, test_size=0.2)

csv_open = open('testset_no_double.csv')
csv_reader = csv.reader(csv_open, delimiter=',')

a = True
testFeatures = []
testLabels = []
for i in csv_reader:
    if a:
        a = False
        continue
    sup = [float(j) for j in i]
    testFeatures.append(sup[1:])
    if sup[0]:
        testLabels.append([1, 0])
    else:
        testLabels.append([0, 1])

testFeatures = np.array(testFeatures)
testLabels = np.array(testLabels)

features_names = ['Workclass', 'Education-num', 'Occupation', 'Relationship', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week']
class_names = ['>50K', '<=50K']

model = tf.keras.models.load_model('NN_4_256t/NN')

explainer = anchor_tabular.AnchorTabularExplainer(class_names, features_names, trainFeatures)
np.random.seed(1)


maleValues = []
femaleValues = []
totValues = []

for i in range(len(testFeatures)):
    test = testFeatures[i]

    exp = explainer.explain_instance(test, classifier, threshold=0.95)
    sup = []
    for info in exp.names():
        for name in features_names:
            if name in info:
                sup.append(name)
                break

    if test[features_names.index('Sex')] == 0:   #male 
        maleValues.append(sup)
    else:     #female 
        femaleValues.append(sup)

    totValues.append(sup)


with open('maleValues.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in maleValues:
        writer.writerow(i)

with open('femaleValues.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in femaleValues:
        writer.writerow(i)

with open('totValues.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in totValues:
        writer.writerow(i)
    



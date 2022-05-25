from __future__ import division
import numpy as np
import random
import tensorflow as tf
import lime
import lime.lime_tabular
import csv
from sklearn.model_selection import train_test_split
import os

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

csv_open = open('test_case-1.csv')
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

features_name = ['Workclass', 'Education-num', 'Occupation', 'Relationship', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week']

class_names = ['>50K', '<=50K']

model = tf.keras.models.load_model('NN_4_256t/NN')

explainer = lime.lime_tabular.LimeTabularExplainer(trainFeatures,feature_names = features_name,class_names= class_names)

samp = random.sample(range(len(testFeatures)), int(0.0015*len(testFeatures)))
sup = []
for i in samp:
    sup.append(testFeatures[i])
testFeatures = np.array(sup)
print(len(testFeatures))

var_info = {}
var_per_info = {}
for name in features_name:
    var_info[name] = []
    var_per_info[name] = []

for num_samp in range(5000, 105000, 5000):
    print(num_samp)
    
    var = {}
    var_per = {}
    for name in features_name:
        var[name] = []
        var_per[name] = []
    
    for test in testFeatures:
        weights = {}
        for name in features_name:
            weights[name] = []

        for _ in range(10):
            exp = explainer.explain_instance(test, model.predict, num_features=len(features_name), num_samples = num_samp)
            res = exp.as_list()

            for elem in res:
                for name in features_name:
                    if name in elem[0]:
                        weights[name].append(elem[1])
        
        for name in features_name:
            weights[name] = np.array(weights[name])
            var[name].append(np.var(weights[name]))
            var_per[name].append(np.std(weights[name])/abs(np.average(weights[name])))

    for name in features_name:
        var[name] = np.array(var[name])
        var_per[name] = np.array(var_per[name])
        var_info[name].append(np.average(var[name]))
        var_per_info[name].append(np.average(var_per[name]))

values = []
for i in range(len(var_info['Sex'])):
    sup = {}
    for n in features_name:
        sup[n] = var_info[n][i]
    values.append(sup)

with open('var.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=features_name)
    writer.writeheader()
    for data in values:
        writer.writerow(data) 

values = []
for i in range(len(var_per_info['Sex'])):
    sup = {}
    for n in features_name:
        sup[n] = var_per_info[n][i]
    values.append(sup)

with open('std_percentage.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=features_name)
    writer.writeheader()
    for data in values:
        writer.writerow(data) 

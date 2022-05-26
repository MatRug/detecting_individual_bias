from __future__ import division
import numpy as np
import tensorflow as tf
import shap
import csv
from sklearn.model_selection import train_test_split
import random

def classifier(input):
    sol = model.predict(input)
    output = []
    for i in sol:
        if i[0] > 0.5:          #opposite to have weights similar ro lime
            output.append([0])
        else:
            output.append([1])
    return np.array(output)

csv_open = open('train_case-1_no_double.csv')
csv_reader = csv.reader(csv_open, delimiter=',')

a = True
features = []
for i in csv_reader:
    if a:
        a = False
        continue
    
    sup = [float(j) for j in i]
    features.append(sup)

trainFeatures = np.array(features)

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

features_name = ['Workclass', 'Education-num', 'Occupation', 'Relationship', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week']
class_names = ['>50K', '<=50K']

model = tf.keras.models.load_model('NN_4_256t/NN')

explainer = shap.KernelExplainer(classifier, shap.kmeans(trainFeatures, 1000))

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

for num_samp in range(100, 2100, 100):
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
            shap_values = explainer.shap_values(test, nsamples=num_samp)

            for i in range(8):
                weights[features_name[i]].append(shap_values[0][i])
        
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






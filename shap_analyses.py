from __future__ import division
import numpy as np
import tensorflow as tf
import shap
import csv
from sklearn.model_selection import train_test_split
import os

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
doc_name = ['Workclass_weight',  
            'Education-num_weight',  
            'Occupation_weight', 
            'Relationship_weight', 
            'Sex_weight', 
            'Capital-gain_weight', 
            'Capital-loss_weight',
            'Hours-per-week_weight']
class_names = ['>50K', '<=50K']

model = tf.keras.models.load_model('NN_4_256t/NN')

explainer = shap.KernelExplainer(classifier, shap.kmeans(trainFeatures, 500))

maleValues = []
femaleValues = []
totValues = []
maleValuesNorm = []
femaleValuesNorm = []
totValuesNorm = []

for i in range(len(testFeatures)):
    test = testFeatures[i]
    
    shap_values = explainer.shap_values(test, nsamples=300)

    sup = {}
    supC = {}
    
    toNorm = 0
    for j in range(len(features_name)):
        sup[doc_name[j]] = shap_values[0][j]
        toNorm += abs(shap_values[0][j])

    supNorm = {}
    for val in sup.keys():
        supNorm[val] = abs(sup[val])/toNorm


    if test[features_name.index('Sex')] == 0:   #male 
        maleValues.append(sup)
        maleValuesNorm.append(supNorm)
    else:     #female 
        femaleValues.append(sup)
        femaleValuesNorm.append(supNorm)

    totValues.append(sup)
    totValuesNorm.append(supNorm)

os.mkdir('real_values_doc')

with open('real_values_doc/maleValues1.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in maleValues:
        writer.writerow(data)

with open('real_values_doc/femaleValues1.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in femaleValues:
        writer.writerow(data)

with open('real_values_doc/totValues1.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in totValues:
        writer.writerow(data)
    


os.mkdir('norm_values_doc')

with open('norm_values_doc/maleValues1.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in maleValuesNorm:
        writer.writerow(data)

with open('norm_values_doc/femaleValues1.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in femaleValuesNorm:
        writer.writerow(data)

with open('norm_values_doc/totValues1.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in totValuesNorm:
        writer.writerow(data)
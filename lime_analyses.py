from __future__ import division
import numpy as np
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
doc_name = ['Workclass_weight', 'Workclass_pos', 
            'Education-num_weight', 'Education-num_pos', 
            'Occupation_weight', 'Occupation_pos', 
            'Relationship_weight', 'Relationship_pos', 
            'Sex_weight', 'Sex_pos',
            'Capital-gain_weight', 'Capital-gain_pos',
            'Capital-loss_weight', 'Capital-loss_pos',
            'Hours-per-week_weight', 'Hours-per-week_pos',
            'Type_result', 'Type_change']
class_names = ['>50K', '<=50K']

model = tf.keras.models.load_model('NN_4_256t/NN')

explainer = lime.lime_tabular.LimeTabularExplainer(trainFeatures,feature_names = features_name,class_names= class_names)

maleValues = []
femaleValues = []
totValues = []
maleValuesNorm = []
femaleValuesNorm = []
totValuesNorm = []

maleChangeValues = []
femaleChangeValues = []
totChangeValues = []
maleChangeValuesNorm = []
femaleChangeValuesNorm = []
totChangeValuesNorm = []

for i in range(len(testFeatures)):
    test = testFeatures[i]

    exp = explainer.explain_instance(test, model.predict, num_features=len(features_name))
    res = exp.as_list()
    prediction = model.predict(np.array([test]))

    sup = {}
    supC = {}
    
    toNorm = 0
    for elem in res:
        toNorm += abs(elem[1])
        for name in features_name:
            if name in elem[0]:
                sup[name + '_weight'] = elem[1]
                sup[name + '_pos'] = res.index(elem)+1
                break

    if testLabels[i][0] > 0.5 and prediction[0][0] > 0.5:
        sup['Type_result'] = 0
        supC['Type_result'] = 0
    elif testLabels[i][0] <= 0.5 and prediction[0][0] <= 0.5:
        sup['Type_result'] = 1
        supC['Type_result'] = 1
    elif testLabels[i][0] <= 0.5 and prediction[0][0] > 0.5:
        sup['Type_result'] = 2
        supC['Type_result'] = 2
    else:
        sup['Type_result'] = 3
        supC['Type_result'] = 3

    if test[features_name.index('Sex')] == 0:   #male
        test[features_name.index('Sex')] = 1
    else:     #female
        test[features_name.index('Sex')] = 0

    changePre = model.predict(np.array([test]))
    exp = explainer.explain_instance(test, model.predict, num_features=len(features_name))
    changeRes = exp.as_list()
        
    if prediction[0][0] > 0.5 and changePre[0][0] <= 0.5:
        sup['Type_change'] = 2
        supC['Type_change'] = 2
    elif prediction[0][0] <= 0.5 and changePre[0][0] > 0.5:
        sup['Type_change'] = 1
        supC['Type_change'] = 1
    else:
        sup['Type_change'] = 0
        supC['Type_change'] = 0

    toNormC = 0
    for elem in changeRes:
        toNormC += abs(elem[1])
        for name in features_name:
            if name in elem[0]:
                supC[name + '_weight'] = elem[1]
                supC[name + '_pos'] = changeRes.index(elem)+1
                break

    supNorm = {}
    for val in sup.keys():
        if 'weight' in val:
            supNorm[val] = abs(sup[val])/toNorm
        else:
            supNorm[val] = sup[val]
    
    supNormC = {}
    for val in supC.keys():
        if 'weight' in val:
            supNormC[val] = abs(supC[val])/toNormC
            a += supNormC[val]
        else:
            supNormC[val] = supC[val]

    if test[features_name.index('Sex')] == 1:   #male change to female
        test[features_name.index('Sex')] = 0
        maleValues.append(sup)
        maleValuesNorm.append(supNorm)
        maleChangeValues.append(supC)
        maleChangeValuesNorm.append(supNormC)
    else:     #female change to male
        test[features_name.index('Sex')] = 1
        femaleValues.append(sup)
        femaleValuesNorm.append(supNorm)
        femaleChangeValues.append(supC)
        femaleChangeValuesNorm.append(supNormC)

    totValues.append(sup)
    totValuesNorm.append(supNorm)
    totChangeValues.append(supC)
    totChangeValuesNorm.append(supNormC)

os.mkdir('real_values_doc')

with open('real_values_doc/maleValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in maleValues:
        writer.writerow(data)

with open('real_values_doc/femaleValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in femaleValues:
        writer.writerow(data)

with open('real_values_doc/maleChangeValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in maleChangeValues:
        writer.writerow(data)

with open('real_values_doc/femaleChangeValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in femaleChangeValues:
        writer.writerow(data)

with open('real_values_doc/totValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in totValues:
        writer.writerow(data)

with open('real_values_doc/totChangeValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in totChangeValues:
        writer.writerow(data)
    


os.mkdir('norm_values_doc')

with open('norm_values_doc/maleValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in maleValuesNorm:
        writer.writerow(data)

with open('norm_values_doc/femaleValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in femaleValuesNorm:
        writer.writerow(data)

with open('norm_values_doc/maleChangeValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in maleChangeValuesNorm:
        writer.writerow(data)

with open('norm_values_doc/femaleChangeValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in femaleChangeValuesNorm:
        writer.writerow(data)

with open('norm_values_doc/totValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in totValuesNorm:
        writer.writerow(data)

with open('norm_values_doc/totChangeValues.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=doc_name)
    writer.writeheader()
    for data in totChangeValuesNorm:
        writer.writerow(data)
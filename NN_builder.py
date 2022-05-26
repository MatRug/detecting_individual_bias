import numpy as np
import tensorflow as tf
import lime
import lime.lime_tabular
import csv
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

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

epochs = 20
infoTrain = []
infoValidation = []
infoTest = []

nameNN = 'NN_3_100'

for i in range(10):

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('NN_{}.h5'.format(i), save_best_only=True)

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=100, input_shape=[8], activation='relu'),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax', name='last')
        ])

        model.compile(optimizer='adam',
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])

        history = model.fit(trainFeatures, trainLabels,
                            validation_data=(valFeatures, valLabels),
                            callbacks = [model_checkpoint],
                            epochs = epochs) 
        tf.saved_model.save(model, './' + nameNN + '_{}/NN'.format(i))

        graph = tf.compat.v1.get_default_graph().as_graph_def()
        output_node_names = ['last/Softmax']

        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph, output_node_names)
        with open('./' + nameNN + '_{}/modelForNetron.pb'.format(i), 'wb') as _f:
                _f.write(frozen_graph_def.SerializeToString())

        plt.close()
        plt.plot(range(1, epochs+1), history.history['accuracy'], 'r')
        plt.plot(range(1, epochs+1), history.history['val_accuracy'], 'k')
        plt.xlabel('EPOCHS')
        plt.ylabel('Accuracy')
        plt.legend(['train set', 'validation set'])
        plt.savefig('./' + nameNN + '_{}/accuracy.png'.format(i))

        plt.close()
        plt.plot(range(1, epochs+1), history.history['loss'], 'r')
        plt.plot(range(1, epochs+1), history.history['val_loss'], 'k')
        plt.xlabel('EPOCHS')
        plt.ylabel('Loss')
        plt.legend(['train set', 'validation set'])
        plt.savefig('./' + nameNN + '_{}/loss.png'.format(i))
        
        infoTrain.append(model.evaluate(trainFeatures, trainLabels))
        infoValidation.append(model.evaluate(valFeatures, valLabels))
        infoTest.append(model.evaluate(testFeatures, testLabels))
    sess.close()

trainAccTot = 0
valAccTot = 0
testAccTot = 0
trainLossTot = 0
valLossTot = 0
testLossTot = 0

documentation = open('documentation_' + nameNN + '.txt', 'w')
documentation.write('Documentation ' + nameNN + ':\n\n\n')
documentation.write('n.  \t  accuracy_trainData  \t  accuracy_valData  \t  accuracy_testData  \t  average_accuracy\n')
best = 0
bestAcc = 0
worst = 0
worstAcc = 1
toStd = []
for i in range(len(infoTest)):
    avv = (infoTrain[i][1] + infoValidation[i][1] + infoTest[i][1])/3
    documentation.write('{}:  \t  {}  \t  {}  \t  {}  \t  {}\n'.format(i, infoTrain[i][1], infoValidation[i][1], infoTest[i][1], avv))
    trainAccTot += infoTrain[i][1]
    valAccTot += infoValidation[i][1]
    testAccTot += infoTest[i][1]
    toStd.append(infoTest[i][1])
    if infoTest[i][1] > bestAcc:
        bestAcc = infoTest[i][1]
        best = i
    if infoTest[i][1] < worstAcc:
        worstAcc = infoTest[i][1]
        worst = i

documentation.write('\n\n')

documentation.write('n.  \t  loss_trainData  \t  loss_validationData  \t  loss_testData\n')
for i in range(len(infoTest)):
    documentation.write('{}:  \t  {}  \t  {}  \t  {}\n'.format(i, infoTrain[i][0], infoValidation[i][0], infoTest[i][0]))
    trainLossTot += infoTrain[i][0]
    valLossTot += infoValidation[i][0]
    testLossTot += infoTest[i][0]
documentation.write('\n\n')

documentation.write('Average accuracy on training data:   {}\n'.format(trainAccTot/len(infoTrain)))
documentation.write('Average accuracy on validation data: {}\n'.format(valAccTot/len(infoTrain)))
documentation.write('Average accuracy on test data:       {}\n'.format(testAccTot/len(infoTrain)))
documentation.write('Standard deviation on test data:     {}\n'.format(np.std(toStd)))
documentation.write('Average accuracy on all data:        {}\n'.format((trainAccTot + valAccTot + testAccTot)/(3*len(infoTrain))))
documentation.write('Best accuracy NN_{} with:            {}\n'.format(best, bestAcc))
documentation.write('Worst accuracy NN_{} with:           {}\n\n'.format(worst, worstAcc))

documentation.write('Average loss on training data:   {}\n'.format(trainLossTot/len(infoTrain)))
documentation.write('Average loss on validation data: {}\n'.format(valLossTot/len(infoTrain)))
documentation.write('Average loss on test data:       {}\n'.format(testLossTot/len(infoTrain)))
documentation.write('Average loss on all data:        {}\n'.format((trainLossTot + valLossTot + testLossTot)/(3*len(infoTrain))))

documentation.close()
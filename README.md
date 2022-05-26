# On Detecting Biased Predictions with Post-hoc Explanation Methods

The current repository contains the codes used for the study, on the detecting of biased prediction with post-hoc explanation methods.

### Preliminaries of the experiment

The code *NN_builder.py* is the code used to train the models on the datasets presented in section 3.1. The neural network setup and architectures are presented in section A.1, and the results on the accuracy of the models are shown in section B.1. The code creates a txt file, where it prints the average accuracy and loss of the training set, validation set, and testset.

The codes *lime_conv.py* and *shap_conv.py* are used to investigate how the number of samples influences LIME and SHAP output. The codes create a csv file with the coefficient of variation of each feature for the different number of samples.
The results of the converge analyses are shown in section A.1.

### Weights investigation

*lime_analyses.py*, *shap_analyses.py*, and *anchor_analyses.py* are the code used to get the results from the post-hoc explanation tools used for the investigation. The codes for LIME and SHAP create 3 csv files with the real weights and the ranked position (obtained through the normalized weights) for each feature of each instance only for *male*, only for *female*, and all instances together. The same 3 files are produced for normalized weights. Anchors code creates the 3 csv files with the *anchors* of each instance for *male*, *female*, and all of them together.
The results are shown in sections 4.2 and B.2.


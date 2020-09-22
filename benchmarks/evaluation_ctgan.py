import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from ctgan import CTGANSynthesizer
from utils import *

model = sys.argv[1]

if model != "ctgan":
    print("Model {} not supported".format(model))
    sys.exit()

df_occutherm_train = pd.read_pickle("data/occutherm/df_feature1_train.pkl")
df_cresh_train = pd.read_pickle("data/cresh/cresh_train.pkl")

df_occutherm_test = pd.read_pickle("data/occutherm/df_feature1_test.pkl")
df_cresh_test = pd.read_pickle("data/cresh/cresh_test.pkl")

datasets_train = {"occutherm": df_occutherm_train, "cresh": df_cresh_train}
datasets_test = {"occutherm": df_occutherm_test, "cresh": df_cresh_test}

test_size_percentage = 0.2 # for CV within train split

for df in datasets_train.keys():
    # load synthetic dataset
    df_synth_str = "data/cresh/df_synth_" + model + "_" + df + ".csv"
    df_synth = pd.read_csv(df_synth_str)
    print("Resampled ({}) synth dataset shape {}".format(model, Counter(np.array(df_synth.iloc[:, -1]))))

    # merge synthethic + real dataset
    df_real_synth = pd.concat([df_synth, datasets_train[df]])
    print("Resampled ({}) real + synth dataset shape {}".format(model, Counter(np.array(df_real_synth.iloc[:, -1]))))

    print(df)
    print(df_synth_str)
    
    ###################################
    # Accuracy of generated samples
    # use best models NB, KNN, SVM, RDF
    accgen_acc_test, accgen_acc_train, accgen_models = evaluation_accuracy(df_synth, df)
    print("Accuracy of generated samples of {} for {}: {}".format(model, df, accgen_acc_test))

    ###################################
    # Variability of generated samples
    variability = evaluation_variability(df_synth)
    print("Variability of generated samples of {} for {}: {}".format(model, df, variability))

    #################################################
    # Class diversity with respect to the training set
    diversity = evaluation_diversity(df_synth, datasets_train[df], baseline=False)
    print("Class diversity with respect to the training set of {} for {}: {}".format(model, df, diversity))

    #####################################
    # Quality on the final classification
    # use best models NB, KNN, SVM, RDF
    class_acc_test, class_acc_train, class_models = evaluation_classification(df_real_synth, datasets_test[df], rdf_depth=None,test_size_percentage=test_size_percentage)
    print("Quality on the final classification of {} for {}: {}".format(model, df, class_acc_test))
    print("Make sure to update the rdf_depth within this script based on the elbow plot you see")

    #####################################
    # Saving results
    # Format is folder/dataset_string/_<metric or model>_<test or train>_algorithm.pkl
    save_pickle(accgen_acc_test, "metrics/" + df + "_accgen_test_" + model + ".pkl")
    save_pickle(accgen_acc_train, "metrics/" + df + "_accgen_train_" + model + ".pkl")
    save_pickle(accgen_models, "models/" + df + "_accgen_models_" + model + ".pkl")

    save_pickle(variability, "metrics/" + df + "_variability_" + model + ".pkl")

    save_pickle(diversity, "metrics/" + df + "_diversity_" + model + ".pkl")

    save_pickle(class_acc_test, "metrics/" + df + "_classification_test_" + model + ".pkl")
    save_pickle(class_acc_train, "metrics/" + df + "_classification_train_" + model + ".pkl")
    save_pickle(class_models, "models/" + df + "_classification_model_" + model + ".pkl")


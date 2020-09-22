import sys
import json
import numpy as np
import pandas as pd
import torch
from statistics import mean 
from collections import Counter

from utils import *

# load config file from CLI

# only evaluate ashrae dataset
if sys.argv[1] == "-a":
    with open(str(sys.argv[2]), 'r') as f:
        config_ashrae = json.load(f)
    
    experiment_name = config_ashrae['name']
    
    # location of datasets and dictionaries containing them
    df_ashrae_train = pd.read_pickle("data/ashrae/ashrae_train_reduced.pkl")
    df_ashrae_test = pd.read_pickle("data/ashrae/ashrae_test_reduced.pkl")

    datasets_train = {"ashrae": df_ashrae_train}
    datasets_test = {"ashrae": df_ashrae_test}
    model_path = {"ashrae": config_ashrae['model_path']}

# only evaluate ashrae dataset
elif sys.argv[1] == "-o":
    with open(str(sys.argv[2]), 'r') as f:
        config_occutherm = json.load(f)
    
    experiment_name = config_occutherm['name']
    
    # location of datasets and dictionaries containing them
    df_occutherm_train = pd.read_pickle("data/occutherm/df_feature1_train_reduced.pkl")
    df_occutherm_test = pd.read_pickle("data/occutherm/df_feature1_test_reduced.pkl")

    datasets_train = {"occutherm": df_occutherm_train}
    datasets_test = {"occutherm": df_occutherm_test}
    model_path = {"occutherm": config_occutherm['model_path']}    


tree_depth = 10 # fixed depth

test_size_percentage = 0.2 # for CV within train split

num_trials = 30

def sample_comfortgan(df_name, comfortgan, column_names):
    
    # load require samples to balance and initial count
    if df_name == "occutherm":
        occutherm_req_one =  config_occutherm['occutherm_req_one']
        occutherm_req_zero =  config_occutherm['occutherm_req_zero']
        occutherm_req_minus_one =  config_occutherm['occutherm_req_minus_one']
        count_one = 0
        count_minus_one = 0
    elif df_name == "cresh":
        cresh_req_nine = config_cresh['cresh_req_nine']
        cresh_req_ten = config_cresh['cresh_req_ten']
        cresh_req_eleven = config_cresh['cresh_req_eleven']
        count_nine = 0
        count_eleven = 0
    elif df_name == "ashrae":
        # samples to generate per label
        ashrae_req_one = config_ashrae['ashrae_req_one'] 
        ashrae_req_zero = config_ashrae['ashrae_req_zero']
        ashrae_req_minus_one = config_ashrae['ashrae_req_minus_one']
        # count of generated samples per label
        count_one = 0
        count_minus_one = 0

    samples_count = 0
    finish_loop = False
    
    # samples to generated every loop
    if df_name == "ashrae":
        print_threshold = 10000
    else:
        print_threshold = 1000
    
    # initiliaze synthetic dataframe
    df_synth = pd.DataFrame(columns=column_names)
#     print("Features on {} dataset:".format(df_name))
#     print(column_names)

    while True:
        if finish_loop:
#             print("All samples generated!")
            samples_count += print_threshold
            break

        # generate `prit_threshold` samples
        curr_df = comfortgan.generate_batch(print_threshold)

        # iterate through the generated samples
        if df_name == "occutherm":
            for index, row in curr_df.iterrows():  
                if (row['Discrete Thermal Comfort_TA'] == 1.0) and (count_one != occutherm_req_one): 
                    df_synth = df_synth.append(row)
                    count_one += 1
                elif (row['Discrete Thermal Comfort_TA'] == -1.0) and (count_minus_one != occutherm_req_minus_one): 
                    df_synth = df_synth.append(row)
                    count_minus_one += 1
                elif (count_one == occutherm_req_one) and \
                    (count_minus_one == occutherm_req_minus_one):
                    finish_loop = True
                    break
        elif df_name == "cresh":
            for index, row in curr_df.iterrows():  
                if (row['thermal_cozie'] == 9.0) and (count_nine != cresh_req_nine): 
                    df_synth = df_synth.append(row)
                    count_nine += 1 
                elif (row['thermal_cozie'] == 11.0) and (count_eleven != cresh_req_eleven): 
                    df_synth = df_synth.append(row)
                    count_eleven += 1
                elif (count_nine == cresh_req_nine) and (count_eleven == cresh_req_eleven):
                    finish_loop = True
                    break
        elif df_name == "ashrae":
            for index, row in curr_df.iterrows(): 
                if (row['Thermal sensation rounded'] == 1.0) and (count_one != ashrae_req_one): 
                    df_synth = df_synth.append(row)
                    count_one += 1
                elif (row['Thermal sensation rounded'] == -1.0) and (count_minus_one != ashrae_req_minus_one): 
                    df_synth = df_synth.append(row)
                    count_minus_one += 1
                elif (count_one == ashrae_req_one) and \
                     (count_minus_one == ashrae_req_minus_one):
                    finish_loop = True
                    break

        samples_count += print_threshold
        #print("Generated {} samples".format(print_threshold))

    return df_synth
    

# evaluates both datasets, thus only requires either config file
for df in datasets_train.keys():
    print("################################################################################")
    print("# Evaluation of dataset {}".format(df))
    print("################################################################################")
    print("Features:")
    print(datasets_train[df].columns.values)
    
    variability_list = []
    diversity_list = []

    class_acc_test_list_0 = []
    class_acc_test_list_1 = []
    class_acc_test_list_2 = []
    class_acc_test_list_3 = []
    
    class_report_rdf_list = []

    for i in range(0, num_trials):
        ###################################
        # Sample synthethic dataset for 'df'
        
        # load model for the current dataset
        comfortgan = torch.load(model_path[df])
        
        # synthethic dataset
        df_synth = sample_comfortgan(df, comfortgan, datasets_train[df].columns.values)

        # merge synthethic + real dataset
        df_real_synth = pd.concat([df_synth, datasets_train[df]])
        
        ###################################
        # Variability of generated samples
        variability = evaluation_variability(df_synth)
        variability_list.append(variability)
    
        #################################################
        # Class diversity with respect to the training set
        diversity = evaluation_diversity(df_synth, datasets_train[df], baseline=False)
        diversity_list.append(diversity)

        #####################################
        # Quality on the final classification
        # use best models NB, KNN, SVM, RDF
        class_acc_test, class_acc_train, class_models, class_report_rdf = evaluation_classification(df_real_synth, datasets_test[df], rdf_depth=tree_depth, depth_file_name='default', test_size_percentage=test_size_percentage)
        class_acc_test_list_0.append(class_acc_test[0])
        class_acc_test_list_1.append(class_acc_test[1])
        class_acc_test_list_2.append(class_acc_test[2])
        class_acc_test_list_3.append(class_acc_test[3])
        class_report_rdf_list.append(class_report_rdf)

        ########################
        # end of for loop trials
        print("End of {} trial".format(i + 1))
        
    # get average of trials
    variability = mean(variability_list)
    diversity = mean(diversity_list)
    class_acc_test = [mean(class_acc_test_list_0), mean(class_acc_test_list_1), mean(class_acc_test_list_2), mean(class_acc_test_list_3)]
    
    #####################################
    # Saving results
    # Format is folder/<dataset_string>-<experiment_name>_<metric or model>_<test or train>_<model>.pkl
    save_pickle(variability, "metrics/" + df + "-" + "reduced-" + experiment_name + "_variability_comfortgan_trials.pkl")
    save_pickle(diversity, "metrics/" + df + "-" + "reduced-" + experiment_name + "_diversity_comfortgan_trials.pkl")
    save_pickle(class_acc_test, "metrics/" + df + "-" + "reduced-" + experiment_name + "_classification_test_comfortgan_trials.pkl")
    save_pickle(class_report_rdf_list, "label-metrics/" + df + "reduced-" + "_class_report_" + experiment_name + "_comfortgan_trials.pkl")
    
    print("################################################################################")
    print("# Metrics and models for dataset {} saved!".format(df))
    print("################################################################################")

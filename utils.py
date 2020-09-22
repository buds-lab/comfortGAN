# Others
import math
import time
import random
import pickle
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from scipy.spatial import distance

# Pytorch
from torch.utils.data import DataLoader

# Sklearn
from sklearn import tree
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances

def encode_categorical(column, gamma):
    """
    Encodig of a categorical column: One-hot-encoding plus uniform noise
    Args:
        column (numpy.array)
            1D dataframe containing labels
        gamma (int):
            Max value for uniform noise for categorical variables
    
    Return:
        ohe_noisy (numpy.array)
            1D containing transformed data
        
        ohe (OheHotEndocer object):
            scaler object for ohe-hot-encoding the categorical features
        list_label (list)
            Array/list with the sorted label values
    """
    list_label = np.unique(column)

    ohe = OneHotEncoder(sparse=False)
    one_cat = ohe.fit_transform(column)
    # adds uniform noise to the columns
    noise = np.random.uniform(0, gamma, one_cat.shape)
    ohe_noisy = (one_cat + noise) / np.sum(one_cat + noise, keepdims=True, axis=1)
    return ohe_noisy, ohe, list_label

def data_transform(dataframe, cat_cols=None, scaling='2minmax', gamma=0.3):
    """
    Prepares, shuffles, and arrange data into batches. Assumes label is the last column
    Categorical columns are encoded using a scikit-learn OneHotEncoder and added some uniform noise.
    Continous columns are scale using the 'scaling' argument
    
    Args:    
        dataframe (pandas.DataFrame):
            Source dataframe with both features and labels
        cat_cols (list):
            List of categorical features indices
        scaling (string):    
            Type of scaling, if can either be 'minmax', 'standard', or '2minmax'
        gamma (int):
            Max value for uniform noise for categorical variables
        
    Return:
        X_train (numpy.ndarray):
            Numpy array of features with categorical features already one-hot encoded with noise
        scaler (scaler object):
            Scaler object used for continous columns
        ohe_cat (OneHotEncoder object):
            OnehotEncoder object for categorical variables
        ohe_label (OneHotEncoder object):
            OnehotEncoder object for label column
        list_label (list):
            List of all labels values
        num_cont (int):
            number of continuous columns
    """
    
    columns = dataframe.columns.values
    dataframe_copy = dataframe.copy()
    
    # remove categorical columns and label
    for cat_col in cat_cols:
        dataframe_copy.drop(dataframe_copy.columns[cat_col], axis=1, inplace=True)
    dataframe_copy.drop(dataframe_copy.columns[-1], axis=1, inplace=True)
    num_cont = dataframe_copy.shape[1]
    
    # continous columns scaling
    X = np.array(dataframe_copy)
    minus_one_one = False
    if scaling == 'minmax':
        scaler = MinMaxScaler() # bounded in [0, 1]
    elif scaling == '2minmax':
        scaler = MinMaxScaler() # bounded in [0, 1] and later to [-1, 1]
        minus_one_one = True
    elif scaling == 'standard':
        scaler = StandardScaler() # zero mean unit variance
    else:
        print("Please type a valid supporterd scaling method: minmax or standard")
        exit()
    
    X_scaled = scaler.fit_transform(X)
    
    if minus_one_one:
        X_scaled = -1 + 2*X_scaled # bounded in [-1, 1]
    
    # if there are categorical columns, do encoding
    if cat_cols:
        cat_columns = dataframe.iloc[:, cat_cols].to_numpy().reshape(-1, 1)
        cat_encoded, ohe_cat, _ = encode_categorical(cat_columns, gamma)

    # label encoding
    labels = dataframe.iloc[:, -1].to_numpy().reshape(-1, 1)
    labels_encoded, ohe_label, list_label = encode_categorical(labels, gamma)
    
    # concatenate continuous + categorical + label
    if cat_cols: 
        X_encoded = np.concatenate((X_scaled, cat_encoded, labels_encoded), axis=1)
    else:
        X_encoded = np.concatenate((X_scaled, labels_encoded), axis=1)
        ohe_cat = None
        
    return X_encoded, scaler, ohe_cat, ohe_label, list_label, num_cont

DEFAULT_K = 10 # default number of folds

def train_test_split_holistics(dataframe, list_complete_participants, train_test_split=0.7, user_split=False):
    """
    Prepare a dataframe and split it into train_test_split. Return both sets.
    It's assumed the dataframe does have the participant_no feature
    """
    
    df = dataframe.copy()
    
    if user_split:
        random.seed(75)
        random.shuffle(list_complete_participants)
        random.seed(75)
        test_participants = random.sample(set(list_complete_participants), 
                                          int(round((1 - train_test_split) * len(list_complete_participants))))

        print("Num participants in test set: {}".format(len(test_participants)))

        # only pick the train_test_split% of the complete participants for testing
        df_test = df[df['Participant_No'].isin(test_participants)]

        print("Testing on participants:")
        print(df_test['Participant_No'].unique())

        # use the rest for training (the negate of above)
        df_train = df[~df['Participant_No'].isin(test_participants)]

    else:
        # shuffle
        df = df.sample(frac=1, random_state=100).reset_index(drop=True)

        # determine split
        idx_split = int(df.shape[0] * train_test_split)

        # split the dataframe
        df_train = df.iloc[:idx_split, :]
        df_test = df.iloc[idx_split:, :]

    # removing the participant number since it's a holistic model
    del df_test['Participant_No']
    del df_train['Participant_No']

    # shuffle 
    df_train = df_train.sample(frac=1, random_state=100).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=100).reset_index(drop=True)
    
    # create binary label versions of the sets
    df_train_binary = df_train.copy()
    df_test_binary = df_test.copy()
    df_train_binary['Discrete Thermal Comfort_TA'] = df_train['Discrete Thermal Comfort_TA'].map(lambda x: 1 if x != 0 else 0)
    df_test_binary['Discrete Thermal Comfort_TA'] = df_test['Discrete Thermal Comfort_TA'].map(lambda x: 1 if x != 0 else 0)

    return df_train, df_test, df_train_binary, df_test_binary

def choose_k(train_labels):
    """
    Determine number of folds
    """
    DEFAULT_K = 10
    
    class_counter = Counter(train_labels)
    num_least_common_class = min(class_counter.values())
    return min(num_least_common_class, DEFAULT_K)

def find_model_param(train_vectors, train_labels, trainclf, parameters, scorer, useSampleWeight=False, log=False):
    """
    Choose the best combination of parameters for a given model
    """
    
    k = choose_k(train_labels) # get number of folds

    stratifiedKFold = StratifiedKFold(n_splits = k)
    if useSampleWeight:
        n_samples = len(train_labels)
        n_classes = len(set(train_labels))
        classCounter = Counter(train_labels)
        sampleWeights = [n_samples / (n_classes * classCounter[label]) for label in train_labels]
        
        chosen_cv = stratifiedKFold
        
        gridSearch = GridSearchCV(trainclf, parameters, cv = chosen_cv, scoring = scorer, fit_params = {'sample_weight' : sampleWeights})
    else:
        chosen_cv = stratifiedKFold
        
        gridSearch = GridSearchCV(trainclf, parameters, cv = chosen_cv, scoring = scorer)
    
    gridSearch.fit(train_vectors, train_labels)
    
    if log:
        print("Number of folds: " + str(k))
        print("Best parameters set found on development set:")
        print(gridSearch.best_params_)
    
    return gridSearch.best_estimator_

def train_nb(dataframe, test_size_percentage=0.2, log=False):
    """
    Breakdown the dataframe into X and y arrays. Later split them in train and test set. Train the model with CV
    and report the accuracy
    """
    
    DEFAULT_K = 10
    
    # create design matrix X and target vector y
    X = np.array(dataframe.iloc[:, 0:dataframe.shape[1] - 1]) # minus 1 for the comfort label
    y = np.array(dataframe.iloc[:, -1])

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    # split into train and test

    # X_train = train + cv set
    # X_test = test set
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = test_size_percentage, random_state = 100, stratify = y)

    # instantiate learning model
    nb_classifier = GaussianNB()

    # k-fold cross validation
    scores = cross_val_score(nb_classifier, X_train, y_train, cv = DEFAULT_K, scoring = 'accuracy') # accuracy here is f1 micro
    
    # fitting the model
    nb_classifier.fit(X_train, y_train)

    # predict the response
    y_pred = nb_classifier.predict(X_test)

    # Metrics
    nb_acc = clf_metrics(y_test, y_pred, log)
    
    if log:
        print("Features: {}".format(dataframe.columns.values[:-1]))  # minus 1 for the comfort label    
        print("Expected accuracy (f1 micro) based on Cross-Validation: ", scores.mean())
        print(nb_classifier)    

    return nb_acc, nb_classifier

def train_knn(dataframe, test_size_percentage=0.2, tuning=False, log=False):
    """
    Breakdown the dataframe into X and y arrays. Later split them in train and test set. Train the model with CV
    and report the accuracy
    """
    
    # create design matrix X and target vector y
    X = np.array(dataframe.iloc[:, 0:dataframe.shape[1] - 1]) # minus 1 for the comfort label
    y = np.array(dataframe.iloc[:, -1])
    
    # split into train and test
    # X_train = train + cv set (train_vectors)
    # X_test = test set (test_vectors)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_percentage, random_state = 100, stratify = y)

    # from occutherm:
    # k-NN models had for FS1: brute-force search as algorithm, standard Euclidean distance as metric and K = 14; 
    # for FS2: K changed to 5; for 
    # FS3: K changed to 13; 
    # for FS4: K changed to 4; and 
    # for FS5 K changed to 15
    parameters = {'n_neighbors' : [4, 5, 13, 14, 15], # [3, 5, 7, 9, 10, 11, 12, 13, 14, 15], 
                  'weights' : ['uniform', 'distance'], 
                  'metric' : ['seuclidean'], 
                  'algorithm' : ['brute']}
    scorer = 'f1_micro'
    clf = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform', metric = 'seuclidean', algorithm = 'brute')
    
    if tuning:
        knn_classifier = find_model_param(X_train, y_train, clf, parameters, scorer)
    else:
        knn_classifier = clone(clf)

    # fitting the model
    knn_classifier.fit(X_train, y_train)

    # predict the response
    y_pred = knn_classifier.predict(X_test)

    # evaluate accuracyt
    knn_acc = clf_metrics(y_test, y_pred, log)

    if log:
        print("Features: {}".format(dataframe.columns.values[:-1]))  # minus 1 for the comfort label
        print(knn_classifier)
        
    return knn_acc, knn_classifier

def train_svm(dataframe, test_size_percentage=0.2, tuning=False, log=False):
    """
    Breakdown the dataframe into X and y arrays. Later split them in train and test set. Train the model with CV
    and report the accuracy
    """

    # create design matrix X and target vector y
    X = np.array(dataframe.iloc[:, 0:dataframe.shape[1] - 1]) # minus 1 for the comfort label
    y = np.array(dataframe.iloc[:, -1])

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    # split into train and test

    # X_train = train + cv set (train_vectors)
    # X_test = test set (test_vectors)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = test_size_percentage, random_state = 100, stratify = y)
    
    # from occutherm:
    # SVM models had for all first four FS: C = 1000, balanced class weight, gamma of 0.1, radial basis function kernel, and one-versus-all decision function shape, with the exception that C = 1 and gamma of 0.001 for FS5
    parameters = {'C' : [1, 1000], 
                   'kernel' : ['rbf'], 
                   'gamma' : [0.1, 0.01], 
                   'class_weight' : ['balanced']}
#     parameters = [{'C' : [1, 10, 100, 1000],
#                    'kernel' : ['linear'], 
#                    'class_weight' : ['balanced']},
#                   {'C' : [1, 10, 100, 1000], 
#                    'kernel' : ['rbf'], 
#                    'gamma' : [0.1, 0.01, 0.001, 0.0001], 
#                    'class_weight' : ['balanced']}]
    clf = SVC(C = 1, kernel = 'linear', class_weight = None, random_state = 100)
    scorer = 'f1_micro'
    
    if tuning:
        svm_classifier = find_model_param(X_train, y_train, clf, parameters, scorer)
    else:
        svm_classifier = clone(clf)

    # fitting the model
    svm_classifier.fit(X_train, y_train)

    # predict the response
    y_pred = svm_classifier.predict(X_test)

    # evaluate accuracy
    svm_acc = clf_metrics(y_test, y_pred, log)
    
    if log:
        print("Features: {}".format(dataframe.columns.values[:-1]))  # minus 1 for the comfort label
        print(svm_classifier)
            
    return svm_acc, svm_classifier

def train_rdf(dataframe, rdf_depth=None, depth_file_name='default', test_size_percentage=0.2, tuning=False, log=False):
    """
    Breakdown the dataframe into X and y arrays. Later split them in train and test set. Train the model with CV
    and then find the optimal tree depth and report the accuracy
    """
    
    # create design matrix X and target vector y
    X = np.array(dataframe.iloc[:, 0:dataframe.shape[1] - 1]) # minus 1 for the comfort label
    y = np.array(dataframe.iloc[:, -1])
    
    # split into train and test CV
    
    # X_train = train + cv set (train_vectors)
    # X_test = test set (test_vectors)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_percentage, random_state = 100, stratify = y)
    
    # from occutherm:
    # FS1: Balanced class weights, Gini Index criterion, 2 minimum sample split, 100 estimators
    # FS2: changed to 1000 estimators; 
    # FS3: changed to entropy criterion, and 100 estimators; 
    # FS4: changed to balanced subsamples, 100 estimators;
    # FS5: changed to 1000 estimators, Gini criterion
    
    parameters = {'n_estimators' : [100], #[10, 100, 1000],
                  'criterion' : ['gini'], # ['entropy', 'gini'],
                  'min_samples_split' : [2], # [2, 10, 20, 30], 
                  'class_weight' : ['balanced']} # ['balanced', 'balanced_subsample']}
    
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=2, class_weight='balanced') #random_state = 100)
    scorer = 'f1_micro'
    
    if tuning:
        rdf_classifier = find_model_param(X_train, y_train, clf, parameters, scorer)
    else:
        # RDF is fixed for all models, uncomment if a tuned model is needed
        rdf_classifier = clone(clf)
        
    if rdf_depth is None:
        # find optimal depth and generate model
        optimal_depth = optimal_tree_depth(rdf_classifier, X_train, y_train, depth_file_name)
        # generate the model with the selected paramters plus the optimal depth and do the model fitting
        rdf_optimal = rdf_classifier.set_params(max_depth = optimal_depth)
    else:
        # this statement will be executed when the user inputs a number as the depth based on elbow method plot
        rdf_optimal = rdf_classifier.set_params(max_depth = rdf_depth)
    
    # fitting the model
    rdf_optimal.fit(X_train, y_train)

    # predict the response
    y_pred = rdf_optimal.predict(X_test)

    # evaluate accuracy
    rdf_acc, _ = clf_metrics(y_test, y_pred, log)
    
    if log:
        print("Features: {}".format(dataframe.columns.values[:-1]))  # minus 1 for the comfort label
        print(rdf_optimal)
        
    return rdf_acc, rdf_optimal


def optimal_tree_depth(clf, train_vectors, train_labels, file_name):
    """
    Choose the optimal depth of a tree model 
    """
    
    DEFAULT_K = 10
    
    # generate a list of potential depths to calculate the optimal
    depths = list(range(1, 20))

    # empty list that will hold cv scores
    cv_scores = []

    print("Finding optimal tree depth")
    # find optimal tree depth    
    for d in depths:
        clf_depth = clf.set_params(max_depth = d) # use previous parameters while changing depth
        scores = cross_val_score(clf_depth, train_vectors, 
                                 train_labels, cv = choose_k(train_labels),
                                 scoring = 'accuracy') # accuracy here is f1 micro
        cv_scores.append(scores.mean())

    # changing to misclassification error and determining best depth
    MSE = [1 - x for x in cv_scores] # MSE = 1 - f1_micro
    optimal_depth = depths[MSE.index(min(MSE))]
    print("The optimal depth is: {}".format(optimal_depth))
    print("Expected accuracy (f1 micro) based on Cross-Validation: {}".format(cv_scores[depths.index(optimal_depth)]))
    
    # plot misclassification error vs depths
    fig = plt.figure(figsize=(12, 10))
    plt.plot(depths, MSE)
    plt.xlabel('Tree Depth', fontsize = 20)
    plt.ylabel('Misclassification Error', fontsize = 20)
    plt.legend(fontsize = 15)
    plt.savefig("depth_tree-" + file_name + ".png")
    # plt.show()

    return optimal_depth

def test_clf(df_test, clf_optimal, log=False):
    # last column is the thermal comfort label
    X_test = np.array(df_test.iloc[:, 0:df_test.shape[1] - 1])
    y_test = np.array(df_test.iloc[:,-1])
    
    #predict the response on test set
    y_pred = clf_optimal.predict(X_test)

    # get metrics
    acc, class_report = clf_metrics(y_test, y_pred, log)
    return acc, class_report, y_pred

def clf_metrics(test_labels, pred_labels, log=False):
    """
    Compute different validation metrics for a classification problem.
    Metrics:
    - micro and macro F1 score
    - Confusion Matrix
    - Classification Report
    """
    
    acc = accuracy_score(test_labels, pred_labels)
    class_report = classification_report(test_labels, pred_labels, output_dict=True, zero_division=0)
    if log:
        print("Accuracy (f1 micro) on test set: ", acc)
        print("F1 micro on test set: ", f1_score(test_labels, pred_labels, average = 'micro'))
        print("F1 macro on test set: ", f1_score(test_labels, pred_labels, average = 'macro'))
        print("Confusion Matrix: ")
        print(confusion_matrix(test_labels, pred_labels))
        print("Classification Metrics: ")
        print(classification_report(test_labels, pred_labels, zero_division=0))

    return acc, class_report

def evaluation_accuracy(df_synth, dataset_string="occutherm"):
    """
    Source:
    Mariani, G., Scheidegger, F., Istrate, R., Bekas, C., & Malossi, C. (2018). 
    BAGAN: Data Augmentation with Balancing GAN, 1–9. Retrieved from http://arxiv.org/abs/1803.09655
    
    To verify that the generated samples are representative of the 
    original dataset, we classify them by a model trained on the original 
    dataset and we verify if the predicted class (model output) match 
    the target ones (grount truth synthetic y).
    
    For ease of calculations, the number of samples per class on df_synth
    is determined by the highest numnber of instances among all classes
    on df_test

    Models: Naive-Bayes, K-Nearest Neighbours, Support Vector Machine, based on:
    Francis, J., Quintana, M., Frankenberg, N. Von, & Bergés, M. (2019). 
    OccuTherm : Occupant Thermal Comfort Inference using Body Shape Information. 
    In BuildSys ’19 Proceedings of the 6th ACM International Conference on Systems 
    for Energy-Efficient Built Environments]. New York, NY, USA. https://doi.org/10.1145/3360322.3360858
    """

    # load models trained on real data and their train accuracy
    nb_optimal = pickle.load(open( "models/" + dataset_string + "_nb_reall_full.pkl", "rb" ))
    acc_train_nb = pickle.load(open( "metrics/" + dataset_string + "_nb_reall_full_acc.pkl", "rb" ))
    knn_optimal = pickle.load(open( "models/" + dataset_string + "_knn_reall_full.pkl", "rb" ))
    acc_train_knn = pickle.load(open( "metrics/" + dataset_string + "_knn_reall_full_acc.pkl", "rb" ))
    svm_optimal = pickle.load(open( "models/" + dataset_string + "_svm_reall_full.pkl", "rb" ))
    acc_train_svm = pickle.load(open( "metrics/" + dataset_string + "_svm_reall_full_acc.pkl", "rb" ))
    rdf_optimal = pickle.load(open( "models/" + dataset_string + "_rdf_reall_full.pkl", "rb" ))
    acc_train_rdf = pickle.load(open( "metrics/" + dataset_string + "_rdf_reall_full_acc.pkl", "rb" ))
    
    # using lodead models, test on synthetic data
    acc_test_nb, _, _ = test_clf(df_synth, nb_optimal)
    acc_test_knn, _, _ = test_clf(df_synth, knn_optimal)
    acc_test_svm, _, _ = test_clf(df_synth, svm_optimal)
    acc_test_rdf, _, _ = test_clf(df_synth, rdf_optimal)
    
    return [acc_test_nb, acc_test_knn, acc_test_svm, acc_test_rdf], [acc_train_nb, acc_train_knn, acc_train_svm, acc_train_rdf], [nb_optimal, knn_optimal, svm_optimal, rdf_optimal]
 
def evaluation_variability(df, max_k=30):
    """
    Source: Mariani, G., Scheidegger, F., Istrate, R., Bekas, C., & Malossi, C. (2018). 
    BAGAN: Data Augmentation with Balancing GAN, 1–9. Retrieved from http://arxiv.org/abs/1803.09655
    
    For each class, randomly sample two instances and calculate the euclidean distance between them.
    Repeat the process k times and average the resullts across al k * c samples.
    The baseline value is determined by sampling from the original dataset. 
    The higher the value thebetter, and also the closer to the baseline. 
    """

    distances = []
    all_classes = df.iloc[:,-1].unique()
        
    # for each class sample 2 instances randomly for k times
    for c in all_classes:
        df_c = df[df.iloc[:, -1] == c]
        k = 0
#         print('Thermal Comfort: {}'.format(c))
        
        while k < max_k:
            rows = df_c.sample(2)
            euclidean_distance = distance.euclidean(rows.iloc[0, :].values, rows.iloc[1, :].values) # returns an array
            
            ######## DEBUG
#             print(rows.iloc[0, :])
#             print(rows.iloc[1, :])
#             print(euclidean_distance)
            ########

            # save value
            distances.append(euclidean_distance)
            k += 1

    avg_distances = statistics.mean(distances)
    return avg_distances
 
def evaluation_diversity(df_source, df_target, baseline=False, max_k=30):
    """
    Source: Mariani, G., Scheidegger, F., Istrate, R., Bekas, C., & Malossi, C. (2018). 
    BAGAN: Data Augmentation with Balancing GAN, 1–9. Retrieved from http://arxiv.org/abs/1803.09655
    
    For df_source randomly sample one instance and find the euclidean distance to the
    closest datapoint from df_target.
    Repeat the process k times and average the results.
    The reference value is determined by doing this with df_source and df_target being the original 
    train set.
    The closer these values are, the better: it means there is no overfitting.
    """
    
    k = 0
    min_distances = []
        
    while k < max_k:
        curr_row = df_source.sample()
        distances = euclidean_distances(curr_row, df_target) # returns an array

        if baseline:
            # when the source and target datasets are the same (baseline scenario)
            # curr_row is also in df_target, therefore there will be one diff 
            # that is 0: the distance to that same datapount (curr_row).
            # thus we look for the 2nd smallet distance
            min_dist = second_smallest(distances[0, :])
        
        else:
            min_dist = np.amin(distances[0, :])

        ######## DEBUG
#         pd.DataFrame(distances.T).to_csv("test-files/dist_diver.csv",  mode="a")
#         curr_row.to_csv("test-files/curr_row_diver.csv",  mode='a')
#         df_target.to_csv("test-files/df_target.csv")
#         print(distances)
#         print(min_dist)
#         print(second_min_dist)
#         return
        ########

        # save value
        min_distances.append(min_dist)
        k += 1

    avg_min_dist = statistics.mean(min_distances)
    return avg_min_dist

def second_smallest(numbers):
    """
    Find second smallest number on a list
    """
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2

def evaluation_classification(df_train, df_test, rdf_depth=None, depth_file_name='default', test_size_percentage=0.2):
    """
    Compute the accuracy (f1-micro score) for multiple classification models based on datasets with
    synthetic and real samples
    Baseline accuracy: classifier trained on imbalanced set
    Models: Naive-Bayes, K-Nearest Neighbours, Support Vector Machine (based on Occutherm)
    """

    # train models
    acc_train_nb, nb_optimal = train_nb(df_train, test_size_percentage)
    acc_train_knn, knn_optimal = train_knn(df_train, test_size_percentage)
    acc_train_svm, svm_optimal = train_svm(df_train, test_size_percentage)
    acc_train_rdf, rdf_optimal = train_rdf(df_train, rdf_depth, depth_file_name, test_size_percentage)
    
    # using the optimal model, test on test split
    acc_test_nb, _, _ = test_clf(df_test, nb_optimal)
    acc_test_knn, _, _ = test_clf(df_test, knn_optimal)
    acc_test_svm, _, _ = test_clf(df_test, svm_optimal)
    acc_test_rdf, class_report_rdf, _ = test_clf(df_test, rdf_optimal)    

    return [acc_test_nb, acc_test_knn, acc_test_svm, acc_test_rdf], [acc_train_nb, acc_train_knn, acc_train_svm, acc_train_rdf], [nb_optimal, knn_optimal, svm_optimal, rdf_optimal], class_report_rdf

def print_network(nn):
    num_params = 0
    for param in nn.parameters():
        num_params += param.numel()
    print(nn)
    print('Total number of parameters: %d' % num_params)
    return

def save_pickle(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, KFold
import os
import sys
import math
import scipy

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from hdc import hdc

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut

from data_preprocessing import experiments, process_daily_activities, foldLaundry, writeHello, openJar, screwLightbulb, combHair, tieShoes

def HDC_flow(feat_dir, sub, temporal_ngram, downsample_factor, feature_window, seed, split_seed, D, accSpan, accCIMLevels, accCIMSpan, accIMLevels):
    filename = feat_dir + 'daily_activities_' + 'S' + f'{sub:03}' + '_features.mat'
    data = sio.loadmat(filename)
    #%%time
    # metadata for the run

    rng = np.random.default_rng(seed)
    #print(rng)

    numEMGChannel = data['emgFeature_disc'].shape[1]
    numWin = data['trialLabel_disc'].shape[1]

    emgIM = hdc.gen_im(rng,D=D,N=numEMGChannel)

    accCIM = []
    accIM = []
    for ax in range(3):
        accCIM.append(hdc.gen_cim(rng,D=D,N=accCIMLevels[ax],span=accCIMSpan[ax]))
        accIM.append(hdc.gen_im(rng,D=D,N=accIMLevels[ax]))  
        
    #%%time
    # EMG spatial encoding
    print(f"shape of emgFeature: {np.shape(data['emgFeature_disc'])}")
    data_emg = data['emgFeature_disc']
    data_label = data['subGestureLabel_disc']
    data_acc = data['accFeature_disc']
    # Downsampling
    data['emgFeature_disc'] = data_emg[::downsample_factor]
    data['subGestureLabel_disc'] = data_label[:,::downsample_factor]
    data['accFeature_disc'] = data_acc[::downsample_factor]
    print(f"shape of emgFeature: {np.shape(data['emgFeature_disc'])}")
    print(f"shape of emgFeature: {np.shape(data['subGestureLabel_disc'])}")
   
    
    spatialHV = hdc.spatial_encode(rng,data['emgFeature_disc'],emgIM)

    #contexts = np.unique(data['positionLabel'][~np.isnan(data['positionLabel'])]).astype('int')
    #contexts = contexts[contexts >= 0]
    #contextIM = hdc.gen_im(rng,D=D,N=len(contexts))
    
    #%%time
    # EMG temporal encoding

    #original
    temporalHV = hdc.temporal_encode(spatialHV,N=temporal_ngram)
    #print(np.shape(temporalHV))
    
    #new
    #remove irrelevant data
    #temporal data goes back t-N samples to generate a vector corresponding to a label at time t
    holdIdx = np.where(data['subGestureLabel_disc'] >= 0)[1]
    yAll = data['subGestureLabel_disc'][:,holdIdx]

    #get all the labels
    all_labels = np.unique(yAll)

    #generate 5 vectors per subgesture label
    subgestureHV = np.zeros((all_labels.shape[0]*5,spatialHV.shape[1])).astype(np.int16)
    subgesture_labels = np.zeros((1,all_labels.shape[0]*5)).astype(np.int16)
    for i in range(all_labels.shape[0]):
        #for the given subgesture label, find all the data
        label_idx = np.where(data['subGestureLabel_disc'] == all_labels[i])[1]
        #start from N-1 to exclude the ngrams that include data from a prior subgesture or transition
        for x in range(temporal_ngram-1,label_idx.shape[0]):
            #print(label_idx.shape[0])
            index = math.floor(x/((5000/feature_window)/downsample_factor))
            if index < 5:
                subgestureHV[i*5 + index] = subgestureHV[i*5 + index] + temporalHV[label_idx[x]]
                subgesture_labels[0,i*5+index] = all_labels[i]
            #else:
                #print("there were more than 5 discrete trials for: ",all_labels[i])

    #rng = np.random.default_rng(seed)
    for i in range(subgestureHV.shape[0]):
        subgestureHV[i] = hdc.bipolarize(rng,subgestureHV[i])
    #dataset has 5 HVs per subgesture label
    #print(subgesture_labels)
    #print(subgestureHV[5])
    yAll = subgesture_labels
    XAll = subgestureHV
    
    #%%time
    # accelerometer IM encoding

    accIMHV = []
    for ax in range(3):
        accIdx = hdc.convert_to_idx(data['accFeature_disc'][:,ax], accSpan, accIMLevels[ax])
        accIMHV.append(accIM[ax][accIdx])

    #leave one out cross validation
    avg_activity_accuracy = 0 
    avg_activities_accuracies = np.zeros([1,6])
    n_splits = 5
    all_accuracies = np.zeros([1,n_splits])
    all_activity_accuracies = np.zeros([n_splits,6])
    skf = StratifiedKFold(n_splits=n_splits)
    yAll = np.transpose(yAll)
    split_count = 0

    for train_index, test_index in skf.split(XAll, yAll):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = XAll[train_index], XAll[test_index]
        y_train, y_test = yAll[train_index], yAll[test_index]
        #print(y_train.shape)
        #print(y_test.shape)
        #print(X_train.shape)
        #print(X_test.shape)
        y_train1 = list(y_train)
        y_test1 = list(y_test)
        acc,pred,AM=hdc.train_test_split_accuracy(seed,X_train, X_test, y_train1, y_test1 )
        correct = 0
        correct_within2 = 0
        correct_within3 = 0
        correct_within4 = 0
        correct_activity = 0
        activities_accuracies = np.zeros([1,6])
        activity_count = np.zeros([1,6])
        for i in range(len(pred)):
            if math.floor(pred[i]/100) == math.floor(y_test[i]/100):
                correct_activity += 1
                activities_accuracies[0,math.floor(y_test[i]/100)] += 1
            activity_count[0,math.floor(y_test[i]/100)] += 1
        activity_accuracy = correct_activity/len(pred)
        activities_accuracies = np.divide(activities_accuracies, activity_count)
        all_accuracies[0,split_count] = activity_accuracy
        all_activity_accuracies[split_count] = activities_accuracies
        split_count += 1
        avg_activities_accuracies = avg_activities_accuracies + activities_accuracies
        avg_activity_accuracy = avg_activity_accuracy + activity_accuracy
    
    avg_activities_accuracies = avg_activities_accuracies/n_splits
    avg_activity_accuracy = avg_activity_accuracy/n_splits
    print("average activity accuracy is:", avg_activity_accuracy)
    print("train-test split activity accuracy range is: ",all_accuracies)
    print("the accuracies per activity are: ", avg_activities_accuracies) 
    print("train-test split accuracies per activity range is: ", all_activity_accuracies)
    
    vectors = AM[1]
    cf_matrix = confusion_matrix(y_test, pred)
    hamming = np.zeros([len(vectors),len(vectors)])
    for i in range(len(vectors)):
        for y in range(len(vectors)):
            v1 = vectors[i]
            v2 = vectors[y]
            hamming[i,y] = scipy.spatial.distance.hamming(v1,v2)
    plt.figure(1)
    plt.rcParams["figure.figsize"] = (17,8.5)
    sns.heatmap(hamming)
    return avg_activity_accuracy
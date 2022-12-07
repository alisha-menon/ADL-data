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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

package_path,current = os.path.split(module_path)
if package_path not in sys.path:
    sys.path.append(package_path)
    
from data_preprocessing import experiments, process_daily_activities, foldLaundry, writeHello, openJar, screwLightbulb, combHair, tieShoes

from hdc import hdc




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
    print(f"shape of labels: {np.shape(data['subGestureLabel_disc'])}")
    data_acc = data['accFeature_disc']
    #remove irrelevant accelerometer data
    holdaccIdx = np.where(data['subGestureLabel_disc'] >= 0)[1]
    yaccAll = data['subGestureLabel_disc'][:,holdaccIdx]
    print(max(holdaccIdx))
    #print(data_acc[holdaccIdx].shape)

    #plt.plot(data_acc[holdaccIdx])
    
    ########## everything below commented for accelerometer data
    
#     # Downsampling
#     data['emgFeature_disc'] = data_emg[::downsample_factor]
#     data['subGestureLabel_disc'] = data_label[:,::downsample_factor]
#     data['accFeature_disc'] = data_acc[::downsample_factor]
#     print(f"shape of emgFeature: {np.shape(data['emgFeature_disc'])}")
#     print(f"shape of emgFeature: {np.shape(data['subGestureLabel_disc'])}")
   
    
#     spatialHV = hdc.spatial_encode(rng,data['emgFeature_disc'],emgIM)

#     #contexts = np.unique(data['positionLabel'][~np.isnan(data['positionLabel'])]).astype('int')
#     #contexts = contexts[contexts >= 0]
#     #contextIM = hdc.gen_im(rng,D=D,N=len(contexts))
    
#     #%%time
#     # EMG temporal encoding

#     #original
#     temporalHV = hdc.temporal_encode(spatialHV,N=temporal_ngram)
#     #print(np.shape(temporalHV))
    
#     #new
#     #remove irrelevant data
#     #temporal data goes back t-N samples to generate a vector corresponding to a label at time t
#     holdIdx = np.where(data['subGestureLabel_disc'] >= 0)[1]
#     yAll = data['subGestureLabel_disc'][:,holdIdx]

#     #get all the labels
#     all_labels = np.unique(yAll)

#     #generate 5 vectors per subgesture label
#     subgestureHV = np.zeros((all_labels.shape[0]*5,spatialHV.shape[1])).astype(np.int16)
#     subgesture_labels = np.zeros((1,all_labels.shape[0]*5)).astype(np.int16)
#     for i in range(all_labels.shape[0]):
#         #for the given subgesture label, find all the data
#         label_idx = np.where(data['subGestureLabel_disc'] == all_labels[i])[1]
#         #start from N-1 to exclude the ngrams that include data from a prior subgesture or transition
#         for x in range(temporal_ngram-1,label_idx.shape[0]):
#             #print(label_idx.shape[0])
#             index = math.floor(x/((5000/feature_window)/downsample_factor))
#             if index < 5:
#                 subgestureHV[i*5 + index] = subgestureHV[i*5 + index] + temporalHV[label_idx[x]]
#                 subgesture_labels[0,i*5+index] = all_labels[i]
#             #else:
#                 #print("there were more than 5 discrete trials for: ",all_labels[i])

#     #rng = np.random.default_rng(seed)
#     for i in range(subgestureHV.shape[0]):
#         subgestureHV[i] = hdc.bipolarize(rng,subgestureHV[i])
#     #dataset has 5 HVs per subgesture label
#     #print(subgesture_labels)
#     #print(subgestureHV[5])
#     yAll = subgesture_labels
#     XAll = subgestureHV
    
#     #%%time
#     # accelerometer IM encoding

#     accIMHV = []
#     for ax in range(3):
#         accIdx = hdc.convert_to_idx(data['accFeature_disc'][:,ax], accSpan, accIMLevels[ax])
#         accIMHV.append(accIM[ax][accIdx])

#     #leave one out cross validation
#     avg_accuracy = 0
#     avg_2_accuracy = 0
#     avg_3_accuracy = 0
#     avg_4_accuracy = 0
#     avg_activity_accuracy = 0 
#     avg_activities_accuracies = np.zeros([1,6])
#     n_splits = 5
#     all_accuracies = np.zeros([1,n_splits])
#     all_activity_accuracies = np.zeros([n_splits,6])
#     skf = StratifiedKFold(n_splits=n_splits)
#     yAll = np.transpose(yAll)
#     split_count = 0

#     for train_index, test_index in skf.split(XAll, yAll):
#         #print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = XAll[train_index], XAll[test_index]
#         y_train, y_test = yAll[train_index], yAll[test_index]
#         #print(y_train.shape)
#         #print(y_test.shape)
#         #print(X_train.shape)
#         #print(X_test.shape)
#         y_train1 = list(y_train)
#         y_test1 = list(y_test)
#         acc,pred,AM=hdc.train_test_split_accuracy(seed,X_train, X_test, y_train1, y_test1 )
#         correct = 0
#         correct_within2 = 0
#         correct_within3 = 0
#         correct_within4 = 0
#         correct_activity = 0
#         activities_accuracies = np.zeros([1,6])
#         activity_count = np.zeros([1,6])
#         for i in range(len(pred)):
#             if pred[i] == y_test[i]:
#                 correct += 1
#                 #print(i)
#                 #print("correctly guessed subgesture: ", y_test[i])
#             elif (y_test[i] >= (pred[i]-2)) and (y_test[i] <= (pred[i]+2)):
#                 correct_within2 += 1
#             elif (y_test[i] >= (pred[i]-3)) and (y_test[i] <= (pred[i]+3)):
#                 correct_within3 += 1
#             elif (y_test[i] >= (pred[i]-4)) and (y_test[i] <= (pred[i]+4)):
#                 correct_within4 += 1
#             if math.floor(pred[i]/100) == math.floor(y_test[i]/100):
#                 correct_activity += 1
#                 activities_accuracies[0,math.floor(y_test[i]/100)] += 1
#             activity_count[0,math.floor(y_test[i]/100)] += 1
#         accuracy = correct/len(pred)
#         print('Accuracy on fold:', accuracy)
#         accuracy_within2 = (correct+correct_within2)/len(pred)
#         accuracy_within3 = (correct+correct_within2+correct_within3)/len(pred)
#         accuracy_within4 = (correct+correct_within2+correct_within3+correct_within4)/len(pred)
#         activity_accuracy = correct_activity/len(pred)
#         activities_accuracies = np.divide(activities_accuracies, activity_count)
#         all_accuracies[0,split_count] = activity_accuracy
#         all_activity_accuracies[split_count] = activities_accuracies
#         split_count += 1
#         avg_accuracy = avg_accuracy + accuracy
#         avg_2_accuracy = avg_2_accuracy + accuracy_within2
#         avg_3_accuracy = avg_3_accuracy + accuracy_within3
#         avg_4_accuracy = avg_4_accuracy + accuracy_within4
#         avg_activity_accuracy = avg_activity_accuracy + activity_accuracy
#         avg_activities_accuracies = avg_activities_accuracies + activities_accuracies
    
#     avg_accuracy = avg_accuracy/n_splits
#     avg_2_accuracy = avg_2_accuracy/n_splits
#     avg_3_accuracy = avg_3_accuracy/n_splits
#     avg_4_accuracy = avg_4_accuracy/n_splits
#     avg_activity_accuracy = avg_activity_accuracy/n_splits
#     avg_activities_accuracies = avg_activities_accuracies/n_splits
#     print("")
#     print("average accuracy is: ",avg_accuracy)
#     print("accuracy within 2 subgestures is: ",avg_2_accuracy)
#     print("accuracy within 3 subgestures is: ",avg_3_accuracy)
#     print("accuracy within 4 subgestures is: ",avg_4_accuracy)
#     print("activity accuracy for the final train-test split is:", activity_accuracy)
#     print("cross validation activity accuracy is:", avg_activity_accuracy)
#     print("the range of activity accuracy over train-test splits is: ",all_accuracies)
#     print("the cross validation individual activity accuracies are: ", avg_activities_accuracies) 
#     print("the individual activity accuracies range over train-test splits is: ", all_activity_accuracies)
#     #for train, test in skf.split(XAll, yAll):
#     #     print('train -  {}   |   test -  {}'.format(np.bincount(yAll[train]), np.bincount(yAll[test])))
    
#     vectors = AM[1]
#     cf_matrix = confusion_matrix(y_test, pred)
#     #print(cf_matrix)
#     #plt.figure(1)
#     #plt.rcParams["figure.figsize"] = (20,10)
#     #ConfusionMatrixDisplay.from_predictions(y_test,pred,include_values=False)
#     #sns.heatmap(cf_matrix)
#     #print(len(vectors))
#     hamming = np.zeros([len(vectors),len(vectors)])

#     #print(len(vectors))
#     for i in range(len(vectors)):
#         for y in range(len(vectors)):
#             v1 = vectors[i]
#             v2 = vectors[y]
#             hamming[i,y] = scipy.spatial.distance.hamming(v1,v2)
#     #plt.figure(1)
#     #plt.rcParams["figure.figsize"] = (17,8.5)
#     #sns.heatmap(hamming)
    return yaccAll, data_acc,data_emg,holdaccIdx
import numpy as np
import sklearn.model_selection
import os
import sys
import matplotlib.pyplot as plt
import math
import tsfresh

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

package_path,current = os.path.split(module_path)
if package_path not in sys.path:
    sys.path.append(package_path)

import data_preprocessing as data_prep
from data_preprocessing import experiments, process_daily_activities, foldLaundry, writeHello, openJar, screwLightbulb, combHair, tieShoes
import full_accelerometer_classification as HDC_sb_flow

from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def load_accelerometer_data():

    module_path = os.path.abspath(os.path.join(''))
    if module_path not in sys.path:
        sys.path.append(module_path)

    package_path,current = os.path.split(module_path)
    if package_path not in sys.path:
        sys.path.append(package_path)

    feat_dir = os.getcwd() + '/../../S1/features/'
    sub = 1
    temporal_ngram = 2
    downsample_factor = 1
    feature_window = 50
    seed = 12345
    split_seed = 1
    D = 10000

    accSpan = [-1,1]
    accCIMLevels = [59, 55, 14]
    accCIMSpan = [10000, 5000, 6000]
    accIMLevels = [1, 4, 5]

    temporal_ngrams = [3]
    downsample_factors = [1]
    #feature_windows = [50,100,200,500,1000]
    feature_windows = [50]
    subtask_accuracies = np.zeros([len(temporal_ngrams),len(downsample_factors)])
    task_accuracies = np.zeros([len(temporal_ngrams),len(downsample_factors)])


    subtask_label,accelerometer_data,emg_data,data_during_subtask = HDC_sb_flow.HDC_flow(feat_dir, sub, temporal_ngram, downsample_factor, feature_window, seed, split_seed, D,accSpan,accCIMLevels,accCIMSpan,accIMLevels)
    accelerometer_data = accelerometer_data[data_during_subtask]

    return subtask_label, accelerometer_data

def timeseries_classifier(subtask_label, accelerometer_data,relabel=False,split_tasks=True,verbosity=True,extra_features=True,remove_reps=True,remove_similar=True,include_std=False,include_delta=False,include_full_features=True,feature_window=10):

    recorded_trials = 5  # Recorded sessions
    experiment_trials = 1  # Experiments over different k-fold validation schemes
    window_size = 100

    skf = StratifiedKFold(n_splits=recorded_trials)



    #Split dataset into tasks or classify all possible subtasks
    if not split_tasks: # all 75 subtasks classified at the same time
        task_len=1
        task_locations=[np.array(list(range(len(subtask_label))))]
        activity_names = ['All activities']
    else: #only use the subset of the dataset corresponding to current task
        task_len=len(data_prep.experiments)
        task_locations = [[]] * task_len
        task_locations[0] = np.where(subtask_label < 100)[1]
        task_locations[1] = np.where((subtask_label >= 100) & (subtask_label < 200))[1]
        task_locations[2] = np.where((subtask_label >= 200) & (subtask_label < 300))[1]
        task_locations[3] = np.where((subtask_label >= 300) & (subtask_label < 400))[1]
        task_locations[4] = np.where((subtask_label >= 400) & (subtask_label < 500))[1]
        task_locations[5] = np.where((subtask_label >= 500) & (subtask_label < 600))[1]

        a_labels=[[]]*6
        a_labels[0]=['ReachOut','Pinch','LiftUp','Move','PutDown','Fold','Release','Return']
        a_labels[1]=['Reach','Thumb','HandDown','Pinch','LiftPen','Rot-Wrist','Paper','PenDown','Hello','PenUp','PenBack','Rot-wrist','PenDown','Ret-Wrist','Ret-Neutral']
        a_labels[2]=['PlaceHand','PowerGrip','HandDown','Grip','Untwist','ReAlign','Untwist','ReAlign','Untwist','ReAlign','Untwist','ReAlign','PlaceCap','Neutral']
        a_labels[3]=['HandOver','HandDown','Place','Align','Twist','Align','Twist','Align','Twist','Align','Twist','Align','Neutral']
        a_labels[4]=['HandOver','HandOnto','Slide','LiftRight','GripRight','LeftDone','LiftComb','Comb','LiftComb','Comb','LiftComb','Comb','LiftComb','Comb','PutDown','Neutral']
        a_labels[5]= ['HandOver','Grip','LeftLoop','RightLoop','RightPinch','LeftPinch','PullThrough','Release','Return']

        data_info = [foldLaundry, writeHello, openJar, screwLightbulb, combHair, tieShoes]
        activity_names = ['foldLaundry', 'writeHello','openJar', 'screwLightbulb', 'combHair', 'tieShoes']

    accs_all=[0]*task_len
    accs_acc = np.zeros([recorded_trials,task_len+1])
    accs_error = [0]*(task_len+1)
    accs_mean_figure = [0]*(task_len+1)
    all_accs_trials = [0]*recorded_trials

    accs_all_svm=[0]*task_len
    accuracies_out={}
    accuracies_out_svm={}

    if remove_reps:
        # map repetitive motion to each other
        #opening jar
        subtask_label[np.where(subtask_label==207)] = 205
        subtask_label[np.where(subtask_label==209)] = 205
        subtask_label[np.where(subtask_label==211)] = 205
        subtask_label[np.where(subtask_label==208)] = 206
        subtask_label[np.where(subtask_label==210)] = 206
        subtask_label[np.where(subtask_label==212)] = 206

        #screwing lightbulb
        subtask_label[np.where(subtask_label==306)] = 304
        subtask_label[np.where(subtask_label==308)] = 304
        subtask_label[np.where(subtask_label==310)] = 304
        subtask_label[np.where(subtask_label==312)] = 304
        subtask_label[np.where(subtask_label==307)] = 305
        subtask_label[np.where(subtask_label==309)] = 305
        subtask_label[np.where(subtask_label==311)] = 305

        #combing hair
        subtask_label[np.where(subtask_label==409)] = 407
        subtask_label[np.where(subtask_label==411)] = 407
        subtask_label[np.where(subtask_label==413)] = 407
        subtask_label[np.where(subtask_label==410)] = 408
        subtask_label[np.where(subtask_label==412)] = 408
        subtask_label[np.where(subtask_label==414)] = 408

    if remove_similar:
        #combing hair
        subtask_label[np.where(subtask_label==401)] = 402
        #write hello
        subtask_label[np.where(subtask_label==108)] = 103
        subtask_label[np.where(subtask_label==105)] = 110

    # max_std_value = -10
    # min_std_value = 10
    # max_delta_value = -10
    # min_delta_value = 10
    #for debug in range(1):
    #    task = 4
    for task in range(task_len):
        if verbosity:
            print('\nTask: ', activity_names[task])
        if not split_tasks:  # all 75 subtasks classified at the same time
            task_3d = np.array([np.transpose(accelerometer_data[window_size * i:100 * (i + 1), :]) for i in
                                range(len(np.unique(subtask_label)) * recorded_trials)])
            task_labels_3d = np.array([subtask_label[0][window_size * i] for i in range(len(np.unique(subtask_label)) * recorded_trials)])
        else: 
            #only use the subset of the dataset corresponding to current task
            first_task = task_locations[task]  # np.where(subtask_label < 100)[1]
            first_task_labels = subtask_label[0][first_task]
            first_task = accelerometer_data[first_task]
            labels = a_labels[task]

            #for rocket

            #for case when there's not enough data for last label (trial accidentally ended early), zero fill
            if task==4:
                i=79
                if np.shape(first_task[100*i:100*(i+1),:])!=(100, 3):
                    x = np.shape(first_task[100*i:100*(i+1),:])
                    existing = x[0]
                    to_add = 100-existing
                    first_task = np.append(first_task,np.zeros([to_add,3]),axis=0)

            task_3d=np.array([np.transpose(first_task[100*i:100*(i+1),:]) for i in range(len(labels)*recorded_trials)])

            # extract features
            if extra_features:

                # setup feature data array for size of features
                if include_full_features:
                    data_with_features = np.zeros((len(labels)*recorded_trials,8,window_size))
                elif include_std ^ include_delta:
                    data_with_features = np.zeros((len(labels)*recorded_trials,6,window_size))
                elif include_std and include_delta:
                    data_with_features = np.zeros((len(labels)*recorded_trials,9,window_size))

                # generate features
                for i in range(len(labels)*recorded_trials):
                    # xyz original accelerometer signals
                    data = task_3d[i][:][:]
                    # full feature set
                    if include_full_features:
                        mag = np.zeros((1,window_size))
                        jerk = np.zeros((3,window_size))
                        jerk_mag = np.zeros((1,window_size))
                        mean = np.zeros((8,window_size))
                        std = np.zeros((8,window_size))
                        mean_abs_dev = np.zeros((8,window_size))
                        min_frame = np.zeros((8,window_size))
                        max_frame = np.zeros((8,window_size))
                        sig_mag_area = np.zeros((8,window_size))
                        energy_measure = np.zeros((8,window_size))
                        interquartile_range = np.zeros((8,window_size))
                        signal_entropy = np.zeros((8,window_size))
                        std = np.zeros((8,window_size))
                        auto_reg_coeffs = np.zeros((8,window_size))
                        # magnitude
                        for x in range(0,window_size,1):
                            mag[0,x] = math.sqrt(tsfresh.feature_extraction.feature_calculators.abs_energy(np.transpose(data[:,x])))
                        # jerk
                        for x in range(1,window_size,1):
                            for channel in range(3):
                                jerk[channel,x] = data[channel,x] - data[channel,x-1]   
                            jerk_mag[0,x] = math.sqrt(tsfresh.feature_extraction.feature_calculators.abs_energy(np.transpose(jerk[:,x])))
                        # main 8 signals
                        signals = np.append(mag,jerk,axis=0)
                        extra_features_data = np.append(signals,jerk_mag,axis=0)
                        #print("shape of added features",np.shape(extra_features_data))
                        #print(extra_features_data)

                        # extracted features on the 8 signals
                    
                    # both delta and std
                    elif include_std and include_delta:
                        std_data = np.zeros((3,window_size))
                        delta_data = np.zeros((3,window_size))
                        for x in range(feature_window,window_size,1):
                            std_data[:,x-1] = np.std(data[:,x-feature_window:x],axis=1)
                        for x in range(1,window_size,1):
                            delta_data[:,x] = data[:,x] - data[:,x-1]
                        extra_features_data = np.append(std_data,delta_data,axis=0)

                    # only std
                    elif include_std:
                        std_data = np.zeros((3,window_size))
                        for x in range(feature_window,window_size,1):
                            std_data[:,x-1] = np.std(data[:,x-feature_window:x],axis=1)
                        extra_features_data = std_data

                    # only delta
                    elif include_delta:
                        delta_data = np.zeros((3,window_size))
                        for x in range(1,window_size,1):
                            delta_data[:,x] = data[:,x] - data[:,x-1]
                        extra_features_data = delta_data

                    # for additional analysis
                    # if np.amax(std_data) > max_std_value:
                    #     max_std_value = np.amax(std_data)
                    # if np.amin(std_data) < min_std_value:
                    #     min_std_value = np.amin(std_data)
                    # if np.amax(delta_data) > max_delta_value:
                    #     max_delta_value = np.amax(delta_data)
                    # if np.amin(delta_data) < min_delta_value:
                    #     min_delta_value = np.amin(delta_data)

                    #append new features to accelerometer data 
                    data_with_features[i][:][:] = np.append(task_3d[i][:][:], extra_features_data,axis=0)
                #assign to data array
                task_3d = data_with_features

            task_labels_3d=np.array([first_task_labels[100*i] for i in range(len(labels)*recorded_trials)])
            #print(task_3d.shape)
            #print(task_labels_3d.shape)

            #for svm
            XAll = first_task
            yAll = first_task_labels
            #print(XAll.shape)
            #print(yAll.shape)

            # XAll = np.array([np.mean(first_task[100*i:100*(i+1),:],axis=0) for i in range(len(labels)*recorded_trials)])
            # yAll = np.array([first_task_labels[100*i] for i in range(len(labels)*recorded_trials)])
            # print(XAll.shape)
            # print(yAll.shape)

        for trial in range(experiment_trials):
            #svm
            accuracies_svm = []
            fold=0

            #random forest
            # for trainIdx, testIdx in skf.split(XAll, yAll):
            #     XTrain, XTest = XAll[trainIdx], XAll[testIdx]
            #     yTrain, yTest = yAll[trainIdx], yAll[testIdx]
            #     classifier = RandomForestClassifier(n_estimators=50)
            #     classifier.fit(XTrain,yTrain)
            #     y_pred = classifier.predict(XTest)
            #     accuracies_svm.append(accuracy_score(yTest, y_pred))
            #     if verbosity:
            #         print('Random Forest Classifier accuracy  for fold ', fold,': ', accuracy_score(yTest, y_pred), )
            #     fold+=1
            # accs_all_svm[task]=np.mean(np.array(accuracies_svm))

            # for trainIdx, testIdx in skf.split(XAll, yAll):
            #     XTrain, XTest = XAll[trainIdx], XAll[testIdx]
            #     yTrain, yTest = yAll[trainIdx], yAll[testIdx]
            #     clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5,max_iter=2000))
            #     #clf = LinearSVC(random_state=0, tol=1e-5)
            #     clf.fit(XTrain,yTrain)
            #     y_pred = clf.predict(XTest)
            #     accuracies_svm.append(accuracy_score(yTest, y_pred))
            #     if verbosity:
            #         print('SVM accuracy  for fold ', fold,': ', accuracy_score(yTest, y_pred), )
            #     fold+=1
            # accs_all_svm[task]=np.mean(np.array(accuracies_svm))

            #rocket
            accuracies = []
            fold=0
            for train_index, test_index in skf.split(task_3d, task_labels_3d):
                #Reshape numpy array to 3D (samples,dims,time)
                task_3d_train = task_3d[train_index][:][:]
                task_labels_3d_train = task_labels_3d[train_index][:][:]
                task_3d_test = task_3d[test_index][:][:]
                task_labels_3d_test = task_labels_3d[test_index][:][:]
                #Predict accuracy with the rocket classifier
                rocket = RocketClassifier()
                rocket.fit(task_3d_train, task_labels_3d_train)
                y_pred = rocket.predict(task_3d_test)
                accuracies.append(accuracy_score(task_labels_3d_test, y_pred))
                if verbosity:
                    print('Rocket accuracy  for fold ', fold,': ', accuracy_score(task_labels_3d_test, y_pred), )
                if (fold==0):
                    test_full = task_labels_3d_test
                    pred_full = y_pred
                else:
                    test_full = np.append(test_full,task_labels_3d_test,axis=0)
                    pred_full = np.append(pred_full,y_pred,axis=0) 
                fold+=1
            accs_acc[:,task] = accuracies
            accs_all[task]=np.mean(np.array(accuracies))
            plt.rcParams["figure.figsize"] = (13,10)
            plt.figure(fold)
            disp = ConfusionMatrixDisplay.from_predictions(test_full,pred_full,display_labels = np.arange(len(np.unique(task_labels_3d))), include_values=False)
            plt.title('\nTask: '+str(activity_names[task]))
            plt.savefig(str(activity_names[task])+'.png')
    print(accs_all)
    average_accuracy = np.mean(np.array(accs_all))
    all_accs_trials = np.mean(accs_acc,axis=1)
    #print(accs_acc)
    #print(all_accs_trials)
    accs_acc[:,task_len] = np.transpose(all_accs_trials)
    #print(accs_acc)
    accs_error = np.std(accs_acc,axis=0)
    accs_mean_figure = np.append(accs_all,average_accuracy)
    print("the cross validation accuracy per activity and overall average for this subject is", accs_mean_figure)
    print("the error per activity and overall for this subject is",accs_error)
    # print("the max std value is: ",max_std_value)
    # print("the min std value is: ",min_std_value)
    # print("the max max_delta_value is: ", max_delta_value)
    # print("the min delta value is: ",min_delta_value)

    #accs_all = np.append(accs_all,average_accuracy)
    # labels = ['Folding laundry', 'Writing with pen', 'Opening a jar','Screwing lightbulb','Combing hair','Tying shoelaces','All']
    # x_pos = np.arange(len(labels))
    # fig, ax = plt.subplots()
    # ax.set_ylabel('Task recognition accuracy (%)',fontsize=20)
    # ax.set_xticks(x_pos,labels,fontsize=16)
    # S1_bar = ax.bar(x_pos,accs_all)
    # ax.bar_label(S1_bar,padding=3, fontsize=14)

    # # Save the figure and show
    # plt.yticks(fontsize=16)
    # fig.tight_layout()
    # plt.show()
    # plt.savefig(str('all_accuracies.png'))

    if verbosity:
        print('Rocket: Average 5-fold stratified accuracy over %2d trials: %3f ' % (experiment_trials,np.mean(np.array(accs_all))))
    accuracy_out=np.mean(np.array(accs_all))
    accuracies_out[activity_names[task]]=accuracy_out
    # print(accs_all_svm)
    # if verbosity:
    #     print('SVM: Average 5-fold stratified accuracy over %2d trials: %3f ' % (experiment_trials,np.mean(np.array(accs_all_svm))))
    # accuracy_out_svm=np.mean(np.array(accs_all_svm))
    # accuracies_out_svm[activity_names[task]]=accuracy_out_svm
    return accuracies_out, accuracies_out_svm

    #These lines correspond to another timeseries classifier just for reference, ignore for now
    # hc2 = HIVECOTEV2(time_limit_in_minutes=1)
    # hc2.fit(task_3d_train, task_labels_3d_train)
    # y_predhc = hc2.predict(task_3d_test)
    # print('hc2', accuracy_score(task_labels_3d_test, y_predhc))


subtask_label, accelerometer_data=load_accelerometer_data()

#For individual tasks
accuracy_dict_out, accuracies_out_svm =timeseries_classifier(subtask_label, accelerometer_data,split_tasks=True,extra_features=True,remove_reps=True,remove_similar=True, include_std=True,include_delta=True,include_full_features=False,feature_window=10)
#For all 75 classes
#accuracy_dict_out=timeseries_classifier(subtask_label, accelerometer_data)


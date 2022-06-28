import numpy as np
from scipy import signal
from typing import Dict

foldLaundry = {'name': 'Fold Laundry', 'idx': 0, 'actionLength': 8, 'tutorialTime': 18, 'relaxTime': 15, 'continuousTime': 20}
writeHello = {'name': 'Write Hello', 'idx': 1, 'actionLength': 15, 'tutorialTime': 43, 'relaxTime': 5, 'continuousTime': 20}
openJar = {'name': 'Open Jar', 'idx': 2, 'actionLength': 14, 'tutorialTime': 32, 'relaxTime': 15, 'continuousTime': 20}
screwLightbulb = {'name': 'Screw in Lightbulb', 'idx': 3, 'actionLength': 13, 'tutorialTime': 34, 'relaxTime': 15, 'continuousTime': 20}
combHair = {'name': 'Comb Hair', 'idx': 4, 'actionLength': 16, 'tutorialTime': 40, 'relaxTime': 5, 'continuousTime': 20}
tieShoes = {'name': 'Tie Shoelaces', 'idx': 5, 'actionLength': 9, 'tutorialTime': 20, 'relaxTime': 5, 'continuousTime': 20}

experiments = {0: foldLaundry, 1: writeHello, 2: openJar, 3: screwLightbulb, 4: combHair, 5: tieShoes}

def extract_relevant_data(exp, emg, force, acc, reps):
    # grab relevant section to experiment
    fullDiscreteTime = 10 + exp['actionLength'] * 5 + exp['relaxTime']
    print("full discrete time is: ",fullDiscreteTime)
    fullContinuousTime = 5 + exp['continuousTime'] + exp['relaxTime']
    print("full continuous time is: ",fullContinuousTime)
    startBufferTime = 2 + exp['tutorialTime'] + fullDiscreteTime # filter out tutorial 
    print("full start time is: ",startBufferTime)

    print(f"total expected samples: {(startBufferTime + fullDiscreteTime * reps + fullContinuousTime * reps) * 1000}")
    
    fullContinuousLength = fullContinuousTime * 1000 
    fullDiscreteLength = fullDiscreteTime * 1000 

    emg = emg[startBufferTime * 1000:]
    force = force[startBufferTime * 1000:]
    acc = acc[startBufferTime * 1000:]

    emg_disc = emg[:fullDiscreteLength * reps]
    force_disc = force[:fullDiscreteLength * reps]
    acc_disc = acc[:fullDiscreteLength * reps]

    cont_start = fullDiscreteLength * reps
    cont_end = cont_start + fullContinuousLength * reps

    emg_cont = emg[cont_start: cont_end]
    force_cont = force[cont_start: cont_end]
    acc_cont = acc[cont_start: cont_end]

    return emg_disc, force_disc, acc_disc, emg_cont, force_cont, acc_cont, fullDiscreteLength, fullContinuousLength

def common_preprocessing(rec, dropSamples=10, log=False):

    # extract data
    crc = np.copy(rec['crc'])
    data = np.copy(rec['raw'])
    if log:
        print(f"shape of raw recording: {data.shape}, type of data: {type(data)}")
    
    # clean crcs
    # deal with situation where first sample is corrupt
    firstClean = np.where(crc==0)[0][0]
    data[:firstClean] = data[firstClean]
    
    # deal with remaining crc values
    crcIdx = np.where(crc!=0)[0]
    crcIdx = crcIdx[crcIdx > firstClean]
    for i in crcIdx:
        data[i] = data[i-1]
    
    if log:
        print(f"type of data: {type(data)}")

    if log:
        print(f"shape of emg1: {np.shape(data[:,0:32])}, shape of emg2: {np.shape(data[:,37:64])}, force: {np.shape(data[:,32:37])}, acc: {np.shape(data[:,64:])}")        
    
    # extract emg, force and acc separately - force sensor idx wrong
    if (rec['wide']):
        print(f"shape of emg1: {np.shape(data[:,0:32])}, shape of emg2: {np.shape(data[:,37:64])}, force: {np.shape(data[:,32:37])}, acc: {np.shape(data[:,64:])}")        
        emg = np.concatenate((np.copy(data[:,0:32]), np.copy(data[:,37:64])),axis=1) * 400 / 2**15 - 200 # convert from raw uint16 to voltage (-+200mv)
        force = np.copy(data[:,32:37]) * 400 / 2**15 - 200
    else:
        emg = np.concatenate((np.copy(data[:,0:32]), np.copy(data[:,37:64])),axis=1) * 100 / 2**15 - 50 # convert from raw uint16 to voltage (-+50mv)
        force = np.copy(data[:,32:37]) * 100 / 2**15 - 50
    acc = ((data[:,64:] + 2**15) % 2**16 - 2**15)/2**14
    
    emg = emg[dropSamples:]
    force = force[dropSamples:]
    acc = acc[dropSamples:]

    if log:
        print(f"shapes after initial cleaning and extraction ==> emg: {np.shape(emg)}, force: {np.shape(force)}, acc: {np.shape(acc)}")
        print(f"lengths after initial cleaning and extraction ==> emg: {len(emg)}, force: {len(force)}, acc: {len(acc)}")

    return emg, force, acc

def process_single_trial_daily_activity(rec, dropSamples = 10, featureWindow = 50, log = False):
    emg, force, acc = common_preprocessing(rec, dropSamples, log)
    exp = experiments[rec['experiment']]

    theoreticalLength = 0
    if rec['continuous']:
        theoreticalLength = (5 + exp['continuousTime']) * 1000
    elif rec['discrete']:
        theoreticalLength = (10 + exp['actionLength'] * 5) * 1000

    # data collection starts at same time as experiment, so the first samples are the actual data
    emg = emg[:theoreticalLength]
    acc = acc[:theoreticalLength]
    force = force[:theoreticalLength]

    if log:
        print(f"length of emg: {len(emg)}, acc: {len(acc)}, force: {len(force)}, theoretical num samples: {theoreticalLength}")

    recordingLength = len(emg)

    # create labels
    trialLabel = np.ones(recordingLength)

    if rec['discrete']:
        # label each sample; can label each subgesture separately or all together; can include or exclude relax and transitions
        activityGestureLabels = discrete_labels(recordingLength, recordingLength, exp, labelingMethod=2)
    elif rec['continuous']:
        # label each sample, no subgestures, so single label for each activity
        activityGestureLabels = continuous_labels(recordingLength, recordingLength, exp, labelingMethod=1)
    
    return smooth_into_features(emg, acc, force, trialLabel, activityGestureLabels, featureWindow)


def smooth_into_features(emg, acc, force, trialLabels, subGestureLabels, featureWindow):
    # downsample labels for feature processing
    featTrialLabel = trialLabels[::featureWindow]
    featSubGestureLabels = subGestureLabels[::featureWindow]

    recordingLength = len(emg)
    
    # process discrete raw into features by smoothing

    numWindow = int(recordingLength/featureWindow)

    numChannel = emg.shape[1]
    emgFeat = np.zeros((numWindow, numChannel))
    numChannel = acc.shape[1]
    accFeat = np.zeros((numWindow, numChannel))
    numChannel = force.shape[1]
    forceFeat = np.zeros((numWindow, numChannel))
    for i in range(numWindow):
        t = np.arange(featureWindow) + i*featureWindow
        emgFeat[i,:] = np.mean(abs(signal.detrend(emg[t],axis=0)),axis=0)
        accFeat[i,:] = np.mean(acc[t],axis=0)
        forceFeat[i,:] = np.mean(force[t], axis=0)
    
    return emgFeat, accFeat, forceFeat, featTrialLabel, featSubGestureLabels


def process_daily_activities(rec, dropSamples = 10, featureWindow=50, log=True):
    
    # extract data
    crc = np.copy(rec['crc'])
    data = np.copy(rec['raw'])
    if log:
        print(f"shape of raw recording: {data.shape}, type of data: {type(data)}")
    
    # clean crcs
    # deal with situation where first sample is corrupt
    firstClean = np.where(crc==0)[0][0]
    data[:firstClean] = data[firstClean]
    
    # deal with remaining crc values
    crcIdx = np.where(crc!=0)[0]
    crcIdx = crcIdx[crcIdx > firstClean]
    for i in crcIdx:
        data[i] = data[i-1]
    
    if log:
        print(f"type of data: {type(data)}")

    if log:
        print(f"shape of emg1: {np.shape(data[:,0:32])}, shape of emg2: {np.shape(data[:,37:64])}, force: {np.shape(data[:,32:37])}, acc: {np.shape(data[:,64:])}")        
    
    # extract emg, force and acc separately - force sensor idx wrong
    if (rec['wide']):
        print(f"shape of emg1: {np.shape(data[:,0:32])}, shape of emg2: {np.shape(data[:,37:64])}, force: {np.shape(data[:,32:37])}, acc: {np.shape(data[:,64:])}")        
        emg = np.concatenate((np.copy(data[:,0:32]), np.copy(data[:,37:64])),axis=1) * 400 / 2**15 - 200 # convert from raw uint16 to voltage (-+200mv)
        force = np.copy(data[:,32:37]) * 400 / 2**15 - 200
    else:
        emg = np.concatenate((np.copy(data[:,0:32]), np.copy(data[:,37:64])),axis=1) * 100 / 2**15 - 50 # convert from raw uint16 to voltage (-+50mv)
        force = np.copy(data[:,32:37]) * 100 / 2**15 - 50
    acc = ((data[:,64:] + 2**15) % 2**16 - 2**15)/2**14
    
    emg = emg[dropSamples:]
    force = force[dropSamples:]
    acc = acc[dropSamples:]

    if log:
        print(f"shapes after initial cleaning and extraction ==> emg: {np.shape(emg)}, force: {np.shape(force)}, acc: {np.shape(acc)}")
        print(f"lengths after initial cleaning and extraction ==> emg: {len(emg)}, force: {len(force)}, acc: {len(acc)}")

    exp = experiments[rec['experiment']]

    # extract relevant data by dropping relax and transitions 
    emg_disc, force_disc, acc_disc, emg_cont, force_cont, acc_cont, fullDiscLength, fullContLength = extract_relevant_data(exp, emg, force, acc, rec['reps'])

    if log:
        print(f"length of disc emg: {len(emg_disc)}, acc: {len(acc_disc)}, force: {len(force_disc)}, theoretical size: {fullDiscLength * rec['reps']}")
        print(f"length of cont emg: {len(emg_cont)}, acc: {len(acc_cont)}, force: {len(force_cont)}, theoretical size: {fullContLength * rec['reps']}")

    discreteRecordingLength = len(emg_disc)
    continuousRecordingLength = len(emg_cont)

    # create labels
    discTrialLabel = np.empty(discreteRecordingLength)
    discTrialLabel[:] = np.nan
    for r in range(rec['reps']):
        idxStart = r * fullDiscLength
        idxEnd = idxStart + fullDiscLength
        discTrialLabel[idxStart:idxEnd] = r

    # label each sample; can label each subgesture separately or all together; can include or exclude relax and transitions
    discActivityGestureLabels = discrete_labels(fullDiscLength, rec['reps'] * fullDiscLength, exp, labelingMethod=2)

    contTrialLabel = np.empty(continuousRecordingLength)
    contTrialLabel[:] = np.nan
    for r in range(rec['reps']):
        idxStart = r * fullContLength
        idxEnd = idxStart + fullContLength
        contTrialLabel[idxStart:idxEnd] = r

    # label each sample, no subgestures, so single label for each activity
    contActivityGestureLabels = continuous_labels(fullContLength, rec['reps'] * fullContLength, exp, labelingMethod=1)

    # downsample labels for feature processing
    discTrialLabel = discTrialLabel[::featureWindow]
    contTrialLabel = contTrialLabel[::featureWindow]
    discActivityGestureLabels = discActivityGestureLabels[::featureWindow]
    contActivityGestureLabels = contActivityGestureLabels[::featureWindow]
    
    # process discrete raw into features by smoothing

    numWindow = int(discreteRecordingLength/featureWindow)
    numChannel = emg_disc.shape[1]
    emgFeat_disc = np.zeros((numWindow, numChannel))
    numChannel = acc_disc.shape[1]
    accFeat_disc = np.zeros((numWindow, numChannel))
    numChannel = force_disc.shape[1]
    forceFeat_disc = np.zeros((numWindow, numChannel))
    for i in range(numWindow):
        t = np.arange(featureWindow) + i*featureWindow
        emgFeat_disc[i,:] = np.mean(abs(signal.detrend(emg_disc[t],axis=0)),axis=0)
        accFeat_disc[i,:] = np.mean(acc_disc[t],axis=0)
        forceFeat_disc[i,:] = np.mean(force_disc[t], axis=0)
    
    # process continuous raw into features by smoothing
    numWindow = int(continuousRecordingLength/featureWindow)
    numChannel = emg_cont.shape[1]
    emgFeat_cont = np.zeros((numWindow, numChannel))
    numChannel = acc_cont.shape[1]
    accFeat_cont = np.zeros((numWindow, numChannel))
    numChannel = force_cont.shape[1]
    forceFeat_cont = np.zeros((numWindow, numChannel))
    for i in range(numWindow):
        t = np.arange(featureWindow) + i*featureWindow
        emgFeat_cont[i,:] = np.mean(abs(signal.detrend(emg_cont[t],axis=0)),axis=0)
        accFeat_cont[i,:] = np.mean(acc_cont[t],axis=0)
        forceFeat_cont[i, :] = np.mean(force_cont[t], axis=0)

    return (emgFeat_disc, accFeat_disc, forceFeat_disc, discTrialLabel, discActivityGestureLabels, emgFeat_cont, accFeat_cont, forceFeat_cont, contTrialLabel, contActivityGestureLabels)


def discrete_labels(trialLength: int, totalLength: int, experiment: Dict, labelingMethod: int) -> np.ndarray:
    labels = np.empty(totalLength)

    if labelingMethod == 0: # include buffer and relax
        labels[:] = experiment['idx']
    
    elif labelingMethod == 1: # exclude buffer and relax, all actions same label
        reps = int(totalLength / trialLength) # number of reps
        for r in range(reps):
            idxStart = r*trialLength
            idxEnd = idxStart + 10 * 1000
            labels[idxStart:idxEnd] = -1
            idxStart = idxEnd
            idxEnd = idxStart + int(experiment['actionLength'] * 5 * 1000)
            labels[idxStart:idxEnd] = experiment['idx'] * 100
            idxStart = idxEnd
            idxEnd = idxStart + int(experiment['relaxTime'] * 1000)
            labels[idxStart:idxEnd] = -1
    
    elif labelingMethod == 2: # exclude buffer and relax, all actions different label
        reps = int(totalLength / trialLength) # number of reps
        print(reps)
        for r in range(reps):
            idxStart = r*trialLength
            idxEnd = idxStart + 10 * 1000
            labels[idxStart:idxEnd] = -1
            for a in range(experiment['actionLength']):
                idxStart = idxEnd
                idxEnd = idxStart + 5 * 1000 # one more action which takes 5s
                labels[idxStart:idxEnd] = experiment['idx'] * 100 + a + 1 # start at 1 for the discretized action
            idxStart = idxEnd
            idxEnd = idxStart + int(experiment['relaxTime']*1000)
            labels[idxStart:idxEnd] = -1
    
    return labels


def continuous_labels(trialLength: int, totalLength: int, experiment: Dict, labelingMethod: int) -> np.ndarray:
    labels = np.empty(totalLength)

    if labelingMethod == 0: # include buffer and relax
        labels[:] = experiment['idx']
    
    elif labelingMethod == 1: # exclude buffer and relax
        reps = int(totalLength / trialLength) # number of reps
        for r in range(reps):
            idxStart = r*trialLength
            idxEnd = idxStart + 5 * 1000
            labels[idxStart:idxEnd] = -1
            idxStart = idxEnd
            idxEnd = idxStart + int(experiment['continuousTime']*1000)
            labels[idxStart:idxEnd] = experiment['idx'] * 100
            idxStart = idxEnd
            idxEnd = idxStart + int(experiment['relaxTime']*1000)
            labels[idxStart:idxEnd] = -1
    
    return labels
    

def process_recording_discrete_gestures(rec, dropSamples=0, featureWindow=50):
    # extract data
    crc = np.copy(rec['crc'])
    clean = np.copy(rec['raw'])
    
    # clean crcs
    # deal with situation where first sample is corrupt
    firstClean = np.where(crc==0)[0][0]
    clean[:firstClean] = clean[firstClean]
    
    # deal with remaining crc values
    crcIdx = np.where(crc!=0)[0]
    crcIdx = crcIdx[crcIdx > firstClean]
    for i in crcIdx:
        clean[i] = clean[i-1]
        
    # extract emg and accelerometer data separately
    emg = np.copy(clean[:,0:64])*100/2**15 - 50
    acc = ((clean[:,64:] + 2**15) % 2**16 - 2**15)/2**14
    emg = emg[dropSamples:]
    acc = acc[dropSamples:]
    
    # grab relevant section to experiment
    idxStart = int((rec['tBuffer'] + 1)*1000 - rec['tRelax']*1000/2)
    idxEnd = idxStart + int(rec['reps']*(4*rec['tTransition'] + rec['tGesture'] + rec['tRelax'])*1000)
    emg = emg[idxStart:idxEnd]
    acc = acc[idxStart:idxEnd]
    
    # create labels
    recLen = idxEnd - idxStart
    trialLabel = np.zeros(recLen)
    trialLabel[:] = np.nan
    for r in range(rec['reps']):
        idxStart = int(rec['tRelax']*1000/2 + r*(4*rec['tTransition'] + rec['tGesture'] + rec['tRelax'])*1000)
        idxEnd = idxStart + int((4*rec['tTransition'] + rec['tGesture'])*1000)
        trialLabel[idxStart:idxEnd] = r
    
    gestureLabel = np.zeros(recLen)
    gestureLabel[:] = np.nan
    for r in range(rec['reps']):
        idxStart = int(rec['tRelax']*1000/2 + r*(4*rec['tTransition'] + rec['tGesture'] + rec['tRelax'])*1000 + rec['tTransition']*1000)
        idxEnd = idxStart + int(rec['tTransition']*1000)
        gestureLabel[idxStart:idxEnd] = -1
        idxStart = idxEnd
        idxEnd = idxStart + int(rec['tGesture']*1000)
        gestureLabel[idxStart:idxEnd] = rec['gesture'] - 100
        idxStart = idxEnd
        idxEnd = idxStart + int(rec['tTransition']*1000)
        gestureLabel[idxStart:idxEnd] = -1
    
    positionLabel = np.zeros(recLen)
    positionLabel[:] = np.nan
    for r in range(rec['reps']):
        idxStart = int(rec['tRelax']*1000/2 + r*(4*rec['tTransition'] + rec['tGesture'] + rec['tRelax'])*1000)
        idxEnd = idxStart + int(rec['tTransition']*1000)
        positionLabel[idxStart:idxEnd] = -1
        idxStart = idxEnd
        idxEnd = idxStart + int(rec['tGesture']*1000 + 2*rec['tTransition']*1000)
        positionLabel[idxStart:idxEnd] = rec['position']
        idxStart = idxEnd
        idxEnd = idxStart + int(rec['tTransition']*1000)
        positionLabel[idxStart:idxEnd] = -1
        
    # downsample labels for feature processing
    trialLabel = trialLabel[::featureWindow]
    gestureLabel = gestureLabel[::featureWindow]
    positionLabel = positionLabel[::featureWindow]
    
    # process emg
    numWindow = int(recLen/featureWindow)
    numChannel = emg.shape[1]
    emgFeat = np.zeros((numWindow, numChannel))
    numChannel = acc.shape[1]
    accFeat = np.zeros((numWindow, numChannel))
    for i in range(numWindow):
        t = np.arange(featureWindow) + i*featureWindow
        emgFeat[i,:] = np.mean(abs(signal.detrend(emg[t],axis=0)),axis=0)
        accFeat[i,:] = np.mean(acc[t],axis=0)
        
    return (emgFeat, accFeat, trialLabel, gestureLabel, positionLabel)

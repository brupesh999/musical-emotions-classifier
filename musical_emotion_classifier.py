#from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
import librosa
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import IPython
import librosa.display
import warnings
import pickle
from argparse import ArgumentParser
import time

def create_df(dir):
  target = []
  audio = []
  for i in os.listdir(dir) :
    if i == 'Sad':
      for i in os.listdir(dir+'/'+'Sad') : 
        target.append('sad')
        audio.append(dir+'/'+'Sad'+'/'+i) 
    else:
      for i in os.listdir(dir+'/'+'Happy') : 
        target.append('happy')
        audio.append(dir+'/'+'Happy'+'/'+i) 
    df = pd.DataFrame(columns=['audio','target'])
    df['audio'] = audio
    df['target'] = target
  return df

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data, hop_length = 20).T, axis=0)
    result=np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate,n_fft = 20,hop_length = 20).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data, frame_length=100).T, axis=0)
    result = np.hstack((result, rms))

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, hop_length = 20).T, axis=0)
    result = np.hstack((result, mel))
    return result

def get_features(path, dur):
    try:
        data, sample_rate = librosa.load(path, sr=41100, offset=0.0, duration=dur)
        res1 = extract_features(data, sample_rate)
        result = np.array(res1)
        return result
    except:
        return None

def clean_null(input_list_x, input_list_y):
    for index, val in enumerate(input_list_x):
        try:
            len(val)
        except:
            input_list_x.pop(index)
            input_list_y.pop(index)
    return input_list_x, input_list_y

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('-a', '--action', dest="act", \
        help="f:everything, l:load audio, r:logic regression", type=str, choices=["f","l","r", "lin", "ridge", "sgd"])
    parser.add_argument('-f', '--file', dest="file",help="file to check", type=str)
    parser.add_argument('-d', '--duration', dest="duration",help="duration of audio sample", type=float)
    parser.add_argument('-i', '--iter', dest="iter",help="max iteration for regression", type=int)

    args = parser.parse_args()
    duration = 9.5
    max_iter = 10000

    start_time = time.time()

    warnings.filterwarnings("ignore")

    if args.act == "f" or args.act == "l":
        tst_dir = "Audio_Files/Audio_Files/Test"
        train_dir = "Audio_Files/Audio_Files/Train"

        train_df = create_df(train_dir)
        test_df = create_df(tst_dir)

        train_df['target'].value_counts()

        if args.duration is not None:
            duration = args.duration
        trainX, trainY = [], []
        for path, emotion in zip(train_df.audio, train_df.target):
            feature = get_features(path,duration)
            trainX.append(feature)
            trainY.append(emotion)

        testX, testY = [], []
        for path, emotion in zip(test_df.audio, test_df.target):
            feature = get_features(path,duration)
            testX.append(feature)
            testY.append(emotion)

        print("--- %s seconds ---" % (time.time() - start_time))
        pickle.dump(trainX, open("trainX.f", 'wb')) 
        pickle.dump(trainY, open("trainY.f", 'wb')) 
        pickle.dump(testX, open("testX.f", 'wb')) 
        pickle.dump(testY, open("testY.f", 'wb')) 

    if args.iter is not None:
            max_iter = args.iter    

    trainX = pickle.load(open("trainX.f", 'rb'))
    trainY = pickle.load(open("trainY.f", 'rb'))
    testX = pickle.load(open("testX.f", 'rb'))
    testY = pickle.load(open("testY.f", 'rb'))

    lb = LabelEncoder()
    trainX, trainY = clean_null(trainX, trainY)
    testX, testY = clean_null(testX, testY)
    ml_x_tr = np.array(trainX)
    ml_x_tst = np.array(testX)
    ml_y_tr = np.array(lb.fit_transform(trainY))
    ml_y_tst = np.array(lb.fit_transform(testY))

    ml_y_tr = ml_y_tr.reshape(-1,1)
    ml_y_tst = ml_y_tst.reshape(-1,1)

    if args.act == "f" or args.act == "r":
        ml_model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=max_iter, \
            multi_class='ovr', l1_ratio=0.5)
    elif args.act == "lin":
        ml_model = sklearn.linear_model.LinearRegression()
    elif args.act == "ridge":
        ml_model = sklearn.linear_model.RidgeClassifier()
    elif args.act == "sgd":
        ml_model = sklearn.linear_model.SGDClassifier()
        
    ml_start_time = time.time()
    ml_model.fit(ml_x_tr, ml_y_tr)
    
    if args.file == None:
        preds = ml_model.predict(ml_x_tst)
        correct = 0
        incorrect = 0
        for pred, gt in zip(preds, ml_y_tst):
            if args.act == "lin":
                final_pred = round(pred[0])
            else:
                final_pred = int(pred)
            if (final_pred) == gt:
                correct += 1
            else:
                incorrect += 1
        print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
        print(f"Time taken: {time.time() - ml_start_time: .2f}")
    else:
        file_feature = get_features(args.file, duration)
        ml_file_feature = np.array(file_feature)
        ml_file_feature = ml_file_feature.reshape(1,-1)
        pred = ml_model.predict(ml_file_feature)
        if args.act == "lin":
            final_pred = round(pred[0][0])
        else:
            final_pred = int(pred)
        if final_pred == 0:
            print(final_pred, "Sad")
        elif final_pred == 1:
            print(final_pred, "Happy")
        else:
            print(final_pred, "Could not make prediction.")

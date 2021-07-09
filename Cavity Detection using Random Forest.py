from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

fixed_size  = tuple((500,500))
train_path = "C:\\pushkara_files\\teeth_dataset\\teeth_dataset\\Trianing"
num_tree = 100
bins = 8
test_size = 0.10
seed = 9

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image],[0,1,2],None,[bins,bins,bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist,hist)
    return hist.flatten()

train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)
global_features = []
labels = []
i, j = 0, 0 
k = 0
images_per_class = 80

%time
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    k = 1
    for file in os.listdir(dir):
        file = dir + "/" + os.fsdecode(file)
        image = cv2.imread(file) 
        
        if image is not None:
            image = cv2.resize(image,fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            labels.append(current_label)
            global_features.append(global_feature)
            i += 1
            k += 1
    print("[STATUS] processed folder: {}".format(current_label))
    j += 1

print("[STATUS] completed Global Feature Extraction...")

%time
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))
print("[STATUS] training Labels {}".format(np.array(labels).shape))
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...{}")
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")
print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))
h5f_data = h5py.File('C:\\pushkara_files\\teeth_dataset\\teeth_dataset\\data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('C:\\pushkara_files\\teeth_dataset\\teeth_dataset\\labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")

h5f_data = h5py.File('C:\\pushkara_files\\teeth_dataset\\teeth_dataset\\data.h5', 'r')
h5f_label = h5py.File('C:\\pushkara_files\\teeth_dataset\\teeth_dataset\\labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

clf  = RandomForestClassifier(n_estimators=100)
clf.fit(trainDataGlobal, trainLabelsGlobal)
clf_pred = clf.predict(trainDataGlobal)
print(classification_report(trainLabelsGlobal,clf_pred))
print(accuracy_score(trainLabelsGlobal,clf_pred))

test_path = "C:\\pushkara_files\\teeth_dataset\\teeth_dataset\\test"
for file in os.listdir(test_path):
    file = test_path + "/" + file
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    


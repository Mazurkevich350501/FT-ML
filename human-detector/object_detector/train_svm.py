from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import os
from config import *
import numpy as np

def train_svm():
    pos_feat_file_paths = os.path.join(pos_feat_path, '*.feat')
    neg_feat_file_paths = os.path.join(neg_feat_path, '*.feat')

    fds = []
    labels = []

    def append_data(feat_path, label):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(label)

    pos_feats = glob.glob(pos_feat_file_paths)
    print("Load positive feats({})...".format(len(pos_feats)))
    for feat_path in pos_feats:
        append_data(feat_path, 1)
    print("Finish")   

    neg_feats = glob.glob(os.path.join(neg_feat_file_paths))
    print("Load negative feats({})...".format(len(neg_feats)))
    for feat_path in neg_feats:
        append_data(feat_path, 0)
    print("Finish")
    
    print (np.array(fds).shape,len(labels))
    clf = LinearSVC()
    print ("Training a Linear SVM Classifier")
    clf.fit(fds, labels)
    joblib.dump(clf, os.path.join(model_path, 'svm.model'))
    print ("Classifier saved to {}".format(model_path))
        
train_svm()
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
from config import *

def generate_features_and_save_to_file(im_path, feat_dir):
    im = imread(im_path, as_grey=True)
    fd = hog(im, orientations, pixels_per_cell, cells_per_block, block_normalization, visualize, normalize)
    fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
    fd_path = os.path.join(feat_dir, fd_name)
    joblib.dump(fd, fd_path) 

def extract_features():
    if not os.path.isdir(pos_feat_path):
        os.makedirs(pos_feat_path)

    if not os.path.isdir(neg_feat_path):
        os.makedirs(neg_feat_path)

    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        generate_features_and_save_to_file(im_path, pos_feat_path)
    print("Positive features saved in {}".format(pos_feat_path))

    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        generate_features_and_save_to_file(im_path, neg_feat_path)
    print("Negative features saved in {}".format(neg_feat_path))

    print("Completed calculating features from training images")

# if __name__=='__main__':
extract_features()
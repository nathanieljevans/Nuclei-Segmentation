# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:36:11 2018

This is the experimental segmentation script to explore the 2018 Data Bowl Competition: https://www.kaggle.com/c/data-science-bowl-2018

@team: teddy-the-magnificient
@author: Nate
"""

import os 
import zipfile
import cv2

def main(): 
    unpack_if_needed()
    
    # plot a few examples 
    img_path = "unpacked_datasets\train-data\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\images\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e"
    img = cv2.imread(img_path)
    cv2.imshow('first cell image!', img)
    

def unpack_if_needed():
    print("checking for available datasets")
    dataset_names = ["stage1_sample_submission.csv.zip", "stage1_test.zip", "stage1_train.zip", "stage1_train_labels.csv.zip"]
    dirs = ["samp", "test", "train-data", "train-labels"]
    target_dir = "unpacked_datasets"
    
    if (not os.path.isdir(target_dir) ):
        os.mkdir(target_dir)
        for d, n in zip(dirs, dataset_names):  
            print(n[0:-4])
            print(str(not os.path.isdir(target_dir + '/' + n[0:-4])))
            print(str(not os.path.exists(target_dir + '/' + n[0:-4])))
            print(str(not os.path.isdir(target_dir + '/' + n[0:-4])) and (not os.path.exists(target_dir + '/' + n[0:-4])))
            print('unpacking data: ' + n)
            zip_ref = zipfile.ZipFile(n, 'r')
            zip_ref.extractall("unpacked_datasets/" + d)
            zip_ref.close()
    else: 
        print("dataset already unpacked")
            
if __name__ == '__main__' :
    main() 


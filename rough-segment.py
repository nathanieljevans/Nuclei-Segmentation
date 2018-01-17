# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:36:11 2018

This is the experimental segmentation script to explore the 2018 Data Bowl Competition: https://www.kaggle.com/c/data-science-bowl-2018

@team: teddy-the-magnificient
@author: Nate
"""

import os 
import zipfile

def unpack_if_needed():
    print("checking for available datasets")
    dataset_names = ["stage1_sample_submission.csv.zip", "stage1_test.zip", "stage1_train.zip", "stage1_train_labels.csv.zip"]
    dirs = ["samp", "test", "train-data", "train-labels"]
    target_dir = "unpacked_datasets"
    
    for d, n in zip(dirs, dataset_names): 
        if (not os.path.isdir(target_dir + '//' + n[0:-4]) and not os.path.exists(target_dir + '//' + n[0:-4])): 
            print('unpacking data: ' + n)
            zip_ref = zipfile.ZipFile(n, 'r')
            if (not os.path.isdir("unpacked_datasets")): 
                os.mkdir(target_dir)
            zip_ref.extractall("unpacked_datasets//" + d)
            zip_ref.close()
        else: 
            print("dataset already unpacked")
    print('data available!')
            
unpack_if_needed()
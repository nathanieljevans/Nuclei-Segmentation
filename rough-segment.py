# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:36:11 2018

This is the experimental segmentation script to explore the 2018 Data Bowl Competition: https://www.kaggle.com/c/data-science-bowl-2018

@team: teddy-the-magnificient
@author: Nate
"""

import os 
import zipfile
from scipy import misc
from matplotlib import pyplot as plt 
import numpy as np

def main(): 
    unpack_if_needed()
    paths = get_example_paths(n=5)
    print(str(paths[0]))
    
    # plot a few examples 
    pth = "unpacked_datasets/train-data/"
    train_dir_names = os.listdir(pth)
    first_example = pth + train_dir_names[0] + '/images'
    img_pth = first_example + '/' + os.listdir(first_example)[0]
    print(img_pth)
    print(str(os.path.exists(img_pth)))
    print("----")
    img = misc.imread(img_pth)
    plt.imshow(img[:,:,1])
    #misc.imshow(img_pth)
    print(np.array(img).shape) 
    
# input
# n is the number the length of the list to return, how many examples 
# output 
# returns list of tuples, tuple holds one example: (img path, list of mask paths)
def get_example_paths(n=0): 
    pth = "unpacked_datasets/train-data/"
    example_paths = []
    for i, example in enumerate(os.listdir(pth)): 
        if (n == 0 or i < n):
            img = pth + example + '/images'
            mask = pth + example + '/masks' 
            img_pth = img + '/' + os.listdir(img)[0]
            mask_pths = list(map(lambda x: mask + '/' + x, os.listdir(mask)))
            example_paths.append( (img_pth, mask_pths) )
    return example_paths

def unpack_if_needed():
    print("checking data state")
    dataset_names = ["stage1_sample_submission.csv.zip", "stage1_test.zip", "stage1_train.zip", "stage1_train_labels.csv.zip"]
    dirs = ["samp", "test", "train-data", "train-labels"]
    target_dir = "unpacked_datasets"
    
    if (not os.path.isdir(target_dir) ):
        print('data is zipped')
        os.mkdir(target_dir)
        for d, n in zip(dirs, dataset_names):  
            print('unpacking data: ' + n)
            zip_ref = zipfile.ZipFile(n, 'r')
            zip_ref.extractall("unpacked_datasets/" + d)
            zip_ref.close()
    else: 
        print("dataset already unpacked")
            
if __name__ == '__main__' :
    main() 


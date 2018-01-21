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
from skimage import filters, segmentation, color, measure

def main(): 
    unpack_if_needed()
    path_q = get_example_paths(n=0)
    
    # plot a few examples 

    img_tup = load_next_image(path_q) 
    
    segmented_img = segment_single_image(img_tup[0])
    
# returns tuple containing: (test_img, [mask_imgs])
def load_next_image(path_q): 
    path = path_q.get_next() 
    test_img = misc.imread(path[0])
    mask_imgs = []
    for p in path[1]: 
        mask_imgs.append(misc.imread(p))
    plt.imshow(test_img[:,:,1])
    plt.show()
    return (test_img, mask_imgs)

# takes in img 
# returns 
def segment_single_image(img):
    OFFSET = -0.05
    img = color.rgb2gray(img)
    print(filters.threshold_otsu(img))
    
    mask = img > filters.threshold_otsu(img) + OFFSET
    print(np.array(mask).shape)
    
    clean_border = segmentation.clear_border(mask)
    plt.imshow(clean_border, cmap='gray')
    plt.show()
    
    labels = measure.label(mask)
    print('labels')
    plt.imshow(labels)
    
    return labels

class path_queue: 
    def __init__(self, img_path, list_of_mask_paths):
        self.index = 0
        self.img_path = img_path
        self.mask_paths = list_of_mask_paths 
    def get_next(self): 
        self.index+=1 
        if (self.index-1 < len(self.img_path)):
            return (self.img_path[self.index-1], self.mask_paths[self.index-1])
        else: 
            print('no paths available')
            return []
        
    
# input
# n is the number the length of the list to return, how many examples 
# output 
# returns queue object, use .get_next() to get tuple (img path, list of mask paths)
def get_example_paths(n=0): 
    pth = "unpacked_datasets/train-data/"
    img_paths = []
    mask_paths = []
    for i, example in enumerate(os.listdir(pth)): 
        if (n == 0 or i < n):
            img = pth + example + '/images'
            mask = pth + example + '/masks' 
            img_pth = img + '/' + os.listdir(img)[0]
            mask_pths = list(map(lambda x: mask + '/' + x, os.listdir(mask)))
            img_paths.append(img_pth)
            mask_paths.append(mask_pths)
        
    return path_queue(img_paths, mask_paths)

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


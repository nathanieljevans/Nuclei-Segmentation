# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:36:11 2018

This is the experimental segmentation script to explore the 2018 Data Bowl Competition: https://www.kaggle.com/c/data-science-bowl-2018

@team: teddy-the-magnificient
@author: Nate
"""

# Function description format
''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
''' 


import os 
import zipfile
from scipy import misc
from matplotlib import pyplot as plt 
import numpy as np
from skimage import filters, segmentation, color, measure

sigma = 2





def main(): 
    unpack_if_needed()
    path_q = get_example_paths(n=0)
    
    path_q = get_test_paths()
    
    if (not os.path.isdir('outputs')): 
        os.makedirs('outputs')
        
    output_file = open('outputs/output.csv', 'w')
    output_file.write('ImageId,EncodedPixels\n')
    
    unique_img_id = set()
    lines = 0
    one_image =True
    while (not path_q.is_empty()):
        img_info = path_q.get_next()
        print(str(path_q.length))
        img_id, img, img_true_masks = load_next_image(img_info)   
        if (img_id in unique_img_id): 
            raise 
        unique_img_id.add(img_id)
        segmented_img, full_mask = segment_single_image(img)
        object_masks = split_into_single_object_masks(segmented_img)
        print("img # : " + str(path_q.index) + ' / ' + str(path_q.length) + ' ---> ' + str(path_q.index/path_q.length*100)[0:3] + ' %')
        for obj_mask in object_masks:
            #plt.imshow(obj_mask)
            #plt.show()
            #output_string = create_submission_line(img_id, obj_mask)
            rle = ''
            for i in (rle_encoding(obj_mask)): 
                rle += str(i) + ' '
            os2 = str(img_id) + ',' +  rle
#            if (output_string != os2): 
#                print(os2)
#                print('---mine-VVV---------')
#                print(output_string)
#                print('------------')
#                raise

            output_file.write(os2 + '\n')
            lines+=1
        #one_image = False
        
    print('finished, total # of lines (objects): ' + str(lines))
     


''' ---------------------------------------------------------------------------< 
Description: 
    This function takes a tuple of values representing an image disk path location
    and returns the images loaded into memory. 
    ------------
Inputs: 
    In1: img_info = (img_path, img_id, img_masks_path)
        img_path = [string] disk path 
        img_id = [string] image identifier 
        img_mask_paths = [list of strings] disk paths to nucleus masks
    ------------
Outputs:     
    Out1: img_id, test_img, mask_imgs
        img_id = [string] image identifier
        test_img = [np.array] loaded test image from given "img_path" 
        mask_imgs = [list of np.arrays] loaded mask images from given img_mask_paths 
''' 
# img_info = (img_path, img_id, img_masks_path)      
# returns tuple containing: img_id, test_img, [mask_imgs]
def load_next_image( img_info ): 
    test_img = misc.imread(img_info[0])
    mask_imgs = []
    for p in img_info[2]: 
        mask_imgs.append(misc.imread(p))
    #plt.imshow(test_img[:,:,1])
    #plt.show()
    return img_info[1], test_img, mask_imgs


''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
''' 
def split_into_single_object_masks(labeled_img): 
    vals = set()
    mask_imgs = []
    
    for i in labeled_img.flatten(): 
        if (i not in vals):
            vals.add(i)
    
    for label in vals: 
        new_img = labeled_img
        shp = new_img.shape
        new_img = new_img.flatten()
        if (label != 0): 
            for i in range(len(new_img)): 
                if (new_img[i] == label): 
                    new_img[i] = 1
                else: 
                    new_img[i] = 0
        
            new_img = new_img.reshape(shp)
            #print('label: ' + str(label))
            #plt.imshow(new_img)
            #plt.show()
            mask_imgs.append(new_img)
    return mask_imgs


''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
''' 
# takes in img 
# returns labeled image 
def segment_single_image(img):
    global sigma
    OFFSET = 0
    img = filters.gaussian(color.rgb2gray(img), sigma)
    
    mask = img > filters.threshold_otsu(img) + OFFSET
    print(np.array(mask).shape)
    
#    clean_border = segmentation.clear_border(mask)
#    plt.imshow(clean_border, cmap='gray')
#    plt.show()
    
    labels = measure.label(mask)
    print('labels')
#    plt.imshow(labels)
#    plt.show()
    
    return labels, mask


''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
''' 
#get next returns (img_path, img_id, mask_path)  -> mask_path will be [''] if a test image (none available )
class path_queue:      
    def __init__(self, img_path, img_ids, list_of_mask_paths=['']):
        self.index = 0
        self.img_path = img_path
        self.mask_paths = list_of_mask_paths*len(img_path) 
        self.img_ids = img_ids
        self.length = len(img_path)

        
    def get_next(self): 
        self.index+=1 
        if (self.index-1 < len(self.img_path)):
            return (self.img_path[self.index-1],self.img_ids[self.index-1], self.mask_paths[self.index-1])
        else: 
            print('no paths available')
            return []
    
    def is_empty(self): 
        if (self.index < self.length): 
            return False
        else: 
            return True
   

''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
'''     
# From https://www.kaggle.com/rakhlin/fast-run-length-encoding-python 
# Thanks rakhlin      
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths        
        
   
''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
'''      
# descrip 1 indexed flat image, start run_length 
#input:  img ID (string)
    #    mask img of items 
#output: returns a string of one image
def create_submission_line(img_ID, mask): 
    output_string = img_ID + ","
    flat_mask = np.array(mask).flatten(order='F')
    on = False
    start = int()
    for i, px in enumerate(flat_mask): 
        if (on): 
            if (not px):
                on = False 
                run_length = i-start
                output_string += str(run_length) + ' '
        else:
            if (px): 
                on = True
                start = i
                output_string += str(start + 1) + ' '
    
    return output_string
    

''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
'''     
# input
# n is the number the length of the list to return, how many examples 
# output 
# returns queue object, use .get_next() to get tuple (img path, list of mask paths)
def get_example_paths(n=0): 
    train_pth = "unpacked_datasets/train-data/"
    img_paths = []
    mask_paths = []
    img_ids=[]
    for i, example in enumerate(os.listdir(train_pth)): 
        if (n == 0 or i < n):
            img = train_pth + example + '/images'
            mask = train_pth + example + '/masks' 
            img_id=os.listdir(img)[0]
            img_pth = img + '/' + img_id
            mask_pths = list(map(lambda x: mask + '/' + x, os.listdir(mask)))
            img_paths.append(img_pth)
            mask_paths.append(mask_pths)
            img_ids.append(img_id)
        
    print('here --------------------------------------------------------')
    print(str(len(mask_paths)*len(mask_paths[0])))
    return path_queue(img_paths, img_id,  mask_paths)


''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
''' 
def get_test_paths(): 
    test_pth = "unpacked_datasets/test/"
    img_paths = []
    img_ids = []
    for i, example in enumerate(os.listdir(test_pth)): 
        img = test_pth + example + '/images'
        img_pth = img + '/' + example + '.png'
        img_paths.append(img_pth)
        img_ids.append(example)
        
    return path_queue(img_paths, img_ids)


''' ---------------------------------------------------------------------------< 
Description: 

    ------------
Inputs: 
    In1: 
    In2: 
    ------------
Outputs:     
    Out1
    
''' 
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
  

'''
--------------------- app run ------------------------
'''
          
if __name__ == '__main__' :
    main() 


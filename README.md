# Nuclei-Segmentation
2018 Data Science Bowl - https://www.kaggle.com/c/data-science-bowl-2018#description - Team Teddy

Requires opencv 
$ pip install opencv-python


This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations.

Each image is represented by an associated ImageId. Files belonging to an image are contained in a folder with this ImageId. Within this folder are two subfolders:

images contains the image file.
masks contains the segmented masks of each nucleus. This folder is only included in the training set. Each mask contains one nucleus. Masks are not allowed to overlap (no pixel belongs to two masks).
The second stage dataset will contain images from unseen experimental conditions. To deter hand labeling, it will also contain images that are ignored in scoring. The metric used to score this competition requires that your submissions are in run-length encoded format. Please see the evaluation page for details.

As with any human-annotated dataset, you may find various forms of errors in the data. You may manually correct errors you find in the training set. The dataset will not be updated/re-released unless it is determined that there are a large number of systematic errors. The masks of the stage 1 test set will be released with the release of the stage 2 test set.

File descriptions
/stage1_train/* - training set images (images and annotated masks)
/stage1_test/* - stage 1 test set images (images only, you are predicting the masks)
/stage2_test/* (released later) - stage 2 test set images (images only, you are predicting the masks)
stage1_sample_submission.csv - a submission file containing the ImageIds for which you must predict during stage 1
stage2_sample_submission.csv (released later) - a submission file containing the ImageIds for which you must predict during stage 2
stage1_train_labels.csv - a file showing the run-length encoded representation of the training images. This is provided as a convenience and is redundant with the mask image files. 


----------------


This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:
IoU(A,B)=A∩BA∪B.
IoU(A,B)=A∩BA∪B.
The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from from 0.5 to 0.95 with a step size of 0.05: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95). In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value tt, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:
TP(t)TP(t)+FP(t)+FN(t).
TP(t)TP(t)+FP(t)+FN(t).
A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:
1|thresholds|∑tTP(t)TP(t)+FP(t)+FN(t).
1|thresholds|∑tTP(t)TP(t)+FP(t)+FN(t).
Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.

Submission File
In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted nucleus segmentation for the given ImageId.

ImageId,EncodedPixels  
0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5,1 1  
0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac,1 1  
0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac,2 3 8 9  
etc...
Submission files may take several minutes to process due to the size.
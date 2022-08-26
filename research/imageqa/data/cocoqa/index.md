<title>Toronto COCO-QA Dataset</title>
<div class="ribbon"></div>

# Toronto COCO-QA Dataset

Reference: Mengye Ren, Ryan Kiros, Richard Zemel, "Exploring Models and Data 
for Image Question Answering", ArXiv preprint

Images: Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona,
Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick, "Microsoft COCO: Common 
Objects in Context", ECCV 2014.

-------------------------------------------------------------------------------

## Overview

* Automatically generated from image captions.
* 123287 images
* 78736 train questions
* 38948 test questions
* 4 types of questions: object, number, color, location
* Answers are all one-word.

-------------------------------------------------------------------------------

## Contact

Please contact [Mengye Ren](http://www.cs.toronto.edu/~mren), if you have any 
questions or concerns about this dataset.

-------------------------------------------------------------------------------

## Downloads

[[zip](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip)] (~2MB) (Updated: **May 17, 2015**)

-------------------------------------------------------------------------------

## Files
The zip file contains two folders: *train* and *test*. Each folder has the 
following files:

* *img_ids.txt*: Each line contains an original image ID in the MS-COCO dataset
* *questions.txt*: Each line contains a question, in plain text
* *answers.txt*: Each line contains an answer to the question, in plain text
* *types.txt*: Each line contains an integer denoting the question type: 
0 -> *object*, 1 -> *number*, 2 -> *color*, 3 -> *location*

<div class="ribbon"></div>

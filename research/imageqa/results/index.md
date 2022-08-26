<title>Image Question Answering</title>
<div class="ribbon"></div>

# Image Question Answering Full Results

Reference: Mengye Ren, Ryan Kiros, Richard Zemel, "Exploring Models and Data
for Image Question Answering", NIPS 2015.

-------------------------------------------------------------------------------

## DAQUAR-37

### Dataset

References:

QAs: Mateusz Malinowski, Mario Fritz, "Towards a Visual Turing Challenge", NIPS
2014 Workshop on Learning Semantics. [[ArXiv](http://arxiv.org/abs/1410.8027)]
[[link](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-
computing/research/vision-and-language/visual-turing-challenge/)]

Images: Nathan Silberman, Pushmeet Kohli, Derek Hoiem, Rob Fergus, "Indoor 
Segmentation and Support Inference from RGBD Images", ECCV 2012.
[[link](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)]

Notes:

1. Here we are only using DAQUAR-37 with one-word answers, a subset of the 37 
object classes dataset.
2. Only test set results are rendered in the links below.

### Individual Models

* [2-VIS+BLSTM](dq-2-vis-blstm/0.html)
* [VIS+LSTM](dq-vis-lstm/0.html)
* [IMG+BOW](dq-img-bow/0.html)
* [LSTM](dq-lstm/0.html)
* [BOW](dq-bow/0.html)

### Model Comparison

* [2-VIS+BLSTM vs VIS+LSTM vs LSTM](dq-2-vis-blstm_vs_vis-lstm_vs_lstm)

-------------------------------------------------------------------------------

## Toronto COCO-QA

QAs: [[link](../data/cocoqa)]

Images: Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona,
Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick, "Microsoft COCO: Common 
Objects in Context", ECCV 2014.

Notes:

1. All images are hosted on Flickr, and some links may not be available 
anymore.
2. Only test set results are rendered in the links below.

### Individual Models

* [2-VIS+BLSTM](cq-2-vis-blstm/0.html)
* [VIS+LSTM](cq-vis-lstm/0.html)
* [IMG+BOW](cq-img-bow/0.html)
* [IMG+PRIOR](cq-img-prior/0.html)
* [IMG](cq-img/0.html)
* [LSTM](cq-lstm/0.html)
* [BOW](cq-bow/0.html)

### Model Comparison

* [IMG+BOW vs 2-VIS+BLSTM vs IMG+PRIOR vs BOW](cq_img-bow_vs_2-vis-blstm_vs_img
-prior_vs_bow)

<div class="ribbon"></div>

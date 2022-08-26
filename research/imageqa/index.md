<title>Image Question Answering</title>
<div class="ribbon"></div>

<h1>Exploring Models and Data for Image Question Answering</h1>

<!--<div class="author">
<p>-->
Mengye Ren<sup>1</sup>, Ryan Kiros<sup>1</sup>, Richard S.
Zemel<sup>1,2</sup><br />
<br />
<sup>1</sup>Department of Computer Science, University of Toronto, Toronto ON, CANADA<br />
<sup>2</sup>Canadian Institute for Advanced Research, Toronto ON, CANADA<br />
<!--</p>
</div>-->
<br/>
<img class="paper-fig" src="img/fig1.png" />

## Abstract
This work aims to address the problem of image-based question-answering (QA)
with new models and datasets. In our work, we propose to use neural networks
and visual semantic embeddings, without intermediate stages such as object
detection and image segmentation, to predict answers to simple questions about
images. Our model performs 1.8 times better than the only published results on
an existing image QA dataset. We also present a question generation algorithm
that converts image descriptions, which are widely available, into QA form. We
used this algorithm to produce an order-of-magnitude larger dataset, with more
evenly distributed answers. A suite of baseline results on this new dataset are
also presented.

-------------------------------------------------------------------------------

## Full Paper
<!-- <img class="paper-snap" src="img/full.png" /> -->
[[pdf](papers/imageqa_nips2015.pdf)]

-------------------------------------------------------------------------------

## Supplementary Materials
<!-- <img class="paper-snap" src="img/supp.png" /> -->
[[pdf](papers/imageqa_supplementary_nips2015.pdf)]

-------------------------------------------------------------------------------

## Dataset
[[link](data/cocoqa)]

-------------------------------------------------------------------------------

## Full results
[[link](results)]

-------------------------------------------------------------------------------

## Code
* To reproduce experimental results:
[[link](https://github.com/renmengye/imageqa-public)]
* To generate questions:
[[link](https://github.com/renmengye/imageqa-qgen)]

-------------------------------------------------------------------------------

## Cite
<pre>
<code>
@inproceedings{ren2015imageqa,
  title={Exploring Models and Data for Image Question Answering},
  author={Mengye Ren and Ryan Kiros and Richard Zemel},
  booktitle={NIPS},
  year={2015}
}
</code>
</pre>
<div class="ribbon"></div>

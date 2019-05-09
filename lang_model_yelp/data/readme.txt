Yelp15 data set used in the following paper:

@article{yang2017improved,
  title={Improved Variational Autoencoders for Text Modeling using Dilated
  Convolutions},
    author={Yang, Zichao and Hu, Zhiting and Salakhutdinov, Ruslan and
    Berg-Kirkpatrick, Taylor},
    journal={arXiv preprint arXiv:1702.08139},
    year={2017}
    }

DESCRIPTION

This data set is constructed from Yelp15 from:
Duyu Tang, Bing Qin, Ting Liu.
Document Modeling with Gated Recurrent Neural Network
for Sentiment Classification. EMNLP 2017.
http://ir.hit.edu.cn/~dytang/paper/emnlp2015/emnlp-2015-data.7z

The original dataset contains 1.2 milliion training samples and 156k validation
and testing samples. We sample 100k as training, 10k as validation and 10k as
testing from the respective sets.

We use a vocabulary size of 20k and replace out of vocabulary tokens as _UNK.

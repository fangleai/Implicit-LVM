The yelp transfer data set is same as

@inproceedings{shen2017style,
	Author = {Shen, Tianxiao and Lei, Tao and Barzilay, Regina and Jaakkola, Tommi},
	Booktitle = {Advances in neural information processing systems},
	Date-Added = {2019-04-14 05:38:33 -0400},
	Date-Modified = {2019-04-14 05:38:33 -0400},
	Pages = {6830--6841},
	Title = {Style transfer from non-parallel text by cross-alignment},
	Year = {2017}}

0 for negative sentiment, 1 for positive sentiment

The data files include
train0.txt
train1.txt
valid0.txt
valid1.txt
test0.txt
test1.txt
test.txt
data.txt: for training text classification model through fasttext library (https://pypi.org/project/fasttext/).
          It combines all labeled data in train, valid and test txt files. After combining those,
          we get 25K negative sentences and 35K positive sentences. We duplicate the first 10K negative sentences,
          and finally reach a balanced 35K vs 35K dataset.
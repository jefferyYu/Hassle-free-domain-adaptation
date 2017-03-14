Jianfei YU
jfyu.2014@phdis.smu.edu.sg
1 September, 2015

Code for:

A Hassle-Free Unsupervised Domain Adaptation Method Using Instance Similarity Features
ACL 2015
https://aclweb.org/anthology/P/P15/P15-2028.pdf

You can run the model for any cross-domain classification tasks.

Runs on Python 2.7 and require the package scikit-learn

1. Instructions
We have attached part of the spam filtering dataset, where where train.feat is the source domain data, and u01.feat is the target domain data.
You can directly run our codes

python ISF_sparse.py 

2. How to apply it to your own task?
If you want to use our codes for your own classification task, you should first preprocess your datasets, and save their files into the standard format (like the input of libSVM).
In our codes, we use a simple demo dataset as an example, where train.feat is the source domain data, and u01.feat is the target domain data.
You can change the source domain and target domain data, but the format should be the same with them.

Acknowledgements
Using this code means you have read and accepted the copyrights set by the dataset providers.

License:
Singapore Management University

